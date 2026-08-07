"""
Media Processing Service (Unified Production Application Boundary)

Orchestrates the policy-driven video processing pipeline with job workspace isolation:
YOLO 4-class Detection -> Class-Aware Tracker -> Action Policy Engine -> RenderingPlan -> Lazy Enrichment -> Action Video Renderer -> Output Validation.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from src.application.dto.processing_request import ProcessingRequest, ProcessingResult
from src.application.services.action_policy_engine import ActionPolicyEngine
from src.domain.entities.canonical_detection import CanonicalDetection, ClassAwareTrack
from src.domain.entities.processing_policy import ProcessingPolicyConfig
from src.domain.services.class_aware_tracker import ClassAwareTracker
from src.infrastructure.benchmark.hybrid_subtitle.yolo_proposal import YoloTextProposalProvider
from src.infrastructure.rendering.action_video_renderer import ActionVideoRenderer
from src.infrastructure.services.lazy_enrichment_service import LazyEnrichmentService

logger = logging.getLogger(__name__)


class MediaProcessingService:
    def __init__(
        self,
        weights_path: str = "models/yolo/best.pt",
        font_path: Optional[str] = None,
        base_workspace_dir: str = "output/jobs",
    ):
        self.weights_path = weights_path
        self.font_path = font_path
        self.base_workspace_dir = Path(base_workspace_dir)
        self.renderer = ActionVideoRenderer(font_path=font_path)

    def process_video(self, request: ProcessingRequest) -> ProcessingResult:
        start_time = time.monotonic()
        job_id = request.job_id or f"job_{int(time.time())}_{os.urandom(4).hex()}"
        timings: Dict[str, float] = {}
        warnings: List[str] = []

        # 1. Job Workspace Isolation
        job_dir = (self.base_workspace_dir / job_id).resolve()
        job_dir.mkdir(parents=True, exist_ok=True)
        logs_dir = job_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)

        input_path = Path(request.input_video).resolve()
        if not input_path.exists():
            raise FileNotFoundError(f"Input video file not found: {input_path}")

        output_path = Path(request.output_video).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if output_path.exists() and not request.overwrite:
            raise FileExistsError(f"Output video already exists (set overwrite=True to replace): {output_path}")

        # Save job.json
        job_json_path = job_dir / "job.json"
        with open(job_json_path, "w", encoding="utf-8") as f:
            json.dump({
                "job_id": job_id,
                "request": request.to_dict(),
                "created_at": time.time(),
            }, f, indent=2, ensure_ascii=False)

        # 2. Resolve Policy
        policy = request.resolve_policy()

        # 3. CUDA / Device Check
        t0 = time.monotonic()
        if request.device in ("cuda", "auto") and torch.cuda.is_available():
            effective_device = "cuda"
        else:
            effective_device = "cpu"

        # 4. YOLO 4-class Detection (Single-pass video decode)
        logger.info("[1/5] Running YOLO 4-class Detection on %s (device=%s)...", input_path.name, effective_device)
        proposal_provider = YoloTextProposalProvider(
            weights_path=self.weights_path,
            conf=0.05,
            iou=0.70,
            imgsz=1024,
            device=effective_device,
        )

        bundle = proposal_provider.detect_video(str(input_path), stride=1)
        timings["detection"] = round(time.monotonic() - t0, 3)

        # Convert proposal bundle detections to CanonicalDetections preserving class_id and class_name
        canonical_dets: List[CanonicalDetection] = []
        class_counts: Dict[str, int] = {}

        for frame in bundle.frames:
            for det in frame.detections:
                cid = int(det.get("class_id", 0))
                cname = str(det.get("class_name") or "subtitle")
                class_counts[cname] = class_counts.get(cname, 0) + 1

                canonical_dets.append(
                    CanonicalDetection(
                        class_id=cid,
                        class_name=cname,
                        confidence=float(det.get("confidence", 0.0)),
                        bbox_xyxy=tuple(det["bbox_xyxy"]),  # type: ignore
                        bbox_normalized=tuple(det["bbox_normalized"]),  # type: ignore
                        frame_index=frame.frame_index,
                        timestamp_sec=frame.timestamp_sec,
                        candidate_id=det.get("candidate_id", ""),
                    )
                )

        # Save detections.json
        detections_json_path = job_dir / "detections.json"
        with open(detections_json_path, "w", encoding="utf-8") as f:
            json.dump([d.to_dict() for d in canonical_dets], f, indent=2, ensure_ascii=False)

        # 5. Class-Aware Spatial-Temporal Tracking
        t0 = time.monotonic()
        logger.info("[2/5] Running Class-Aware Tracking across %d detections...", len(canonical_dets))
        tracker = ClassAwareTracker(
            horizontal_gap_ratio=policy.horizontal_gap_ratio,
            track_iou_threshold=policy.tracking_iou,
            max_gap_frames=policy.max_gap_frames,
            video_width=bundle.video_width,
            video_height=bundle.video_height,
        )
        tracks = tracker.track(canonical_dets)
        timings["tracking"] = round(time.monotonic() - t0, 3)

        track_summary: Dict[str, int] = {}
        for tr in tracks:
            track_summary[tr.class_name] = track_summary.get(tr.class_name, 0) + 1

        # Save tracks.json
        tracks_json_path = job_dir / "tracks.json"
        with open(tracks_json_path, "w", encoding="utf-8") as f:
            json.dump([tr.to_dict() for tr in tracks], f, indent=2, ensure_ascii=False)

        # 6. Action Policy Engine -> RenderingPlan
        t0 = time.monotonic()
        logger.info("[3/5] Evaluating Action Policy Engine across %d tracks...", len(tracks))
        policy_engine = ActionPolicyEngine(policy)
        rendering_plan = policy_engine.create_rendering_plan(
            job_id=job_id,
            tracks=tracks,
            video_width=bundle.video_width,
            video_height=bundle.video_height,
            duration=bundle.total_frames / bundle.fps if bundle.fps > 0 else 0.0,
            fps=bundle.fps,
        )
        timings["policy_engine"] = round(time.monotonic() - t0, 3)

        # Export output/jobs/<job_id>/action_plan.json
        action_plan_path = job_dir / "action_plan.json"
        rendering_plan.save_json(action_plan_path)

        # 7. Lazy Enrichment Service
        t0 = time.monotonic()
        logger.info("[4/5] Executing Lazy Enrichment Service...")
        lazy_service = LazyEnrichmentService(
            translation_provider_name=request.translation_provider,
            ocr_provider_name=request.ocr_provider,
            speech_provider_name=request.speech_provider,
            target_language=request.target_language,
        )
        rendering_plan = lazy_service.enrich_plan(
            rendering_plan=rendering_plan,
            tracks=tracks,
            input_video_path=str(input_path),
            device=effective_device,
        )
        timings["lazy_enrichment"] = round(time.monotonic() - t0, 3)

        # 8. Action Video Renderer & Audio Mux
        t0 = time.monotonic()
        logger.info("[5/5] Rendering Video & Muxing Audio to %s...", output_path.name)
        final_video_path = self.renderer.render(
            input_video_path=str(input_path),
            rendering_plan=rendering_plan,
            output_video_path=str(output_path),
            ffmpeg_encoder="auto",
        )
        timings["render"] = round(time.monotonic() - t0, 3)

        total_duration = time.monotonic() - start_time
        timings["total"] = round(total_duration, 3)

        # 9. Output Validation
        val_info = self._validate_video_output(str(input_path), final_video_path)

        return ProcessingResult(
            job_id=job_id,
            status="SUCCESS",
            input_video_path=str(input_path),
            output_video_path=final_video_path,
            duration_sec=val_info["duration"],
            fps=val_info["fps"],
            resolution=f"{val_info['width']}x{val_info['height']}",
            detections_summary=class_counts,
            tracks_summary=track_summary,
            actions_summary=rendering_plan.summary(),
            action_plan_path=str(action_plan_path),
            workspace_dir=str(job_dir),
            timings=timings,
            warnings=warnings,
        )

    def _validate_video_output(self, input_path: str, output_path: str) -> Dict[str, Any]:
        cmd = [
            "ffprobe",
            "-v", "error",
            "-show_entries", "stream=codec_type,width,height,r_frame_rate,duration",
            "-show_entries", "format=duration",
            "-of", "json",
            output_path,
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(res.stdout)

        v_stream = next((s for s in data.get("streams", []) if s.get("codec_type") == "video"), {})
        duration = float(v_stream.get("duration") or data.get("format", {}).get("duration", 0.0))
        width = int(v_stream.get("width", 0))
        height = int(v_stream.get("height", 0))
        fps_str = v_stream.get("r_frame_rate", "30/1")
        if "/" in fps_str:
            n, d = map(float, fps_str.split("/"))
            fps = n / d if d > 0 else 30.0
        else:
            fps = float(fps_str)

        return {"duration": duration, "width": width, "height": height, "fps": fps}
