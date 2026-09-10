"""
Process Video Application Service

Thin application service orchestrating AudioPipeline and VisualPipeline independently.
Clean Architecture: zero imports of src.infrastructure or third-party packages.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Optional

from src.application.request import ProcessingRequest
from src.application.result import ProcessingResult
from src.core.contracts import VideoUtilPort
from src.core.errors import ValidationError
from src.features.audio.pipeline import AudioPipeline
from src.features.visual.pipeline import VisualPipeline

logger = logging.getLogger(__name__)


class ProcessVideoUseCase:
    def __init__(
        self,
        audio_pipeline: Optional[AudioPipeline] = None,
        visual_pipeline: Optional[VisualPipeline] = None,
        video_util: Optional[VideoUtilPort] = None,
        workspace_base_dir: str = "outputs/jobs",
    ):
        self.audio_pipeline = audio_pipeline or AudioPipeline()
        self.visual_pipeline = visual_pipeline or VisualPipeline()
        self.video_util = video_util
        self.workspace_base_dir = Path(workspace_base_dir)

    def execute(self, request: ProcessingRequest) -> ProcessingResult:
        start_time = time.monotonic()
        job_id = request.job_id or f"job_{int(time.time())}_{os.urandom(4).hex()}"
        timings = {}
        warnings = []

        # Validate before creating job/output artifacts whenever possible.
        input_path = Path(request.input_video).resolve()
        if not input_path.is_file():
            raise ValidationError(f"Input video file not found: {input_path}")
        output_path = Path(request.output_video).resolve()
        if output_path.exists() and not request.overwrite:
            raise ValidationError(f"Output video already exists: {output_path}")
        if self.video_util is None:
            raise ValidationError("VideoUtilPort (video_util) must be injected into ProcessVideoUseCase.")

        job_dir = (self.workspace_base_dir / job_id).resolve()
        job_dir.mkdir(parents=True, exist_ok=True)
        (job_dir / "logs").mkdir(parents=True, exist_ok=True)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self._atomic_json(job_dir / "job.json", {
            "job_id": job_id,
            "request": request.to_dict(),
            "created_at": time.time(),
        })

        # 2. Execute Visual Pipeline
        visual_result = None
        if request.visual.enabled:
            t0 = time.monotonic()
            request.visual.video_path = str(input_path)
            rendered_temp_path = str(job_dir / "visual_rendered.mp4")
            visual_result = self.visual_pipeline.process(
                request=request.visual,
                job_workspace_dir=str(job_dir),
                output_video_path=rendered_temp_path,
            )
            timings["visual_pipeline"] = round(time.monotonic() - t0, 3)

        # 3. Execute Audio Pipeline
        audio_result = None
        if request.audio.enabled:
            t0 = time.monotonic()
            request.audio.audio_path = str(input_path)
            audio_result = self.audio_pipeline.process(
                request=request.audio,
                job_workspace_dir=str(job_dir),
            )
            timings["audio_pipeline"] = round(time.monotonic() - t0, 3)

        # 4. Stream Muxing & Output Assembly
        t0 = time.monotonic()
        visual_video = visual_result.rendered_video_path if (visual_result and visual_result.rendered_video_path) else str(input_path)
        audio_source = audio_result.dubbed_audio_path if (audio_result and audio_result.dubbed_audio_path) else str(input_path)

        final_video_path = self.video_util.mux_audio_video(
            video_no_audio_path=visual_video,
            audio_source_path=audio_source,
            output_path=str(output_path),
        )
        probe_info = self.video_util.probe_video(final_video_path)

        timings["stream_muxing"] = round(time.monotonic() - t0, 3)

        total_duration = time.monotonic() - start_time
        timings["total"] = round(total_duration, 3)

        audio_status = audio_result.status if audio_result else "SKIPPED"
        visual_status = visual_result.status if visual_result else "SKIPPED"
        warnings.extend(audio_result.warnings if audio_result else [])
        warnings.extend(visual_result.warnings if visual_result else [])
        overall_status = self._aggregate_status(audio_status, visual_status)

        result = ProcessingResult(
            job_id=job_id,
            status=overall_status,
            input_video_path=str(input_path),
            output_video_path=final_video_path,
            duration_sec=probe_info["duration"],
            fps=probe_info["fps"],
            resolution=f"{probe_info['width']}x{probe_info['height']}",
            audio_status=audio_status,
            visual_status=visual_status,
            detections_summary=visual_result.detections_summary if visual_result else {},
            tracks_summary=visual_result.tracks_summary if visual_result else {},
            workspace_dir=str(job_dir),
            timings=timings,
            warnings=warnings,
            provenance={
                "input": str(input_path),
                "audio_artifact": audio_source,
                "visual_artifact": visual_video,
                "final_artifact": final_video_path,
            },
        )
        self._atomic_json(job_dir / "result.json", result.to_dict())
        return result

    @staticmethod
    def _aggregate_status(*statuses: str) -> str:
        active = {status for status in statuses if status != "SKIPPED"}
        for status in ("BLOCKED", "FAILED", "PARTIAL"):
            if status in active:
                return status
        return "SUCCESS"

    @staticmethod
    def _atomic_json(path: Path, payload: object) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        handle, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as stream:
                json.dump(payload, stream, ensure_ascii=False, indent=2)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp_name, path)
        except Exception:
            try:
                os.unlink(temp_name)
            except OSError:
                pass
            raise
