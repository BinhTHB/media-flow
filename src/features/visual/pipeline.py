"""
Independent Visual Pipeline

Executes YOLO 4-class detection, Class-Aware Tracking, Class Routing, Subtitle/Logo/Watermark/Other Processors,
and Video Rendering.
Independent from AudioPipeline: zero audio/ASR/TTS imports.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.core.contracts import DetectorPort, SceneBoundaryPort
from src.features.visual.subtitle.region_planner import SubtitleRegionPlanner, same_lane, union_boxes
from src.core.models.detection import Detection
from src.core.models.subtitle_event import SubtitleEvent
from src.core.models.track import Track
from src.features.visual.processors.logo import LogoProcessor
from src.features.visual.processors.other_text import OtherTextProcessor
from src.features.visual.processors.watermark import WatermarkProcessor
from src.features.visual.subtitle.processor import SubtitleProcessor
from src.features.visual.tracker import ClassAwareTracker

logger = logging.getLogger(__name__)

VALID_SUBTITLE_MODES = {"translate_replace", "blur", "preserve"}


@dataclass
class VisualRequest:
    enabled: bool = True
    video_path: str = ""
    subtitle_enabled: bool = True
    subtitle_mode: str = "translate_replace"  # translate_replace | blur | preserve
    subtitle_layout: str = "temporal"  # temporal | legacy_centered
    subtitle_background: str = "blur"
    target_language: str = "vi"
    logo_enabled: bool = True
    logo_action: str = "blur"
    logo_image_path: Optional[str] = None
    watermark_enabled: bool = True
    watermark_action: str = "blur"
    other_text_enabled: bool = True
    other_text_action: str = "blur"


@dataclass
class VisualResult:
    enabled: bool
    status: str  # SUCCESS | SKIPPED | FAILED
    rendered_video_path: Optional[str] = None
    detections_summary: Dict[str, int] = field(default_factory=dict)
    tracks_summary: Dict[str, int] = field(default_factory=dict)
    subtitle_events: List[SubtitleEvent] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    timings: Dict[str, float] = field(default_factory=dict)


class VisualPipeline:
    def __init__(
        self,
        detector: Optional[DetectorPort] = None,
        tracker: Optional[ClassAwareTracker] = None,
        subtitle_processor: Optional[SubtitleProcessor] = None,
        logo_processor: Optional[LogoProcessor] = None,
        watermark_processor: Optional[WatermarkProcessor] = None,
        other_text_processor: Optional[OtherTextProcessor] = None,
        scene_detector: Optional[SceneBoundaryPort] = None,
    ):
        self.detector = detector
        self.tracker = tracker or ClassAwareTracker()
        self.subtitle_processor = subtitle_processor
        self.logo_processor = logo_processor
        self.watermark_processor = watermark_processor
        self.other_text_processor = other_text_processor
        self.scene_detector = scene_detector
        self._region_planner = None

    @staticmethod
    def _validate_request(request: VisualRequest) -> None:
        if request.subtitle_enabled and request.subtitle_mode not in VALID_SUBTITLE_MODES:
            allowed = ", ".join(sorted(VALID_SUBTITLE_MODES))
            raise ValueError(
                f"Unsupported subtitle_mode {request.subtitle_mode!r}; expected one of: {allowed}."
            )

    def _apply_subtitle_actions(
        self,
        frame: Any,
        frame_index: int,
        request: VisualRequest,
        detections_by_frame: Dict[int, List[Any]],
        events_by_frame: Dict[int, List[SubtitleEvent]],
    ) -> Any:
        """Apply the selected subtitle policy without destructive fallback."""
        if not request.subtitle_enabled or request.subtitle_mode == "preserve":
            return frame

        if self.subtitle_processor is None:
            has_work = bool(detections_by_frame.get(frame_index)) or bool(
                events_by_frame.get(frame_index)
            )
            if has_work:
                raise RuntimeError(
                    "SubtitleProcessor is required for subtitle blur/replacement."
                )
            return frame

        if request.subtitle_mode == "blur":
            for detection in detections_by_frame.get(frame_index, []):
                bbox = self.subtitle_processor.get_source_suppression_bbox(
                    detection.bbox, pad_x=18, pad_y=14
                )
                frame = self.subtitle_processor.blur_action.execute(
                    frame, bbox, padding_px=0
                )
            return frame

        ready_events = [
            event
            for event in events_by_frame.get(frame_index, [])
            if event.replacement_status == "READY"
            and event.blur_applied
            and event.overlay_applied
        ]
        if ready_events and self._region_planner is not None:
            # Keep independent lanes separate; select one semantic caption only
            # within the same spatial group.
            groups = []
            for event in ready_events:
                anchor = self.subtitle_processor.get_event_bbox(event, frame_index)
                region = self._region_planner.region_for(frame_index, anchor)
                group = next((g for g in groups if same_lane(g[0], region)), None)
                if group is None:
                    groups.append([region, [event]])
                else:
                    group[0] = union_boxes([group[0], region])
                    group[1].append(event)
            for region, events in groups:
                current = [d.bbox for d in detections_by_frame.get(frame_index, []) if same_lane(region, d.bbox)]
                region = union_boxes([region, *current])
                chosen = self.subtitle_processor.select_render_events(events)
                frame = self.subtitle_processor.render_region_events(frame, chosen, region)
            return frame

        if ready_events:
            # Remove every detected/active subtitle row in the current frame first
            # (including stacked bilingual duplicates). Event bboxes cover frames
            # where YOLO momentarily misses one row; detections preserve exact
            # geometry when present.
            suppression_regions = list(detections_by_frame.get(frame_index, []))
            get_event_bbox = getattr(self.subtitle_processor, "get_event_bbox", None)
            for event in ready_events:
                suppression_regions.append(
                    get_event_bbox(event, frame_index)
                    if callable(get_event_bbox)
                    else event.anchor_bbox
                )
            frame = self.subtitle_processor.suppress_source_subtitles(
                frame, suppression_regions
            )
            ready_events = self.subtitle_processor.select_render_events(ready_events)
            frame = self.subtitle_processor.render_frame_events(
                frame, frame_index, ready_events
            )
        return frame

    def process(self, request: VisualRequest, job_workspace_dir: str, output_video_path: str) -> VisualResult:
        start_time = time.monotonic()
        timings = {}
        warnings = []

        if not request.enabled:
            logger.info("VisualPipeline is disabled in request. Skipping execution.")
            return VisualResult(
                enabled=False,
                status="SKIPPED",
                warnings=["Visual processing disabled in request."],
            )

        self._validate_request(request)

        if self.detector is None:
            raise RuntimeError("YOLO Detector is not configured for enabled VisualPipeline.")

        ws_dir = Path(job_workspace_dir)
        visual_dir = ws_dir / "visual"
        visual_dir.mkdir(parents=True, exist_ok=True)

        # 1. YOLO 4-class Detection
        t0 = time.monotonic()
        detections = self.detector.detect_video(request.video_path, stride=1)
        timings["detection"] = round(time.monotonic() - t0, 3)

        class_counts: Dict[str, int] = {}
        for d in detections:
            class_counts[d.class_name] = class_counts.get(d.class_name, 0) + 1

        # 2. Class-Aware Tracking
        t0 = time.monotonic()
        tracks = self.tracker.track(detections)
        timings["tracking"] = round(time.monotonic() - t0, 3)

        track_summary: Dict[str, int] = {}
        for tr in tracks:
            track_summary[tr.class_name] = track_summary.get(tr.class_name, 0) + 1

        # Route tracks by class
        subtitle_tracks = [t for t in tracks if t.class_name == "subtitle"]
        logo_tracks = [t for t in tracks if t.class_name == "logo"]
        watermark_tracks = [t for t in tracks if t.class_name == "watermark"]
        other_tracks = [t for t in tracks if t.class_name == "other_burn_in_text"]

        self._region_planner = None
        if request.subtitle_enabled and request.subtitle_mode == 'translate_replace' and request.subtitle_layout == 'temporal':
            import json
            cuts = self.scene_detector.detect_cuts(request.video_path) if self.scene_detector else []
            self._region_planner = SubtitleRegionPlanner(detections, cuts)
            (visual_dir / 'subtitle_regions.json').write_text(json.dumps(self._region_planner.to_dict(), indent=2), encoding='utf-8')

        # 3. Subtitle Processing
        t0 = time.monotonic()
        subtitle_events = []
        if (
            request.subtitle_enabled
            and request.subtitle_mode == "translate_replace"
            and subtitle_tracks
        ):
            if self.subtitle_processor is None:
                raise RuntimeError(
                    "SubtitleProcessor is required for translate_replace mode."
                )
            subtitle_events = self.subtitle_processor.process_tracks(
                tracks=subtitle_tracks,
                target_language=request.target_language,
                video_path=request.video_path,
            )
        timings["subtitle_processing"] = round(time.monotonic() - t0, 3)

        # Persist subtitle-event audit so a detector-only success can never be
        # mistaken for a successful replacement render. This is especially
        # important because translate_replace is intentionally fail-safe and
        # preserves source pixels when OCR/translation is unavailable.
        if request.subtitle_enabled and request.subtitle_mode == "translate_replace":
            import json
            audit_rows = []
            status_counts: Dict[str, int] = {}
            for ev in subtitle_events:
                status_counts[ev.replacement_status] = status_counts.get(ev.replacement_status, 0) + 1
                audit_rows.append({
                    "event_id": ev.event_id,
                    "track_id": ev.track_id,
                    "start_time": ev.start_time,
                    "end_time": ev.end_time,
                    "replacement_status": ev.replacement_status,
                    "source_text": ev.source_text,
                    "translated_text": ev.translated_text,
                    "blur_applied": ev.blur_applied,
                    "overlay_applied": ev.overlay_applied,
                    "split_reason": ev.split_reason,
                    "anchor_bbox": [ev.anchor_bbox.x1, ev.anchor_bbox.y1, ev.anchor_bbox.x2, ev.anchor_bbox.y2],
                })
            (visual_dir / "subtitle_events.json").write_text(
                json.dumps({
                    "subtitle_detection_count": class_counts.get("subtitle", 0),
                    "subtitle_track_count": len(subtitle_tracks),
                    "event_count": len(subtitle_events),
                    "status_counts": status_counts,
                    "events": audit_rows,
                }, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        # Initialize Class Processors
        logo_proc = self.logo_processor or LogoProcessor(action_name=request.logo_action)
        watermark_proc = self.watermark_processor or WatermarkProcessor(action_name=request.watermark_action)
        other_proc = self.other_text_processor or OtherTextProcessor(action_name=request.other_text_action)

        # 4. Video Frame Rendering
        t0 = time.monotonic()
        import cv2

        cap = cv2.VideoCapture(request.video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video for visual rendering: {request.video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # VisualPipeline owns pixels only. It deliberately emits a video-only
        # artifact; final audio muxing belongs to ProcessVideoUseCase.
        Path(output_video_path).parent.mkdir(parents=True, exist_ok=True)
        visual_only_path = str(Path(output_video_path).resolve())
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(visual_only_path, fourcc, fps, (width, height))

        frame_idx = 0
        events_by_frame: Dict[int, List[SubtitleEvent]] = {}
        ready_events = sorted(
            [ev for ev in subtitle_events if ev.replacement_status == "READY"],
            key=lambda ev: ev.start_time,
        )
        for index, ev in enumerate(ready_events):
            coarse_start = int(round(ev.start_time * fps))
            coarse_end = int(round(ev.end_time * fps))
            expected_frames = max(1, coarse_end - coarse_start + 1)
            continuity = min(1.0, len(ev.bbox_trajectory) / float(expected_frames))
            if continuity >= 0.85:
                pre_roll, post_roll = 1, 2
            elif continuity >= 0.60:
                pre_roll, post_roll = 2, 3
            else:
                pre_roll, post_roll = 3, 4

            start_f = max(0, coarse_start - pre_roll)
            end_f = coarse_end + post_roll

            if index > 0:
                prev = ready_events[index - 1]
                prev_end = int(round(prev.end_time * fps))
                if prev.track_id == ev.track_id:
                    start_f = max(start_f, prev_end + 1)
            if index + 1 < len(ready_events):
                nxt = ready_events[index + 1]
                next_start = int(round(nxt.start_time * fps))
                if nxt.track_id == ev.track_id:
                    end_f = min(end_f, next_start - 1)

            for f_i in range(start_f, max(start_f, end_f) + 1):
                events_by_frame.setdefault(f_i, []).append(ev)

        sub_dets_by_frame: Dict[int, List[Any]] = {}
        for tr in subtitle_tracks:
            for det in tr.detections:
                sub_dets_by_frame.setdefault(det.frame_index, []).append(det)

            # Backward-fill safety: If track starts at frame N, fill preceding frames (~25 frames / 0.8s) if unmapped
            if tr.detections:
                first_det = tr.detections[0]
                start_idx = first_det.frame_index
                backward_fill_count = int(round(fps * 0.85))  # ~25 frames
                for b_idx in range(max(0, start_idx - backward_fill_count), start_idx):
                    if b_idx not in sub_dets_by_frame:
                        sub_dets_by_frame.setdefault(b_idx, []).append(first_det)

        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            # A. Process non-subtitle visual processors
            if request.logo_enabled:
                frame = logo_proc.process_frame_tracks(frame, frame_idx, logo_tracks)
            if request.watermark_enabled:
                frame = watermark_proc.process_frame_tracks(frame, frame_idx, watermark_tracks)
            if request.other_text_enabled:
                frame = other_proc.process_frame_tracks(frame, frame_idx, other_tracks)

            # B. Subtitle policy: preserve, blur-only, or fail-safe translation.
            frame = self._apply_subtitle_actions(
                frame,
                frame_idx,
                request,
                sub_dets_by_frame,
                events_by_frame,
            )

            writer.write(frame)
            frame_idx += 1

        cap.release()
        writer.release()

        timings["rendering"] = round(time.monotonic() - t0, 3)
        timings["total"] = round(time.monotonic() - start_time, 3)

        visual_status = "SUCCESS"
        if request.subtitle_enabled and request.subtitle_mode == "translate_replace" and subtitle_tracks:
            ready_count = sum(1 for ev in subtitle_events if ev.replacement_status == "READY" and ev.overlay_applied)
            if ready_count == 0:
                visual_status = "FAILED"
                warnings.append(
                    f"Detected {class_counts.get('subtitle', 0)} subtitle boxes across "
                    f"{len(subtitle_tracks)} tracks, but 0 replacement events were READY; "
                    "source pixels were preserved. Inspect visual/subtitle_events.json."
                )
            elif ready_count < len(subtitle_events):
                warnings.append(
                    f"Subtitle replacement partial: {ready_count}/{len(subtitle_events)} events READY."
                )

        return VisualResult(
            enabled=True,
            status=visual_status,
            rendered_video_path=visual_only_path,
            detections_summary=class_counts,
            tracks_summary=track_summary,
            subtitle_events=subtitle_events,
            warnings=warnings,
            timings=timings,
        )
