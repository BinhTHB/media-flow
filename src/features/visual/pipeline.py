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

from src.core.contracts import DetectorPort
from src.core.models.detection import Detection
from src.core.models.subtitle_event import SubtitleEvent
from src.core.models.track import Track
from src.features.visual.processors.logo import LogoProcessor
from src.features.visual.processors.other_text import OtherTextProcessor
from src.features.visual.processors.watermark import WatermarkProcessor
from src.features.visual.subtitle.processor import SubtitleProcessor
from src.features.visual.tracker import ClassAwareTracker

logger = logging.getLogger(__name__)


@dataclass
class VisualRequest:
    enabled: bool = True
    video_path: str = ""
    subtitle_enabled: bool = True
    subtitle_mode: str = "translate_replace"  # translate_replace | blur | preserve
    subtitle_background: str = "blur"
    target_language: str = "vi"
    logo_enabled: bool = True
    logo_action: str = "blur"
    logo_image_path: Optional[str] = None
    watermark_enabled: bool = True
    watermark_action: str = "blur"
    other_text_enabled: bool = True
    other_text_action: str = "preserve"


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
    ):
        self.detector = detector
        self.tracker = tracker or ClassAwareTracker()
        self.subtitle_processor = subtitle_processor
        self.logo_processor = logo_processor
        self.watermark_processor = watermark_processor
        self.other_text_processor = other_text_processor

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

        # 3. Subtitle Processing
        t0 = time.monotonic()
        subtitle_events = []
        if request.subtitle_enabled and subtitle_tracks and self.subtitle_processor is not None:
            subtitle_events = self.subtitle_processor.process_tracks(
                tracks=subtitle_tracks,
                target_language=request.target_language,
                video_path=request.video_path,
            )
        timings["subtitle_processing"] = round(time.monotonic() - t0, 3)

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

        temp_no_audio = str(visual_dir / "rendered_no_audio.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(temp_no_audio, fourcc, fps, (width, height))

        frame_idx = 0
        events_by_frame: Dict[int, List[SubtitleEvent]] = {}
        for ev in subtitle_events:
            for b in ev.bbox_trajectory:
                pass
            # map events to frame indices
            start_f = int(round(ev.start_time * fps))
            end_f = int(round(ev.end_time * fps))
            for f_i in range(start_f, end_f + 1):
                events_by_frame.setdefault(f_i, []).append(ev)

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

            # B. Process Subtitle Processor rendering
            if request.subtitle_enabled and frame_idx in events_by_frame and self.subtitle_processor is not None:
                frame = self.subtitle_processor.render_frame_events(
                    frame, frame_idx, events_by_frame[frame_idx]
                )

            writer.write(frame)
            frame_idx += 1

        cap.release()
        writer.release()

        # 5. Audio Mux (Mux original audio if present)
        from src.infrastructure.video.ffmpeg import mux_audio_video
        out_rendered = mux_audio_video(
            video_no_audio_path=temp_no_audio,
            audio_source_path=request.video_path,
            output_path=output_video_path,
        )
        timings["rendering"] = round(time.monotonic() - t0, 3)

        timings["total"] = round(time.monotonic() - start_time, 3)

        return VisualResult(
            enabled=True,
            status="SUCCESS",
            rendered_video_path=out_rendered,
            detections_summary=class_counts,
            tracks_summary=track_summary,
            subtitle_events=subtitle_events,
            warnings=warnings,
            timings=timings,
        )
