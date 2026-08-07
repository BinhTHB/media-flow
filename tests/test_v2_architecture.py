"""
Unit and Integration Tests for V2 Feature-First Modular Clean Architecture
"""

import os
import pytest
from pathlib import Path

from src.core.models.bbox import BBox
from src.core.models.detection import Detection
from src.core.models.track import Track
from src.core.models.subtitle_event import SubtitleEvent
from src.core.models.speech_segment import SpeechSegment
from src.features.audio.pipeline import AudioPipeline, AudioRequest
from src.features.visual.pipeline import VisualPipeline, VisualRequest
from src.features.visual.subtitle.event_segmenter import SubtitleEventSegmenter
from src.features.visual.subtitle.processor import SubtitleProcessor
from src.features.visual.processors.logo import LogoProcessor
from src.features.visual.processors.watermark import WatermarkProcessor
from src.features.visual.processors.other_text import OtherTextProcessor
from src.features.visual.actions.blur import BlurAction


def test_subtitle_event_segmenter_splits_consecutive_same_position_captions():
    """Two consecutive captions appearing at the same position must produce two distinct SubtitleEvents."""
    bbox = BBox(x1=100.0, y1=800.0, x2=500.0, y2=850.0, frame_width=720, frame_height=1280)
    
    # Caption 1: 0.0s to 1.0s (30 fps)
    dets1 = [
        Detection(class_id=0, class_name="subtitle", confidence=0.9, bbox=bbox, frame_index=i, timestamp_seconds=i / 30.0)
        for i in range(0, 30)
    ]
    # Gap of 0.8s between captions (1.0s to 1.8s)
    # Caption 2: 1.8s to 3.0s (30 fps)
    dets2 = [
        Detection(class_id=0, class_name="subtitle", confidence=0.9, bbox=bbox, frame_index=i, timestamp_seconds=i / 30.0)
        for i in range(54, 90)
    ]

    track = Track(track_id="T001_subtitle", class_id=0, class_name="subtitle", detections=dets1 + dets2)
    segmenter = SubtitleEventSegmenter(max_gap_seconds=0.5)
    events = segmenter.segment_track(track)

    assert len(events) == 2
    assert events[0].event_id == "T001_subtitle_E01"
    assert events[1].event_id == "T001_subtitle_E02"
    assert events[0].end_time < events[1].start_time


def test_audio_pipeline_disabled_skips_execution():
    req = AudioRequest(enabled=False)
    pipeline = AudioPipeline()
    res = pipeline.process(req, job_workspace_dir="outputs/jobs/test_job")

    assert res.enabled is False
    assert res.status == "SKIPPED"


def test_visual_pipeline_disabled_skips_execution():
    req = VisualRequest(enabled=False)
    pipeline = VisualPipeline()
    res = pipeline.process(req, job_workspace_dir="outputs/jobs/test_job", output_video_path="out.mp4")

    assert res.enabled is False
    assert res.status == "SKIPPED"


def test_shared_blur_action_reused_by_processors():
    blur_action = BlurAction(strength=20.0, padding_px=2)
    logo_proc = LogoProcessor(blur_action=blur_action)
    watermark_proc = WatermarkProcessor(blur_action=blur_action)
    other_proc = OtherTextProcessor(blur_action=blur_action)

    assert logo_proc.blur_action == blur_action
    assert watermark_proc.blur_action == blur_action
    assert other_proc.blur_action == blur_action
