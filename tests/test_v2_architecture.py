"""
Unit and Integration Tests for V2 Feature-First Modular Clean Architecture
"""

import pytest

from src.core.models.bbox import BBox
from src.core.models.detection import Detection
from src.core.models.track import Track
from src.core.models.subtitle_event import SubtitleEvent
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


class _RecordingBlur:
    def __init__(self):
        self.calls = []

    def execute(self, frame, bbox, padding_px=0):
        self.calls.append((bbox, padding_px))
        return frame


class _RecordingSubtitleProcessor:
    def __init__(self):
        self.blur_action = _RecordingBlur()
        self.render_calls = []
        self.suppression_calls = []

    def suppress_source_subtitles(self, frame, regions):
        self.suppression_calls.append(list(regions))
        return frame

    select_render_events = staticmethod(SubtitleProcessor.select_render_events)

    @staticmethod
    def get_source_suppression_bbox(bbox, pad_x=18, pad_y=14):
        return bbox

    def render_frame_events(self, frame, frame_index, events):
        self.render_calls.append((frame_index, list(events)))
        return frame


def _subtitle_detection():
    bbox = BBox(
        x1=10.0,
        y1=20.0,
        x2=100.0,
        y2=60.0,
        frame_width=320,
        frame_height=240,
    )
    return Detection(
        class_id=0,
        class_name="subtitle",
        confidence=0.9,
        bbox=bbox,
        frame_index=0,
        timestamp_seconds=0.0,
    )


def _subtitle_event(status="READY"):
    det = _subtitle_detection()
    return SubtitleEvent(
        event_id="event-1",
        track_id="track-1",
        start_time=0.0,
        end_time=1.0,
        anchor_bbox=det.bbox,
        replacement_status=status,
        blur_applied=status == "READY",
        overlay_applied=status == "READY",
    )


def test_visual_request_defaults_other_text_to_documented_blur_policy():
    assert VisualRequest().other_text_action == "blur"


def test_subtitle_preserve_mode_never_changes_pixels():
    processor = _RecordingSubtitleProcessor()
    pipeline = VisualPipeline(subtitle_processor=processor)
    frame = object()

    result = pipeline._apply_subtitle_actions(
        frame,
        0,
        VisualRequest(subtitle_mode="preserve"),
        {0: [_subtitle_detection()]},
        {0: [_subtitle_event()]},
    )

    assert result is frame
    assert processor.blur_action.calls == []
    assert processor.render_calls == []


def test_subtitle_blur_mode_blurs_detections_without_overlay():
    processor = _RecordingSubtitleProcessor()
    pipeline = VisualPipeline(subtitle_processor=processor)

    pipeline._apply_subtitle_actions(
        object(),
        0,
        VisualRequest(subtitle_mode="blur"),
        {0: [_subtitle_detection()]},
        {},
    )

    assert len(processor.blur_action.calls) == 1
    assert processor.render_calls == []


def test_blur_mode_merges_same_frame_detections_into_union_box():
    # Real regression: the generic text model emits word-level boxes; blurring
    # each box separately leaked glyph edges and inter-word gaps of the
    # original caption. Same-frame subtitle detections must collapse into one
    # union box before suppression/blur.
    from src.features.visual.pipeline import merge_same_frame_detections

    left = Detection(
        class_id=0, class_name="subtitle", confidence=0.9,
        bbox=BBox(x1=10.0, y1=20.0, x2=100.0, y2=60.0, frame_width=320, frame_height=240),
        frame_index=0, timestamp_seconds=0.0,
    )
    right = Detection(
        class_id=0, class_name="subtitle", confidence=0.8,
        bbox=BBox(x1=120.0, y1=25.0, x2=200.0, y2=55.0, frame_width=320, frame_height=240),
        frame_index=0, timestamp_seconds=0.0,
    )
    single = Detection(
        class_id=0, class_name="subtitle", confidence=0.7,
        bbox=BBox(x1=10.0, y1=20.0, x2=100.0, y2=60.0, frame_width=320, frame_height=240),
        frame_index=1, timestamp_seconds=1 / 30.0,
    )

    merged = merge_same_frame_detections({0: [left, right], 1: [single]})

    assert len(merged[0]) == 1
    union_bbox = merged[0][0].bbox
    assert (union_bbox.x1, union_bbox.y1, union_bbox.x2, union_bbox.y2) == (10.0, 20.0, 200.0, 60.0)
    assert merged[0][0].confidence == 0.9
    # Single-detection frames pass through untouched.
    assert merged[1] == [single]


def test_translate_replace_preserves_source_when_event_is_not_ready():
    processor = _RecordingSubtitleProcessor()
    pipeline = VisualPipeline(subtitle_processor=processor)

    pipeline._apply_subtitle_actions(
        object(),
        0,
        VisualRequest(subtitle_mode="translate_replace"),
        {0: [_subtitle_detection()]},
        {0: [_subtitle_event(status="OCR_FAILED")]},
    )

    assert processor.blur_action.calls == []
    assert processor.render_calls == []


def test_translate_replace_suppresses_and_overlays_only_ready_events():
    processor = _RecordingSubtitleProcessor()
    pipeline = VisualPipeline(subtitle_processor=processor)
    ready = _subtitle_event()

    pipeline._apply_subtitle_actions(
        object(),
        0,
        VisualRequest(subtitle_mode="translate_replace"),
        {0: [_subtitle_detection()]},
        {0: [ready]},
    )

    # Source glyphs are covered by VisualPipeline's local blur, not the
    # processor's Telea inpaint path. Overlay still goes through the processor.
    assert processor.render_calls == [(0, [ready])]


def test_invalid_subtitle_mode_fails_explicitly():
    with pytest.raises(ValueError, match="Unsupported subtitle_mode"):
        VisualPipeline._validate_request(VisualRequest(subtitle_mode="unknown"))
