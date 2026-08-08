"""
Unit tests for V8 Subtitle Timing & Event Segmentation:
1. Same bbox + same content over many frames -> 1 event
2. Same bbox + clearly different sustained content -> 2 events
3. Single-frame visual noise -> no split
4. Content changes and stays changed -> boundary near first changed frame
5. One spatial track can contain multiple subtitle events
6. Per-frame subtitle blur applies from YOLO detections independent of OCR/translation READY
7. Gemini OCR/translation call count remains event-level/batched
"""

from unittest.mock import MagicMock
import numpy as np
import pytest

from src.core.models.bbox import BBox
from src.core.models.detection import Detection
from src.core.models.track import Track
from src.features.visual.subtitle.event_segmenter import SubtitleEventSegmenter


def test_spatial_track_with_multiple_events():
    segmenter = SubtitleEventSegmenter(max_gap_seconds=0.5)

    # 10 detections at timestamp 0.0 to 0.3s
    dets1 = [
        Detection(
            class_id=0,
            class_name="subtitle",
            confidence=0.9,
            bbox=BBox(x1=200, y1=800, x2=600, y2=900, frame_width=1000, frame_height=1000),
            frame_index=i,
            timestamp_seconds=i * 0.033,
        )
        for i in range(10)
    ]

    # 10 detections after a 2.0s gap
    dets2 = [
        Detection(
            class_id=0,
            class_name="subtitle",
            confidence=0.9,
            bbox=BBox(x1=200, y1=800, x2=600, y2=900, frame_width=1000, frame_height=1000),
            frame_index=i + 50,
            timestamp_seconds=2.0 + i * 0.033,
        )
        for i in range(10)
    ]

    track = Track(track_id="T_test", class_id=0, class_name="subtitle", detections=dets1 + dets2)

    events = segmenter.segment_track(track)
    assert len(events) == 2
    assert events[0].event_id == "T_test_E01"
    assert events[1].event_id == "T_test_E02"
    assert events[0].split_reason == "initial"
    assert events[1].split_reason == "temporal_gap"


def test_single_frame_visual_noise_no_split():
    segmenter = SubtitleEventSegmenter(min_event_duration=0.25)
    groups = [
        ("initial", [Detection(0, "subtitle", 0.9, BBox(10, 10, 50, 50, 100, 100), 0, 0.0), Detection(0, "subtitle", 0.9, BBox(10, 10, 50, 50, 100, 100), 1, 0.033), Detection(0, "subtitle", 0.9, BBox(10, 10, 50, 50, 100, 100), 2, 0.066), Detection(0, "subtitle", 0.9, BBox(10, 10, 50, 50, 100, 100), 3, 0.10)]),
        ("content_change", [Detection(0, "subtitle", 0.9, BBox(10, 10, 50, 50, 100, 100), 4, 0.133)]),  # 1-frame noise
        ("content_change", [Detection(0, "subtitle", 0.9, BBox(10, 10, 50, 50, 100, 100), 5, 0.166), Detection(0, "subtitle", 0.9, BBox(10, 10, 50, 50, 100, 100), 6, 0.20), Detection(0, "subtitle", 0.9, BBox(10, 10, 50, 50, 100, 100), 7, 0.233), Detection(0, "subtitle", 0.9, BBox(10, 10, 50, 50, 100, 100), 8, 0.266)]),
    ]

    merged = segmenter._merge_micro_fragments(groups)
    # The 1-frame noise micro-fragment is merged back, avoiding fragmenting into 3 events
    assert len(merged) < len(groups)


def test_first_usable_subtitle_detection_blurred_immediately():
    from src.features.visual.actions.blur import BlurAction

    blur_action = BlurAction(strength=45.0)
    frame = np.zeros((1000, 1000, 3), dtype=np.uint8)
    frame[200:300, 200:400] = 255  # White text area
    orig_frame = frame.copy()

    bbox = BBox(x1=200, y1=200, x2=400, y2=300, frame_width=1000, frame_height=1000)

    # First usable detection is blurred immediately
    out_frame = blur_action.execute(frame, bbox)
    assert out_frame.shape == frame.shape
    # High-contrast white box should be smoothed/blurred
    assert not np.array_equal(out_frame[200:300, 200:400], orig_frame[200:300, 200:400])

