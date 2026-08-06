"""Unit tests for post-processing and temporal stabilization."""

from __future__ import annotations

from src.infrastructure.detection.postprocess import (
    box_iou,
    clamp_box,
    filter_by_confidence,
    filter_invalid_boxes,
    group_close_boxes_by_class,
    postprocess_frame,
    remove_duplicates,
)
from src.infrastructure.detection.schemas import Detection
from src.infrastructure.detection.temporal import TemporalStabilizer


def _det(x1, y1, x2, y2, ci=0, cn="subtitle", conf=0.9):
    return Detection(
        class_id=ci,
        class_name=cn,
        confidence=conf,
        bbox_xyxy=[float(x1), float(y1), float(x2), float(y2)],
        bbox_normalized=[0, 0, 0, 0],
        frame_index=0,
        timestamp_seconds=0.0,
        frame_width=1920,
        frame_height=1080,
    )


def test_clamp_box_positive_area():
    assert clamp_box((-5, -5, 5, 5), 100, 100) == (0.0, 0.0, 5.0, 5.0)
    b = clamp_box((10, 10, 5, 5), 100, 100)
    assert b[2] > b[0]
    assert b[3] > b[1]


def test_box_iou():
    a = [0, 0, 10, 10]
    b = [5, 5, 15, 15]
    assert abs(box_iou(a, b) - (25.0 / 175.0)) < 1e-6
    assert box_iou(a, a) == 1.0


def test_filter_by_confidence_per_class():
    dets = [
        _det(0, 0, 10, 10, ci=0, cn="subtitle", conf=0.10),
        _det(0, 0, 10, 10, ci=1, cn="watermark", conf=0.5),
    ]
    conf = {"subtitle": 0.15, "watermark": 0.20, "logo": 0.2, "other_burn_in_text": 0.2}
    out = filter_by_confidence(dets, conf)
    assert len(out) == 1
    assert out[0].class_name == "watermark"


def test_filter_invalid_boxes_drops_small():
    dets = [_det(0, 0, 1, 1), _det(0, 0, 50, 50)]
    out = filter_invalid_boxes(dets, min_side=3.0)
    assert len(out) == 1


def test_remove_duplicates_same_class_keeps_highest():
    dets = [
        _det(0, 0, 100, 100, ci=0, cn="subtitle", conf=0.9),
        _det(2, 2, 98, 98, ci=0, cn="subtitle", conf=0.5),
    ]
    out = remove_duplicates(dets)
    assert len(out) == 1
    assert out[0].confidence == 0.9


def test_remove_duplicates_keeps_different_classes():
    dets = [
        _det(0, 0, 100, 100, ci=0, cn="subtitle", conf=0.9),
        _det(0, 0, 100, 100, ci=1, cn="watermark", conf=0.9),
    ]
    out = remove_duplicates(dets)
    assert len(out) == 2


def test_postprocess_frame_clamps_filters_and_dedupes():
    dets = [
        _det(-10, -10, 5000, 5000, ci=0, cn="subtitle", conf=0.01),  # below threshold
        _det(10, 10, 300, 300, ci=0, cn="subtitle", conf=0.9),
    ]
    conf = {"subtitle": 0.15, "watermark": 0.2, "logo": 0.2, "other_burn_in_text": 0.2}
    out = postprocess_frame(dets, conf, width=1920, height=1080, group_close=False)
    assert len(out) == 1
    assert out[0].bbox_xyxy[0] >= 0.0
    assert out[0].confidence == 0.9


def test_group_close_boxes_same_class():
    dets = [
        _det(100, 100, 200, 130, ci=0, cn="subtitle"),
        _det(210, 102, 310, 129, ci=0, cn="subtitle"),
    ]
    out = group_close_boxes_by_class(dets, gap_ratio=2.0)
    assert len(out) == 1
    assert out[0].bbox_xyxy[0] == 100.0
    assert out[0].bbox_xyxy[2] == 310.0


def test_group_close_boxes_keeps_different_classes():
    dets = [
        _det(100, 100, 200, 130, ci=0, cn="subtitle"),
        _det(210, 102, 310, 129, ci=1, cn="watermark"),
    ]
    out = group_close_boxes_by_class(dets, gap_ratio=2.0)
    assert len(out) == 2


def test_temporal_matches_and_smoothes():
    st = TemporalStabilizer(
        enabled=True, max_gap_frames=2, min_track_frames=2,
        iou_match_threshold=0.4, box_smoothing_alpha=0.65,
    )
    f0 = _det(100, 100, 200, 150)
    f1 = _det(105, 102, 205, 152)
    out0 = st.stabilize(0, [f0])
    out1 = st.stabilize(1, [f1])
    # With min_track_frames=2, frame 0 alone is dropped, frame 1 survives.
    assert out0 == []
    assert len(out1) == 1


def test_temporal_gap_carries_short_gap():
    st = TemporalStabilizer(
        enabled=True, max_gap_frames=2, min_track_frames=1,
        iou_match_threshold=0.4, box_smoothing_alpha=0.65,
    )
    st.stabilize(0, [_det(100, 100, 200, 150)])
    # frame 1 empty, frame 2 empty -> gap grows
    st.stabilize(1, [])
    out = st.stabilize(2, [_det(102, 101, 202, 151)])
    assert len(out) == 1


def test_temporal_no_mix_across_classes():
    st = TemporalStabilizer(
        enabled=True, max_gap_frames=2, min_track_frames=1,
        iou_match_threshold=0.4, box_smoothing_alpha=0.65,
    )
    st.stabilize(0, [_det(100, 100, 200, 150, ci=0, cn="subtitle")])
    out1 = st.stabilize(1, [_det(102, 101, 202, 151, ci=1, cn="watermark")])
    # different classes -> new track, not matched
    assert len(out1) == 1


def test_temporal_disabled_passes_through():
    st = TemporalStabilizer(enabled=False)
    dets = [_det(100, 100, 200, 150)]
    assert st.stabilize(0, dets) == dets


def test_adapter_to_proposal_bundle():
    from src.infrastructure.detection.adapter import results_to_proposal_bundle
    from src.infrastructure.detection.schemas import DetectionResult

    det = Detection(
        class_id=0,
        class_name="subtitle",
        confidence=0.9,
        bbox_xyxy=[10.0, 20.0, 300.0, 400.0],
        bbox_normalized=[10 / 1920, 20 / 1080, 300 / 1920, 400 / 1080],
        frame_index=1,
        timestamp_seconds=0.05,
        frame_width=1920,
        frame_height=1080,
    )
    res = DetectionResult(
        frame_index=1, timestamp_seconds=0.05,
        frame_width=1920, frame_height=1080, detections=[det],
    )
    bundle = results_to_proposal_bundle(
        "in.mp4", 30.0,
        video_width=1920, video_height=1080, total_frames=30,
        results=[res],
    )
    assert bundle.fps == 30.0
    assert len(bundle.frames) == 1
    legacy = bundle.frames[0].detections[0]
    # Legacy required keys preserved:
    assert legacy["frame_index"] == 1
    assert legacy["bbox_xyxy"] == [10.0, 20.0, 300.0, 400.0]
    assert legacy["timestamp_ms"] == 50.0
    assert legacy["candidate_id"] == "F00001_C00"
    assert legacy["confidence"] == 0.9
    # New per-class info carried through:
    assert legacy["class_id"] == 0
    assert legacy["class_name"] == "subtitle"
