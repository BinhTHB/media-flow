"""Supplementary unit tests for YOLO QA, benchmark, temporal, and downstream handling."""

from __future__ import annotations

import csv
import json
import tempfile
from pathlib import Path

import pytest

from scripts.run_detection_benchmark import CSV_COLUMNS, stats_to_row, VideoStats
from scripts.test_video_detection import has_audio_stream, mux_audio
from src.infrastructure.detection import (
    Detection,
    DetectionConfig,
    get_yolo_detector,
    YoloBurnInDetector,
)
from src.infrastructure.detection.adapter import detection_to_legacy_dict, results_to_proposal_bundle
from src.infrastructure.detection.schemas import DetectionResult
from src.infrastructure.detection.temporal import TemporalStabilizer


def _det(fi: int, x1: float, y1: float, x2: float, y2: float,
         ci: int = 0, cn: str = "subtitle", conf: float = 0.9) -> Detection:
    return Detection(
        class_id=ci,
        class_name=cn,
        confidence=conf,
        bbox_xyxy=[x1, y1, x2, y2],
        bbox_normalized=[x1 / 1000, y1 / 1000, x2 / 1000, y2 / 1000],
        frame_index=fi,
        timestamp_seconds=fi / 30.0,
        frame_width=1000,
        frame_height=1000,
    )


# 1. Temporal carry gắn track_id
def test_temporal_assigns_track_ids():
    st = TemporalStabilizer(enabled=True, max_gap_frames=2, min_track_frames=1)
    d0 = _det(0, 100, 100, 200, 150, ci=0, cn="subtitle")
    d1 = _det(1, 102, 101, 202, 151, ci=0, cn="subtitle")
    st.stabilize(0, [d0])
    st.stabilize(1, [d1])
    snapshot = st.to_dict()
    assert snapshot["active_tracks"] == 1
    assert snapshot["tracks"][0]["class_name"] == "subtitle"
    assert snapshot["tracks"][0]["n_frames"] == 2


# 2. Expiry sau max_gap_frames
def test_temporal_track_expires_after_max_gap():
    st = TemporalStabilizer(enabled=True, max_gap_frames=2, min_track_frames=1)
    st.stabilize(0, [_det(0, 100, 100, 200, 150)])
    st.stabilize(1, [])  # gap 1
    st.stabilize(2, [])  # gap 2
    st.stabilize(3, [])  # gap 3 (past max_gap)
    # Next detection at frame 4 gets a NEW track
    st.stabilize(4, [_det(4, 100, 100, 200, 150)])
    snapshot = st.to_dict()
    assert snapshot["active_tracks"] == 1
    assert snapshot["tracks"][0]["track_id"].startswith("T000002")


# 3. Không match khác class
def test_temporal_does_not_match_different_classes():
    st = TemporalStabilizer(enabled=True, max_gap_frames=2, min_track_frames=1)
    st.stabilize(0, [_det(0, 100, 100, 200, 150, ci=0, cn="subtitle")])
    st.stabilize(1, [_det(1, 100, 100, 200, 150, ci=1, cn="watermark")])
    snapshot = st.to_dict()
    assert snapshot["active_tracks"] == 2


# 4. Hai track cùng class độc lập
def test_temporal_two_separate_tracks_same_class():
    st = TemporalStabilizer(enabled=True, max_gap_frames=2, min_track_frames=1)
    # Top and bottom subtitles on the same frame
    st.stabilize(0, [
        _det(0, 100, 50, 400, 90, ci=0, cn="subtitle"),
        _det(0, 100, 850, 400, 890, ci=0, cn="subtitle"),
    ])
    snapshot = st.to_dict()
    assert snapshot["active_tracks"] == 2


# 5. Per-class downstream routing
def test_downstream_adapter_preserves_class_metadata():
    det_sub = _det(0, 10, 20, 100, 50, ci=0, cn="subtitle")
    det_wm = _det(0, 800, 20, 950, 50, ci=1, cn="watermark")
    res = DetectionResult(frame_index=0, timestamp_seconds=0.0, frame_width=1000, frame_height=1000,
                          detections=[det_sub, det_wm])
    bundle = results_to_proposal_bundle("v.mp4", 30.0, video_width=1000, video_height=1000, total_frames=1, results=[res])
    dets = bundle.frames[0].detections
    assert len(dets) == 2
    assert dets[0]["class_name"] == "subtitle"
    assert dets[0]["class_id"] == 0
    assert dets[1]["class_name"] == "watermark"
    assert dets[1]["class_id"] == 1


# 6. Audio preservation helper on synthetic non-file path
def test_has_audio_stream_nonexistent_returns_false():
    p = Path("nonexistent_file_xyz.mp4")
    assert has_audio_stream(p) is False


# 7. Benchmark CSV schema validation
def test_benchmark_csv_schema_completeness():
    st = VideoStats(video="test.mp4", config_name="A", device="cpu", width=1920, height=1080, input_fps=30.0, frames=100)
    row = stats_to_row(st)
    for col in CSV_COLUMNS:
        assert col in row, f"Missing CSV column {col}"


# 8. Shared detector caching (model reuse)
def test_detector_cache_reuse():
    # Load same weights twice via get_yolo_detector (using dummy/default weights resolution)
    cfg1 = DetectionConfig(device="cpu")
    cfg2 = DetectionConfig(device="cpu")
    d1 = get_yolo_detector(cfg1)
    d2 = get_yolo_detector(cfg2)
    assert d1 is d2


# 9. Threshold CLI override simulation
def test_config_per_class_confidence_overrides():
    cfg = DetectionConfig()
    cfg.confidence["subtitle"] = 0.05
    cfg.confidence["watermark"] = 0.35
    assert cfg.confidence["subtitle"] == 0.05
    assert cfg.confidence["watermark"] == 0.35
    assert cfg.confidence["logo"] == 0.20


# 10. JSONL metadata output format
def test_jsonl_metadata_format():
    det = _det(12, 100.5, 200.25, 300.75, 400.0, ci=3, cn="other_burn_in_text", conf=0.876543)
    data = det.to_json_compatible()
    assert data["class_id"] == 3
    assert data["class_name"] == "other_burn_in_text"
    assert data["confidence"] == 0.876543
    assert data["bbox_xyxy"] == [100.5, 200.25, 300.75, 400.0]
    assert data["frame_index"] == 12
    assert "timestamp_seconds" in data
    assert "frame_width" in data
    assert "frame_height" in data
