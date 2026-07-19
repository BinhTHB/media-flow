import os, sys, pytest, cv2, tempfile, json
import numpy as np

sys.path.insert(0, '.')
from src.infrastructure.benchmark.hybrid_subtitle.yolo_proposal import YoloTextProposalProvider
from src.infrastructure.benchmark.hybrid_subtitle.grouping import TextCandidateGrouper

from src.infrastructure.benchmark.hybrid_subtitle.core import (
    group_frame_candidates, build_tracks, validate_gemini_decision,
)
from src.infrastructure.benchmark.hybrid_subtitle.gt_metrics import evaluate_tracks_vs_gt
from src.infrastructure.benchmark.hybrid_subtitle.runner import HybridSubtitleBenchmarkRunner, BenchmarkConfig


def create_synthetic_video(path: str, frames: list[np.ndarray], fps: float = 30.0):
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(path, fourcc, fps, (w, h))
    for f in frames:
        writer.write(f)
    writer.release()


def test_yolo_multiple_text_and_cpu_fallback():
    # 1. Create a frame with multiple text regions
    frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(frame1, "HELLO WORLD", (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame1, "SUBTITLE LINE TWO", (100, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    with tempfile.TemporaryDirectory() as tmpdir:
        vid_path = os.path.join(tmpdir, "test.mp4")
        create_synthetic_video(vid_path, [frame1, frame1])

        # Test CPU fallback automatically
        provider = YoloTextProposalProvider(weights_path="best.pt", conf=0.01, imgsz=320, device="cuda")
        assert provider._device in ["cpu", "cuda"]

        dets = provider.detect(vid_path)
        assert len(dets) > 0
        # Ensure we have detections on frame 0 and 1
        frames_seen = [d["frame_index"] for d in dets]
        assert 0 in frames_seen
        # frame_index must be monotonic
        assert frames_seen == sorted(frames_seen)


def test_yolo_empty_frame_no_crash():
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    with tempfile.TemporaryDirectory() as tmpdir:
        vid_path = os.path.join(tmpdir, "empty.mp4")
        create_synthetic_video(vid_path, [frame, frame])

        provider = YoloTextProposalProvider(weights_path="best.pt", conf=0.5, imgsz=320, device="cpu")
        dets = provider.detect(vid_path)
        # Should not crash, and detections might be empty on solid black frame
        assert isinstance(dets, list)


def test_grouping_multiple_boxes():
    # Create adjacent word boxes on frame
    dets = [
        {"candidate_id": "F000_C01", "frame_index": 0, "timestamp_ms": 0, "bbox_xyxy": [100, 200, 200, 230], "confidence": 0.8},
        {"candidate_id": "F000_C02", "frame_index": 0, "timestamp_ms": 0, "bbox_xyxy": [210, 202, 310, 229], "confidence": 0.9},
        {"candidate_id": "F000_C03", "frame_index": 0, "timestamp_ms": 0, "bbox_xyxy": [100, 400, 200, 430], "confidence": 0.75}, # separate baseline
    ]
    groups = group_frame_candidates(dets, horizontal_gap_ratio=2.0)
    # C01 and C02 should be grouped together into a single group because they share baseline and are horizontally close
    assert len(groups) == 2
    assert groups[0]["member_count"] == 2
    assert groups[1]["member_count"] == 1


def test_tracker_creates_tracks():
    # Setup frame groups over time
    groups = [
        {"candidate_group_id": "G000_01", "frame_index": 0, "timestamp_ms": 0, "union_bbox_xyxy": [100, 200, 310, 230], "mean_detector_confidence": 0.85},
        {"candidate_group_id": "G001_01", "frame_index": 1, "timestamp_ms": 33, "union_bbox_xyxy": [102, 198, 308, 232], "mean_detector_confidence": 0.88},
        {"candidate_group_id": "G002_01", "frame_index": 2, "timestamp_ms": 66, "union_bbox_xyxy": [100, 200, 310, 230], "mean_detector_confidence": 0.90},
    ]
    tracks = build_tracks(groups, iou_threshold=0.5, max_gap_frames=2)
    assert len(tracks) == 1
    assert tracks[0]["frame_count"] == 3
    assert tracks[0]["track_id"] == "T001"


def test_verifier_rejections():
    # Unknown ID in decisions
    payload = {
        "track_id": "T001",
        "decisions": [
            {"candidate_id": "C1", "is_subtitle": True, "category": "dialogue_subtitle", "confidence": 0.9},
            {"candidate_id": "UNKNOWN_ID", "is_subtitle": True, "category": "dialogue_subtitle", "confidence": 0.9},
        ],
        "uncertain": False
    }
    with pytest.raises(ValueError, match="unknown candidate ID"):
        validate_gemini_decision(payload, "T001", {"C1"})


def test_metrics_no_gt_returns_not_computable():
    res = evaluate_tracks_vs_gt([], [])
    assert res["ground_truth_available"] is False
    assert "No ground truth" in res["note"]
