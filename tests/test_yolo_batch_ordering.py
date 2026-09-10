"""Tests for batch frame ordering and timestamp continuity under batching."""

from __future__ import annotations

import os

import numpy as np
import pytest

from src.infrastructure.detection import DetectionConfig, YoloBurnInDetector

pytestmark = pytest.mark.skipif(
    os.getenv("MEDIA_FLOW_RUN_YOLO_INTEGRATION") != "1",
    reason="real YOLO integration tests are opt-in",
)


def test_batch_ordering_and_timestamps():
    cfg = DetectionConfig(device="cpu")
    detector = YoloBurnInDetector(config=cfg)
    try:
        # Create 4 synthetic frames
        frames = [np.full((240, 320, 3), 128 + i * 10, dtype=np.uint8) for i in range(4)]
        timestamps = [0.0, 0.033, 0.066, 0.100]

        results = detector.detect_batch(
            frames,
            frame_index_offset=10,
            timestamps_seconds=timestamps,
        )

        assert len(results) == 4
        for i, res in enumerate(results):
            assert res.frame_index == 10 + i
            assert abs(res.timestamp_seconds - timestamps[i]) < 1e-5
            assert res.frame_width == 320
            assert res.frame_height == 240
    finally:
        detector.release()


def test_batch_matches_single_decode_indices():
    cfg = DetectionConfig(device="cpu")
    detector = YoloBurnInDetector(config=cfg)
    try:
        frame = np.full((240, 320, 3), 128, dtype=np.uint8)
        single = detector.detect_frame(frame, frame_index=5, timestamp_seconds=0.166)
        batch = detector.detect_batch([frame], frame_index_offset=5, timestamps_seconds=[0.166])[0]

        assert single.frame_index == batch.frame_index == 5
        assert abs(single.timestamp_seconds - batch.timestamp_seconds) < 1e-5
    finally:
        detector.release()
