"""Smoke test using the real YOLO model.

This test is skipped automatically when the fine-tuned model file is missing, so
it does not force a download and does not require the model in unit-test runs.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from src.infrastructure.detection.config import load_detection_config
from src.infrastructure.detection.paths import resolve_yolo_weights

pytestmark = pytest.mark.skipif(
    os.getenv("MEDIA_FLOW_RUN_YOLO_INTEGRATION") != "1"
    or not (Path(__file__).resolve().parents[1] / "models" / "yolo" / "best.pt").is_file(),
    reason="real YOLO integration tests are opt-in and require models/yolo/best.pt",
)


def test_smoke_load_and_blank_frame():
    cfg = load_detection_config()
    cfg.device = "cpu"
    cfg.imgsz = 640
    cfg.temporal.enabled = False
    from src.infrastructure.detection import YoloBurnInDetector

    detector = YoloBurnInDetector(config=cfg)
    try:
        assert detector.num_classes() == 4
        names = detector.get_class_names()
        assert set(names.values()) == {
            "subtitle", "watermark", "logo", "other_burn_in_text",
        }
        blank = np.full((480, 640, 3), 128, dtype=np.uint8)
        result = detector.detect_frame(blank, frame_index=0, timestamp_seconds=0.0)
        # Blank frame yields no detections but must not crash.
        assert result.detections == []
        assert result.frame_width == 640
    finally:
        detector.release()


def test_smoke_resolution_via_paths():
    resolved = resolve_yolo_weights()
    assert resolved.is_file()
