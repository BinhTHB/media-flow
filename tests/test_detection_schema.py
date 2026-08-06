"""Unit tests for the detection schema."""

from __future__ import annotations

import json

import pytest

from src.infrastructure.detection.schemas import (
    EXPECTED_CLASS_NAMES,
    ClassMappingError,
    Detection,
    DetectionResult,
    verify_class_mapping,
)


def _det(x1=10, y1=20, x2=100, y2=200, ci=0, cn="subtitle", conf=0.9):
    return Detection(
        class_id=ci,
        class_name=cn,
        confidence=conf,
        bbox_xyxy=[x1, y1, x2, y2],
        bbox_normalized=[x1 / 1920, y1 / 1080, x2 / 1920, y2 / 1080],
        frame_index=5,
        timestamp_seconds=0.2,
        frame_width=1920,
        frame_height=1080,
    )


def test_expected_class_mapping():
    assert EXPECTED_CLASS_NAMES == {
        0: "subtitle",
        1: "watermark",
        2: "logo",
        3: "other_burn_in_text",
    }


def test_verify_class_mapping_ok():
    m = verify_class_mapping(list(EXPECTED_CLASS_NAMES.values()))
    assert m[0] == "subtitle"
    assert m[3] == "other_burn_in_text"


def test_verify_class_mapping_wrong_count():
    with pytest.raises(ClassMappingError):
        verify_class_mapping(["subtitle", "watermark"])


def test_verify_class_mapping_wrong_order():
    with pytest.raises(ClassMappingError):
        verify_class_mapping(
            ["watermark", "subtitle", "logo", "other_burn_in_text"]
        )


def test_detection_serializes_to_json():
    d = _det()
    payload = d.to_json_compatible()
    text = json.dumps(payload)
    assert isinstance(text, str)
    assert payload["class_name"] == "subtitle"
    assert payload["frame_index"] == 5


def test_detection_clamp_inside_bounds():
    d = _det(x1=-10, y1=-5, x2=9999, y2=9999)
    c = d.clamp()
    assert c.bbox_xyxy[0] == 0.0
    assert c.bbox_xyxy[1] == 0.0
    assert c.bbox_xyxy[2] <= 1920.0
    assert c.bbox_xyxy[3] <= 1080.0


def test_detection_result_to_dict():
    res = DetectionResult(
        frame_index=0,
        timestamp_seconds=0.0,
        frame_width=100,
        frame_height=60,
        detections=[_det()],
    )
    d = res.to_dict()
    assert d["frame_index"] == 0
    assert len(d["detections"]) == 1
