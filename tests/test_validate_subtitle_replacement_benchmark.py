import json

import pytest

from scripts.validate_subtitle_replacement_benchmark import (
    classify,
    compute_metrics,
    iou,
    load_manifest,
)


def test_loads_benchmark_manifest(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps({
        "samples": [{"id": "center", "path": "center.mp4", "categories": ["center_subtitle"]}]
    }), encoding="utf-8")

    assert load_manifest(tmp_path)[0]["id"] == "center"


def test_computes_iou_and_detects_forbidden_region_overlap():
    left = {"left": 0, "top": 0, "right": 100, "bottom": 100}
    right = {"left": 50, "top": 50, "right": 150, "bottom": 150}
    report = {"captions": [{
        "caption_id": "one", "strategy": "local_band", "mask_bbox": left,
        "source_bbox": left, "mask_area_ratio": 0.1, "coordinate_mapping": "PASS",
        "target_text_clipped": False, "production_style_replacement": "PASS",
    }]}
    ground_truth = {"forbidden_regions": [{"bbox": right}]}

    metrics = compute_metrics(report, ground_truth)

    assert iou(left, right) == pytest.approx(1 / 7)
    assert metrics["false_positive_mask_count"] == 1
    assert classify(metrics) == "FAIL"


def test_computes_fallback_rate():
    report = {"captions": [
        {"caption_id": "one", "strategy": "full_width_fallback", "mask_area_ratio": 0.2, "coordinate_mapping": "PASS", "production_style_replacement": "CONDITIONAL"},
        {"caption_id": "two", "strategy": "local_band", "mask_area_ratio": 0.1, "coordinate_mapping": "PASS", "production_style_replacement": "PASS"},
    ]}

    metrics = compute_metrics(report, {})

    assert metrics["full_width_fallback_rate"] == 0.5
    assert classify(metrics) == "CONDITIONAL"


def test_clipping_prevents_production_style_pass():
    report = {"captions": [{
        "caption_id": "one", "strategy": "local_band", "mask_area_ratio": 0.1,
        "coordinate_mapping": "PASS", "target_text_clipped": True,
        "production_style_replacement": "PASS",
    }]}

    metrics = compute_metrics(report, {})

    assert metrics["target_text_clipping_count"] == 1
    assert classify(metrics) == "FAIL"
