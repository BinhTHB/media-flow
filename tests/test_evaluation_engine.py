"""Regression tests for the evaluation engine, metrics, matching, and policy."""

from __future__ import annotations

import pytest

from src.infrastructure.detection.evaluation import (
    Box,
    ClassMetrics,
    EvaluationReport,
    FrameTruth,
    compute_class_metrics,
    evaluate_frames,
    greedy_class_agnostic_match,
    greedy_class_aware_match,
    micro_macro,
)
from src.infrastructure.detection.evaluation.temporal_eval import compute_frame_coverage
from src.infrastructure.detection.postprocess import (
    class_safe_cover,
    filter_by_detect_policy,
)
from src.infrastructure.detection.schemas import Detection


def _det(class_name: str = "subtitle", ci: int = 0) -> Detection:
    return Detection(
        class_id=ci, class_name=class_name, confidence=0.9,
        bbox_xyxy=[10, 10, 50, 50], bbox_normalized=[0.01, 0.01, 0.05, 0.05],
        frame_index=0, timestamp_seconds=0.0, frame_width=1000, frame_height=1000,
    )


# 1. Matching: same-class only for class-aware
def test_greedy_class_aware_rejects_class_mismatch():
    gt = [Box(class_id=0, bbox=[10, 10, 50, 50])]
    pred = [Box(class_id=1, bbox=[10, 10, 50, 50], confidence=0.9)]
    res = greedy_class_aware_match(gt, pred, iou_threshold=0.5)
    assert len(res.matches) == 0
    assert len(res.unmatched_gt) == 1
    assert len(res.unmatched_pred) == 1


# 2. Matching: class-agnostic pairs different classes
def test_greedy_class_agnostic_matches_across_classes():
    gt = [Box(class_id=0, bbox=[10, 10, 50, 50])]
    pred = [Box(class_id=1, bbox=[10, 10, 50, 50], confidence=0.9)]
    res = greedy_class_agnostic_match(gt, pred, iou_threshold=0.5)
    assert len(res.matches) == 1
    assert res.matches[0].gt_class_id == 0
    assert res.matches[0].pred_class_id == 1


# 3. Metrics calculation correctness
def test_compute_class_metrics():
    m = compute_class_metrics(tp=8, fp=2, fn=2)
    assert m.precision == pytest.approx(0.8)
    assert m.recall == pytest.approx(0.8)
    assert m.f1 == pytest.approx(0.8)


# 4. Zero division handling
def test_compute_class_metrics_zero_division():
    m = compute_class_metrics(tp=0, fp=0, fn=0)
    assert m.precision == 0.0
    assert m.recall == 0.0
    assert m.f1 == 0.0


# 5. Micro/Macro aggregation
def test_micro_macro_aggregation():
    by_class = {
        "subtitle": ClassMetrics(tp=10, fp=0, fn=0, precision=1.0, recall=1.0, f1=1.0),
        "watermark": ClassMetrics(tp=0, fp=10, fn=10, precision=0.0, recall=0.0, f1=0.0),
    }
    agg = micro_macro(by_class)
    assert agg["micro_precision"] == 0.5
    assert agg["micro_recall"] == 0.5
    assert agg["micro_f1"] == 0.5
    assert agg["macro_precision"] == 0.5
    assert agg["macro_recall"] == 0.5
    assert agg["macro_f1"] == 0.5


# 6. Full report on multi-frame scenario
def test_evaluate_frames_full():
    ft1 = FrameTruth(
        frame_id="f1", video_id="v1", frame_index=0, width=100, height=100,
        gt=[Box(0, [10, 10, 50, 30])],
        pred=[Box(0, [10, 10, 50, 30], 0.9)],
    )
    ft2 = FrameTruth(
        frame_id="f2", video_id="v1", frame_index=1, width=100, height=100,
        gt=[Box(1, [60, 60, 90, 90])],
        pred=[Box(2, [60, 60, 90, 90], 0.8)],  # wrong class: logo instead of watermark
    )
    rep = evaluate_frames([ft1, ft2], iou_threshold=0.5)
    assert rep.total_gt == 2
    assert rep.total_pred == 2
    assert rep.by_class["subtitle"].tp == 1
    assert rep.by_class["watermark"].fn == 1
    assert rep.by_class["logo"].fp == 1
    assert len(rep.errors) >= 1
    assert any(e.kind == "wrong_class" for e in rep.errors)


# 7. Class action policy filter
def test_filter_by_detect_policy():
    dets = [_det("subtitle", 0), _det("watermark", 1)]
    policy = {
        "subtitle": {"detect": True, "safe_cover": True},
        "watermark": {"detect": False, "safe_cover": False},
    }
    filtered = filter_by_detect_policy(dets, policy)
    assert len(filtered) == 1
    assert filtered[0].class_name == "subtitle"


# 8. Safe cover query
def test_class_safe_cover():
    policy = {"watermark": {"detect": True, "safe_cover": False}}
    assert class_safe_cover("subtitle", policy) is True  # default True
    assert class_safe_cover("watermark", policy) is False


# 9. Temporal coverage metrics
def test_compute_frame_coverage():
    gt = [True, True, True, True, True]
    pred = [True, True, False, True, True]
    cov = compute_frame_coverage(gt, pred)
    assert cov["coverage_ratio"] == 0.8
    assert cov["max_consecutive_missed"] == 1.0
