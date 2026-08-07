"""Detection quality metrics: Precision / Recall / F1 and localization stats."""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import median
from typing import Dict, List, Optional, Sequence


def compute_class_metrics(tp: int, fp: int, fn: int) -> "ClassMetrics":
    """Precision / Recall / F1 from raw counts (guards against /0)."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return ClassMetrics(tp=tp, fp=fp, fn=fn, precision=precision, recall=recall, f1=f1)


@dataclass
class ClassMetrics:
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float


def micro_macro(by_class: Dict[str, ClassMetrics]) -> Dict[str, float]:
    """Aggregate per-class metrics into micro and macro averages.

    * micro average recomputes P/R/F1 from summed TP/FP/FN (class-imbalance
      aware).
    * macro average is the unweighted mean of each per-class P/R/F1.
    """
    tps = sum(m.tp for m in by_class.values())
    fps = sum(m.fp for m in by_class.values())
    fns = sum(m.fn for m in by_class.values())
    micro = compute_class_metrics(tps, fps, fns)
    if not by_class:
        return {"micro_precision": 0.0, "micro_recall": 0.0, "micro_f1": 0.0,
                "macro_precision": 0.0, "macro_recall": 0.0, "macro_f1": 0.0}
    macro_p = sum(m.precision for m in by_class.values()) / len(by_class)
    macro_r = sum(m.recall for m in by_class.values()) / len(by_class)
    macro_f1 = (
        2 * macro_p * macro_r / (macro_p + macro_r)
        if (macro_p + macro_r) > 0
        else 0.0
    )
    return {
        "micro_precision": micro.precision,
        "micro_recall": micro.recall,
        "micro_f1": micro.f1,
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "macro_f1": macro_f1,
    }


def iou_stats(ious: Sequence[float]) -> Dict[str, float]:
    """Mean / median / P10 / P90 of matched IoUs."""
    vals = sorted(float(v) for v in ious)
    if not vals:
        return {"mean_iou": 0.0, "median_iou": 0.0, "iou_p10": 0.0, "iou_p90": 0.0}
    n = len(vals)
    p10 = vals[max(0, min(n - 1, int(round(0.10 * (n - 1)))))]
    p90 = vals[max(0, min(n - 1, int(round(0.90 * (n - 1)))))]
    return {
        "mean_iou": round(sum(vals) / n, 4),
        "median_iou": round(median(vals), 4),
        "iou_p10": round(p10, 4),
        "iou_p90": round(p90, 4),
    }


def aggregate_iou_stats(grouped: Dict[str, Sequence[float]]) -> Dict[str, Dict[str, float]]:
    """IoU stats grouped per class name."""
    return {name: iou_stats(ious) for name, ious in grouped.items()}


def average_metrics(values: Sequence[ClassMetrics]) -> float:
    """Unweighted mean F1 (used for the macro F1 shortcut)."""
    if not values:
        return 0.0
    return sum(v.f1 for v in values) / len(values)
