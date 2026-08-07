"""Frame-level evaluation engine.

Consumes per-frame ground truth and predictions and produces the full metric
bundle: per-class Precision / Recall / F1, micro/macro averages, localization
IoU statistics, the 4-class confusion matrix, and structured error records for
the gallery.

This module is deliberately independent from the detector: both ground truth
and predictions are plain :class:`Box` lists, so it runs offline and is fully
unit-testable.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from src.infrastructure.detection.evaluation.confusion import (
    CONFUSION_BACKGROUND,
    ConfusionMatrix,
    accumulate_confusion,
    build_confusion_matrix,
)
from src.infrastructure.detection.evaluation.matching import (
    Box,
    MatchResult,
    greedy_class_agnostic_match,
    per_class_counts,
)
from src.infrastructure.detection.evaluation.metrics import (
    ClassMetrics,
    compute_class_metrics,
    iou_stats,
    micro_macro,
)
from src.infrastructure.detection.schemas import EXPECTED_CLASS_NAMES


@dataclass
class FrameTruth:
    """Ground truth and predictions for a single frame."""

    frame_id: str
    video_id: str
    frame_index: int
    width: int
    height: int
    gt: List[Box]
    pred: List[Box]
    timestamp_seconds: float = 0.0


@dataclass
class ErrorRecord:
    kind: str  # false_negative | false_positive | wrong_class | low_iou
    frame_id: str
    video_id: str
    frame_index: int
    width: int
    height: int
    timestamp_seconds: float
    gt_class: Optional[str] = None
    pred_class: Optional[str] = None
    confidence: Optional[float] = None
    iou: Optional[float] = None
    gt_box: Optional[List[float]] = None
    pred_box: Optional[List[float]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "frame_id": self.frame_id,
            "video_id": self.video_id,
            "frame_index": self.frame_index,
            "timestamp_seconds": self.timestamp_seconds,
            "gt_class": self.gt_class,
            "pred_class": self.pred_class,
            "confidence": self.confidence,
            "iou": self.iou,
            "gt_box": self.gt_box,
            "pred_box": self.pred_box,
            "width": self.width,
            "height": self.height,
        }


@dataclass
class EvaluationReport:
    iou_threshold: float
    by_class: Dict[str, ClassMetrics]
    aggregates: Dict[str, float]
    iou_by_class: Dict[str, Dict[str, float]]
    confusion: ConfusionMatrix
    errors: List[ErrorRecord]
    total_gt: int = 0
    total_pred: int = 0
    total_matched: int = 0


def evaluate_frames(
    frames: Sequence[FrameTruth],
    *,
    iou_threshold: float = 0.5,
    class_names: Optional[Dict[int, str]] = None,
) -> EvaluationReport:
    """Evaluate all frames and produce the metric bundle.

    Matching is one-to-one and greedy descending-IoU. Per-class metrics use
    class-aware matching; the confusion matrix uses class-agnostic matching.
    """
    names = dict(class_names or EXPECTED_CLASS_NAMES)
    classes = [names[i] for i in sorted(names)]

    per_class_tp: Dict[int, int] = {}
    per_class_fp: Dict[int, int] = {}
    per_class_fn: Dict[int, int] = {}
    ious_by_class: Dict[int, List[float]] = {}
    confusion_counts: Dict[tuple, int] = {}
    errors: List[ErrorRecord] = []
    total_gt = 0
    total_pred = 0
    total_matched = 0

    for ft in frames:
        gt_by_index: Dict[int, Box] = {i: b for i, b in enumerate(ft.gt)}
        pred_by_index: Dict[int, Box] = {i: b for i, b in enumerate(ft.pred)}
        total_gt += len(ft.gt)
        total_pred += len(ft.pred)

        # Class-aware counts for P/R/F1.
        counts = per_class_counts(ft.gt, ft.pred, iou_threshold=iou_threshold)
        for cid, cc in counts.items():
            per_class_tp[cid] = per_class_tp.get(cid, 0) + cc.tp
            per_class_fp[cid] = per_class_fp.get(cid, 0) + cc.fp
            per_class_fn[cid] = per_class_fn.get(cid, 0) + cc.fn

        # Class-agnostic match for confusion + error analysis.
        agn = greedy_class_agnostic_match(ft.gt, ft.pred, iou_threshold=iou_threshold)
        total_matched += len(agn.matches)
        gt_by_class: Dict[int, List[tuple]] = {}
        for gi, b in enumerate(ft.gt):
            gt_by_class.setdefault(b.class_id, []).append((gi, b))
        confusion_counts = _merge_counts(
            confusion_counts,
            accumulate_confusion(gt_by_class, agn, names),
        )

        for m in agn.matches:
            g = gt_by_index[m.gt_index]
            p = pred_by_index[m.pred_index]
            gt_name = names.get(g.class_id, "unknown")
            pred_name = names.get(p.class_id, "unknown")
            if gt_name != pred_name:
                errors.append(_err("wrong_class", ft, gt_name, pred_name, p.confidence, m.iou, g.bbox, p.bbox))
            ious_by_class.setdefault(g.class_id, []).append(m.iou)
            if m.iou < 0.75:
                errors.append(_err("low_iou", ft, gt_name, pred_name, p.confidence, m.iou, g.bbox, p.bbox))

        for gi in agn.unmatched_gt:
            g = gt_by_index[gi]
            errors.append(_err("false_negative", ft, names.get(g.class_id, "unknown"), None, None, None, g.bbox, None))
        for pi in agn.unmatched_pred:
            p = pred_by_index[pi]
            errors.append(_err("false_positive", ft, None, names.get(p.class_id, "unknown"), p.confidence, None, None, p.bbox))

    by_class_metrics: Dict[str, ClassMetrics] = {}
    for cid in sorted(names):
        name = names[cid]
        by_class_metrics[name] = compute_class_metrics(
            per_class_tp.get(cid, 0),
            per_class_fp.get(cid, 0),
            per_class_fn.get(cid, 0),
        )

    return EvaluationReport(
        iou_threshold=iou_threshold,
        by_class=by_class_metrics,
        aggregates=micro_macro(by_class_metrics),
        iou_by_class={names[cid]: iou_stats(ious) for cid, ious in ious_by_class.items()},
        confusion=build_confusion_matrix(confusion_counts, classes=classes),
        errors=errors,
        total_gt=total_gt,
        total_pred=total_pred,
        total_matched=total_matched,
    )


def _merge_counts(a: Dict[tuple, int], b: Dict[tuple, int]) -> Dict[tuple, int]:
    out = dict(a)
    for k, v in b.items():
        out[k] = out.get(k, 0) + v
    return out


def _err(
    kind: str,
    ft: FrameTruth,
    gt_class: Optional[str],
    pred_class: Optional[str],
    confidence: Optional[float],
    iou: Optional[float],
    gt_box: Optional[List[float]],
    pred_box: Optional[List[float]],
) -> ErrorRecord:
    return ErrorRecord(
        kind=kind,
        frame_id=ft.frame_id,
        video_id=ft.video_id,
        frame_index=ft.frame_index,
        width=ft.width,
        height=ft.height,
        timestamp_seconds=ft.timestamp_seconds,
        gt_class=gt_class,
        pred_class=pred_class,
        confidence=confidence,
        iou=iou,
        gt_box=gt_box,
        pred_box=pred_box,
    )
