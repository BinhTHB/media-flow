"""One-to-one bounding-box matching for detection evaluation.

Two matching modes are provided:

* :func:`greedy_class_aware_match` — only *same-class* prediction / ground-truth
  pairs may match. This is what per-class Precision / Recall / F1 are computed
  from (a prediction that overlaps a GT box but has the wrong class is a FP for
  its own class and a FN for the GT class — it is *not* a true positive).
* :func:`greedy_class_agnostic_match` — any pair above the IoU threshold may
  match regardless of class. This is what the confusion matrix is built from, so
  that ``watermark`` matched to a ``logo`` ground-truth box shows up as a
  confusion instead of a plain miss.

Both use the same greedy, descending-IoU, one-to-one algorithm: each GT box and
each prediction is matched at most once, so a single prediction can never satisfy
two different ground-truth boxes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from src.infrastructure.detection.postprocess import box_iou


@dataclass
class Box:
    """A single box participating in matching.

    ``class_id`` is the 4-class id (0..3). ``bbox`` is an ``[x1, y1, x2, y2]``
    absolute pixel box. ``confidence`` is only meaningful for predictions.
    """

    class_id: int
    bbox: Sequence[float]
    confidence: float = 0.0


@dataclass
class Match:
    """A selected one-to-one prediction<->ground-truth pairing."""

    gt_index: int
    pred_index: int
    iou: float
    gt_class_id: int
    pred_class_id: int
    confidence: float


@dataclass
class MatchResult:
    """Result of matching one frame (or one class)."""

    matches: List[Match]
    unmatched_gt: List[int]  # indices into ground truth
    unmatched_pred: List[int]  # indices into predictions

    @property
    def matched(self) -> int:
        return len(self.matches)


@dataclass
class FrameMatch:
    """Class-aware counts for a single frame, grouped by class."""

    by_class: Dict[int, "ClassCount"]

    def get(self, class_id: int) -> "ClassCount":
        return self.by_class.get(class_id, ClassCount())


@dataclass
class ClassCount:
    tp: int = 0
    fp: int = 0
    fn: int = 0


def _select_pairs(pairs: List[Tuple[float, int, int]]) -> Tuple[List[Match], set, set]:
    """Greedy descending-IoU one-to-one selection.

    ``pairs`` is a list of ``(iou, gt_index, pred_index)`` candidates. Returns
    the selected matches plus the set of used ground-truth and prediction
    indices. Each GT and each prediction claiming at most once.
    """
    pairs_sorted = sorted(pairs, key=lambda p: p[0], reverse=True)
    selected: List[Match] = []
    used_gt: set = set()
    used_pred: set = set()
    for iou, gi, pi in pairs_sorted:
        if gi in used_gt or pi in used_pred:
            continue
        used_gt.add(gi)
        used_pred.add(pi)
        selected.append((gi, pi, iou))
    return selected, used_gt, used_pred


def greedy_class_aware_match(
    gt: Sequence[Box],
    pred: Sequence[Box],
    iou_threshold: float = 0.5,
) -> MatchResult:
    """Match same-class GT / predictions one-to-one by descending IoU."""
    pairs: List[Tuple[float, int, int]] = []
    for gi, g in enumerate(gt):
        for pi, p in enumerate(pred):
            if g.class_id != p.class_id:
                continue
            iou = box_iou(g.bbox, p.bbox)
            if iou >= iou_threshold:
                pairs.append((iou, gi, pi))
    selected, used_gt, used_pred = _select_pairs(pairs)
    matches = [
        Match(
            gt_index=gi,
            pred_index=pi,
            iou=iou,
            gt_class_id=gt[gi].class_id,
            pred_class_id=pred[pi].class_id,
            confidence=pred[pi].confidence,
        )
        for gi, pi, iou in selected
    ]
    return MatchResult(
        matches=matches,
        unmatched_gt=[i for i in range(len(gt)) if i not in used_gt],
        unmatched_pred=[i for i in range(len(pred)) if i not in used_pred],
    )


def greedy_class_agnostic_match(
    gt: Sequence[Box],
    pred: Sequence[Box],
    iou_threshold: float = 0.5,
) -> MatchResult:
    """Match by IoU alone (class ignored) — for confusion-matrix analysis."""
    pairs: List[Tuple[float, int, int]] = []
    for gi, g in enumerate(gt):
        for pi, p in enumerate(pred):
            iou = box_iou(g.bbox, p.bbox)
            if iou >= iou_threshold:
                pairs.append((iou, gi, pi))
    selected, used_gt, used_pred = _select_pairs(pairs)
    matches = [
        Match(
            gt_index=gi,
            pred_index=pi,
            iou=iou,
            gt_class_id=gt[gi].class_id,
            pred_class_id=pred[pi].class_id,
            confidence=pred[pi].confidence,
        )
        for gi, pi, iou in selected
    ]
    return MatchResult(
        matches=matches,
        unmatched_gt=[i for i in range(len(gt)) if i not in used_gt],
        unmatched_pred=[i for i in range(len(pred)) if i not in used_pred],
    )


def match_frame(
    gt: Sequence[Box],
    pred: Sequence[Box],
    iou_threshold: float = 0.5,
    *,
    class_aware: bool = True,
) -> MatchResult:
    """Convenience wrapper; delegates to the requested matching strategy."""
    if class_aware:
        return greedy_class_aware_match(gt, pred, iou_threshold=iou_threshold)
    return greedy_class_agnostic_match(gt, pred, iou_threshold=iou_threshold)


def per_class_counts(
    gt: Sequence[Box],
    pred: Sequence[Box],
    iou_threshold: float = 0.5,
) -> Dict[int, ClassCount]:
    """Return TP/FP/FN per class for one frame using class-aware matching."""
    res = greedy_class_aware_match(gt, pred, iou_threshold=iou_threshold)
    counts: Dict[int, ClassCount] = {}
    for m in res.matches:
        c = counts.setdefault(m.gt_class_id, ClassCount())
        c.tp += 1
    for gi in res.unmatched_gt:
        c = counts.setdefault(gt[gi].class_id, ClassCount())
        c.fn += 1
    for pi in res.unmatched_pred:
        c = counts.setdefault(pred[pi].class_id, ClassCount())
        c.fp += 1
    return counts
