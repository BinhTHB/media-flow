"""Confusion matrix for the 4-class burn-in detector plus background."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

#: Label used for the "no object" row/column of the confusion matrix.
#: A ground-truth box unmatched by any prediction lands in the
#: ``(gt_class, background)`` cell (a miss). A prediction that matches no
#: ground-truth box lands in the ``(background, pred_class)`` cell (a false
#: positive).
CONFUSION_BACKGROUND = "background"


@dataclass
class ConfusionMatrix:
    """Row = ground truth class, column = predicted class.

    The last row/column is the background class. Values are counts; row totals
    are the GT box counts and column totals are the prediction counts.
    """

    classes: List[str]
    matrix: List[List[int]]  # row=gt, col=pred; last row/col = background
    row_labels: List[str]
    col_labels: List[str]

    def get(self, gt_class: str, pred_class: str) -> int:
        try:
            ri = self.row_labels.index(gt_class)
        except ValueError:
            return 0
        try:
            ci = self.col_labels.index(pred_class)
        except ValueError:
            return 0
        return self.matrix[ri][ci]

    def row_total(self, gt_class: str) -> int:
        try:
            ri = self.row_labels.index(gt_class)
        except ValueError:
            return 0
        return sum(self.matrix[ri][ci] for ci in range(len(self.col_labels)))


def build_confusion_matrix(
    counts: Dict[Tuple[str, str], int],
    *,
    classes: Sequence[str],
) -> ConfusionMatrix:
    """Build the matrix from an explicit ``(gt_class, pred_class) -> count`` map.

    ``classes`` defines the ordered class labels (without background, which is
    appended automatically). Background interactions are keyed by
    :data:`CONFUSION_BACKGROUND`.
    """
    labels = [str(c) for c in classes] + [CONFUSION_BACKGROUND]
    n = len(labels)
    matrix: List[List[int]] = [[0] * n for _ in range(n)]
    for (gt, pred), cnt in counts.items():
        gt_s = str(gt)
        pred_s = str(pred)
        ri = labels.index(gt_s) if gt_s in labels else labels.index(CONFUSION_BACKGROUND)
        ci = labels.index(pred_s) if pred_s in labels else labels.index(CONFUSION_BACKGROUND)
        matrix[ri][ci] += int(cnt)
    return ConfusionMatrix(classes=list(labels[:-1]), matrix=matrix,
                           row_labels=labels, col_labels=labels)


def accumulate_confusion(
    gt_by_class: Dict[int, Sequence[object]],
    match_result,
    class_names: Dict[int, str],
) -> Dict[Tuple[str, str], int]:
    """Accumulate a confusion matrix from a class-agnostic match result.

    ``gt_by_class`` maps class id -> list of ``(index, Box)`` pairs so the
    unmatched ground-truth boxes can be placed into the background column.
    ``match_result`` is a :class:`~src.infrastructure.detection.evaluation.matching.MatchResult`.
    """
    counts: Dict[Tuple[str, str], int] = {}
    index_to_class: Dict[int, int] = {}
    for cid, items in gt_by_class.items():
        for idx, _box in items:
            index_to_class[idx] = cid

    for m in match_result.matches:
        gt_name = class_names.get(m.gt_class_id, CONFUSION_BACKGROUND)
        pred_name = class_names.get(m.pred_class_id, CONFUSION_BACKGROUND)
        key = (gt_name, pred_name)
        counts[key] = counts.get(key, 0) + 1
    for gi in match_result.unmatched_gt:
        cid = index_to_class.get(gi, -1)
        gt_name = class_names.get(cid, CONFUSION_BACKGROUND)
        key = (gt_name, CONFUSION_BACKGROUND)
        counts[key] = counts.get(key, 0) + 1
    for pi in match_result.unmatched_pred:
        key = (CONFUSION_BACKGROUND, class_names.get(pi, CONFUSION_BACKGROUND))
        counts[key] = counts.get(key, 0) + 1
    return counts
