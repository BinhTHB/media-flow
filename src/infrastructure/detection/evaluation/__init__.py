"""Ground-truth evaluation for the YOLO burn-in detector.

This package contains the metric engine used to score the detector against
*independent* ground-truth annotations. It is intentionally framework-agnostic:
it only needs per-frame predictions and per-frame ground-truth boxes expressed
as absolute pixel boxes and operates purely on plain Python objects, so it runs
offline, cheaply, and deterministically.

Ground-truth facts are loaded from a canonical JSON schema (or YOLO ``.txt``).
Predictions come from the predictions JSONL produced by the evaluator script.
The matching engine (:mod:`matching`) uses a greedy one-to-one, descending-IoU
strategy that never matches the same box twice; :mod:`report` turns those
matches into the full metric bundle.

This package deliberately does **not** fabricate ground truth: if only
machine-generated pre-labels exist (marked ``UNVERIFIED``), no Precision /
Recall / F1 / confusion-matrix numbers may be reported as real. That status
check lives in the evaluator script.
"""

from src.infrastructure.detection.evaluation.matching import (
    Box,
    ClassCount,
    FrameMatch,
    Match,
    MatchResult,
    greedy_class_agnostic_match,
    greedy_class_aware_match,
    match_frame,
    per_class_counts,
)
from src.infrastructure.detection.evaluation.metrics import (
    ClassMetrics,
    average_metrics,
    compute_class_metrics,
    iou_stats,
    micro_macro,
)
from src.infrastructure.detection.evaluation.confusion import (
    CONFUSION_BACKGROUND,
    ConfusionMatrix,
    build_confusion_matrix,
)
from src.infrastructure.detection.evaluation.report import (
    ErrorRecord,
    EvaluationReport,
    FrameTruth,
    evaluate_frames,
)

__all__ = [
    "Box",
    "Match",
    "MatchResult",
    "ClassCount",
    "FrameMatch",
    "greedy_class_aware_match",
    "greedy_class_agnostic_match",
    "match_frame",
    "per_class_counts",
    "ClassMetrics",
    "compute_class_metrics",
    "micro_macro",
    "average_metrics",
    "ConfusionMatrix",
    "build_confusion_matrix",
    "CONFUSION_BACKGROUND",
    "FrameTruth",
    "EvaluationReport",
    "ErrorRecord",
    "evaluate_frames",
]
