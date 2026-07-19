from __future__ import annotations
import logging
from typing import Any, Dict, List, Optional, Set
from .core import box_iou, box_center
logger = logging.getLogger(__name__)

ALLOWED_CATEGORIES = {
    "dialogue_subtitle", "narration_subtitle", "sound_caption",
    "karaoke_or_lyric_subtitle", "scene_text", "watermark", "logo",
    "ui_text", "social_overlay", "title_card", "credits",
    "decorative_text", "unknown"
}


def evaluate_tracks_vs_gt(
    predicted_tracks: List[Dict[str, Any]],
    ground_truth: List[Dict[str, Any]],
    iou_threshold: float = 0.5,
) -> Dict[str, Any]:
    """Compute precision/recall/F1 by matching predicted subtitle tracks vs GT.
    Only works when ground_truth is a reviewed per-frame annotation set.
    """
    if not ground_truth:
        return {
            "ground_truth_available": False,
            "note": "No ground truth loaded. Review and create manual GT annotations first.",
        }

    # Build GT map: frame_index -> list of subtitle bboxes
    gt_by_frame: Dict[int, List[List[float]]] = {}
    for entry in ground_truth:
        fi = entry.get("frame_index")
        if fi is None:
            continue
        for region in entry.get("subtitle_regions", []):
            gt_by_frame.setdefault(fi, []).append(region["bbox_xyxy"])

    if not gt_by_frame:
        return {
            "ground_truth_available": True,
            "empty_gt": True,
            "precision": 0.0, "recall": 0.0, "f1": 0.0,
        }

    # For each predicted track, check which GT frames it covers
    tp, fp, fn = 0, 0, 0
    gt_covered: Set[int] = set()
    for track in predicted_tracks:
        frames = track.get("frames", [])
        bboxes = track.get("bboxes", [])
        # Find matching GT frames
        matched_gt_frames = 0
        total_gt_frames = sum(1 for fi in frames if fi in gt_by_frame)
        for fi, bbox in zip(frames, bboxes):
            if fi not in gt_by_frame:
                continue
            best_iou = max(
                (bbox_iou(bbox, gt_box) for gt_box in gt_by_frame[fi]),
                default=0.0
            )
            if best_iou >= iou_threshold:
                matched_gt_frames += 1
                gt_covered.add(fi)
        coverage = matched_gt_frames / max(total_gt_frames, 1)
        if coverage >= 0.5:
            tp += 1
        else:
            fp += 1

    all_gt_frames = set(gt_by_frame.keys())
    fn = len(all_gt_frames - gt_covered)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    return {
        "ground_truth_available": True,
        "true_positive_tracks": tp,
        "false_positive_tracks": fp,
        "missed_gt_frames": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "total_predicted_tracks": len(predicted_tracks),
        "total_gt_frames": len(all_gt_frames),
        "iou_threshold": iou_threshold,
    }


def compute_timing_mae(
    predicted: List[Dict[str, Any]],
    reference: List[Dict[str, Any]],
) -> Dict[str, float]:
    if not predicted or not reference:
        return {"timing_mae_ms": 0.0, "timing_mae_frames": 0.0}
    errors_start = [abs(p["start_ms"] - r.get("start_ms", 0))
                   for p, r in zip(predicted, reference)]
    errors_end = [abs(p["end_ms"] - r.get("end_ms", 0))
                 for p, r in zip(predicted, reference)]
    all_errors = errors_start + errors_end
    return {
        "timing_mae_ms": round(sum(all_errors) / max(len(all_errors), 1), 1),
        "timing_mae_frames": round(sum(all_errors) / max(len(all_errors), 1) / 33.33, 2),
    }


def compute_bbox_metrics(
    predicted: List[List[float]],
    reference: List[List[float]],
) -> Dict[str, float]:
    if not predicted or not reference:
        return {"mean_iou": 0.0, "mean_center_dist": 0.0}
    ious = [bbox_iou(p, r) for p, r in zip(predicted, reference)]
    cds = [box_center(p, r) for p, r in zip(predicted, reference)]
    return {
        "mean_iou": round(sum(ious) / len(ious), 4),
        "mean_center_dist": round(sum(cds) / len(cds), 2),
    }


def load_manual_gt(path: str) -> List[Dict[str, Any]]:
    import json
    if not path:
        return []
    from pathlib import Path
    if not Path(path).exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    return data.get("annotations", [])
