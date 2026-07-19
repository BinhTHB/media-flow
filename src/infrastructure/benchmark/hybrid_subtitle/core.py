"""Offline-safe primitives for the hybrid subtitle benchmark."""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import median
from typing import Any, Iterable, Sequence


Box = tuple[float, float, float, float]


def _box(value: Sequence[float]) -> Box:
    if len(value) != 4:
        raise ValueError("bbox_xyxy must have four coordinates.")
    x1, y1, x2, y2 = (float(item) for item in value)
    if x2 <= x1 or y2 <= y1:
        raise ValueError("bbox_xyxy must have positive area.")
    return x1, y1, x2, y2


def box_iou(left: Sequence[float], right: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = _box(left)
    bx1, by1, bx2, by2 = _box(right)
    width = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    height = max(0.0, min(ay2, by2) - max(ay1, by1))
    intersection = width * height
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - intersection
    return intersection / union if union else 0.0


def union_box(boxes: Iterable[Sequence[float]]) -> Box:
    values = [_box(box) for box in boxes]
    if not values:
        raise ValueError("At least one box is required.")
    return (
        min(box[0] for box in values),
        min(box[1] for box in values),
        max(box[2] for box in values),
        max(box[3] for box in values),
    )


def percentile_box(boxes: Iterable[Sequence[float]], low: float = 5, high: float = 95) -> Box:
    values = [_box(box) for box in boxes]
    if not values:
        raise ValueError("At least one box is required.")
    if not 0 <= low <= high <= 100:
        raise ValueError("Percentiles must be between zero and 100.")

    def quantile(items: list[float], pct: float) -> float:
        items = sorted(items)
        position = (len(items) - 1) * pct / 100
        lower, upper = math.floor(position), math.ceil(position)
        return items[lower] if lower == upper else items[lower] + (items[upper] - items[lower]) * (position - lower)

    return (
        quantile([box[0] for box in values], low),
        quantile([box[1] for box in values], low),
        quantile([box[2] for box in values], high),
        quantile([box[3] for box in values], high),
    )


def median_box(boxes: Iterable[Sequence[float]]) -> Box:
    values = [_box(box) for box in boxes]
    if not values:
        raise ValueError("At least one box is required.")
    return tuple(median(box[index] for box in values) for index in range(4))  # type: ignore[return-value]


def expand_box(box: Sequence[float], margin: float, width: int, height: int) -> Box:
    x1, y1, x2, y2 = _box(box)
    return max(0.0, x1 - margin), max(0.0, y1 - margin), min(float(width), x2 + margin), min(float(height), y2 + margin)


def normalize_yolo_result(
    *,
    video_id: str,
    frame_index: int,
    timestamp_ms: int,
    bbox_xyxy: Sequence[float],
    confidence: float,
    width: int,
    height: int,
    candidate_index: int,
) -> dict[str, Any]:
    x1, y1, x2, y2 = _box(bbox_xyxy)
    if not 0 <= confidence <= 1:
        raise ValueError("YOLO confidence must be between zero and one.")
    return {
        "video_id": video_id,
        "frame_index": frame_index,
        "timestamp_ms": timestamp_ms,
        "candidate_id": f"F{frame_index:05d}_C{candidate_index:02d}",
        "bbox_xyxy": [x1, y1, x2, y2],
        "bbox_normalized": [x1 / width, y1 / height, x2 / width, y2 / height],
        "confidence": confidence,
        "class": "text",
        "video_dimensions": [width, height],
    }


def group_frame_candidates(candidates: list[dict[str, Any]], horizontal_gap_ratio: float = 2.0) -> list[dict[str, Any]]:
    """Join word boxes on a common baseline, without assuming a subtitle position."""
    remaining = sorted(candidates, key=lambda item: (item["bbox_xyxy"][1], item["bbox_xyxy"][0]))
    groups: list[list[dict[str, Any]]] = []
    while remaining:
        group = [remaining.pop(0)]
        changed = True
        while changed:
            changed = False
            group_box = union_box(item["bbox_xyxy"] for item in group)
            group_height = group_box[3] - group_box[1]
            for item in list(remaining):
                x1, y1, x2, y2 = _box(item["bbox_xyxy"])
                overlap = max(0.0, min(group_box[3], y2) - max(group_box[1], y1))
                vertical_overlap = overlap / min(group_height, y2 - y1)
                gap = max(0.0, max(group_box[0], x1) - min(group_box[2], x2))
                if vertical_overlap >= 0.3 and gap <= horizontal_gap_ratio * max(group_height, y2 - y1):
                    group.append(item)
                    remaining.remove(item)
                    changed = True
        groups.append(group)
    frame_index = candidates[0]["frame_index"] if candidates else 0
    return [
        {
            "candidate_group_id": f"G{frame_index:05d}_{index:02d}",
            "frame_index": frame_index,
            "timestamp_ms": candidates[0]["timestamp_ms"],
            "member_candidate_ids": [item["candidate_id"] for item in group],
            "union_bbox_xyxy": list(union_box(item["bbox_xyxy"] for item in group)),
            "member_count": len(group),
            "mean_detector_confidence": sum(item["confidence"] for item in group) / len(group),
        }
        for index, group in enumerate(groups, 1)
    ]


@dataclass
class TextTrack:
    track_id: str
    groups: list[dict[str, Any]]

    def as_dict(self) -> dict[str, Any]:
        boxes = [group["union_bbox_xyxy"] for group in self.groups]
        return {
            "track_id": self.track_id,
            "start_frame": self.groups[0]["frame_index"],
            "end_frame": self.groups[-1]["frame_index"],
            "start_ms": self.groups[0]["timestamp_ms"],
            "end_ms": self.groups[-1]["timestamp_ms"],
            "frame_count": len(self.groups),
            "representative_bbox_xyxy": list(median_box(boxes)),
            "temporal_union_bbox_xyxy": list(union_box(boxes)),
            "median_bbox_xyxy": list(median_box(boxes)),
            "mean_detector_confidence": sum(group["mean_detector_confidence"] for group in self.groups) / len(self.groups),
            "max_detector_confidence": max(group["mean_detector_confidence"] for group in self.groups),
            "groups": self.groups,
        }


def build_tracks(groups: list[dict[str, Any]], iou_threshold: float = 0.25, max_gap_frames: int = 5) -> list[dict[str, Any]]:
    """Deterministic one-to-one matching between adjacent frame groups."""
    tracks: list[TextTrack] = []
    for group in sorted(groups, key=lambda item: (item["frame_index"], item["candidate_group_id"])):
        group_box = group["union_bbox_xyxy"]
        candidates = [
            track for track in tracks
            if group["frame_index"] - track.groups[-1]["frame_index"] <= max_gap_frames
        ]
        best = max(candidates, key=lambda track: box_iou(track.groups[-1]["union_bbox_xyxy"], group_box), default=None)
        if best is not None and box_iou(best.groups[-1]["union_bbox_xyxy"], group_box) >= iou_threshold:
            best.groups.append(group)
        else:
            tracks.append(TextTrack(track_id=f"T{len(tracks) + 1:03d}", groups=[group]))
    return [track.as_dict() for track in tracks]


def representative_group_ids(track: dict[str, Any], maximum: int = 5) -> list[str]:
    groups = track["groups"]
    if len(groups) <= maximum:
        return [group["candidate_group_id"] for group in groups]
    indices = sorted({0, len(groups) // 2, len(groups) - 1, max(range(len(groups)), key=lambda index: groups[index]["mean_detector_confidence"])})
    return [groups[index]["candidate_group_id"] for index in indices][:maximum]



def box_center(left: Sequence[float], right: Sequence[float]) -> float:
    la = _box(left)
    ra = _box(right)
    lc = ((la[0] + la[2]) / 2, (la[1] + la[3]) / 2)
    rc = ((ra[0] + ra[2]) / 2, (ra[1] + ra[3]) / 2)
    return ((lc[0] - rc[0]) ** 2 + (lc[1] - rc[1]) ** 2) ** 0.5

ALLOWED_CATEGORIES = {
    "dialogue_subtitle", "narration_subtitle", "sound_caption", "karaoke_or_lyric_subtitle",
    "scene_text", "watermark", "logo", "ui_text", "social_overlay", "title_card", "credits",
    "decorative_text", "unknown",
}


def validate_gemini_decision(payload: dict[str, Any], track_id: str, candidate_ids: set[str]) -> dict[str, Any]:
    if payload.get("track_id") != track_id:
        raise ValueError("Gemini response track_id does not match the requested track.")
    decisions = payload.get("decisions")
    if not isinstance(decisions, list) or not decisions:
        raise ValueError("Gemini response requires at least one decision.")
    seen: set[str] = set()
    for decision in decisions:
        candidate_id = decision.get("candidate_id")
        if candidate_id not in candidate_ids:
            raise ValueError("Gemini response included an unknown candidate ID.")
        if candidate_id in seen:
            raise ValueError("Gemini response included a duplicate candidate ID.")
        seen.add(candidate_id)
        if decision.get("category") not in ALLOWED_CATEGORIES:
            raise ValueError("Gemini response included an invalid category.")
        if not isinstance(decision.get("is_subtitle"), bool):
            raise ValueError("Gemini response is_subtitle must be boolean.")
        confidence = decision.get("confidence")
        if not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
            raise ValueError("Gemini response confidence must be between zero and one.")
        if any(key in decision for key in ("bbox", "bbox_xyxy", "box_2d")):
            raise ValueError("Gemini semantic verifier must not generate geometry.")
    return payload


def load_reviewed_ground_truth(path: str) -> dict[str, Any] | None:
    """Load independently reviewed benchmark labels, never model output."""
    import json
    from pathlib import Path

    file_path = Path(path)
    if not file_path.is_file():
        return None
    payload = json.loads(file_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("review_status") != "reviewed":
        return None
    if not isinstance(payload.get("annotations"), list):
        return None
    for item in payload["annotations"]:
        if not isinstance(item, dict) or not isinstance(item.get("is_subtitle"), bool):
            return None
        _box(item.get("bbox_xyxy", []))
    return payload


def proposal_metrics(
    tracks: Sequence[dict[str, Any]],
    reviewed_ground_truth: dict[str, Any] | None,
    *,
    iou_threshold: float = 0.5,
) -> dict[str, Any] | None:
    """Evaluate local track geometry only against reviewed human annotations."""
    if reviewed_ground_truth is None:
        return None
    expected = [
        item for item in reviewed_ground_truth["annotations"]
        if item["is_subtitle"]
    ]
    remaining = set(range(len(tracks)))
    matched = 0
    ious: list[float] = []
    for annotation in expected:
        best = max(
            remaining,
            key=lambda index: box_iou(annotation["bbox_xyxy"], tracks[index]["representative_bbox_xyxy"]),
            default=None,
        )
        if best is not None:
            iou = box_iou(annotation["bbox_xyxy"], tracks[best]["representative_bbox_xyxy"])
            if iou >= iou_threshold:
                remaining.remove(best)
                matched += 1
                ious.append(iou)
    false_positives = len(remaining)
    false_negatives = len(expected) - matched
    precision = matched / (matched + false_positives) if matched + false_positives else 0.0
    recall = matched / len(expected) if expected else 0.0
    return {
        "ground_truth_source": "independent_manual_review",
        "iou_threshold": iou_threshold,
        "reviewed_subtitle_annotations": len(expected),
        "proposed_tracks": len(tracks),
        "true_positives": matched,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "mean_matched_iou": sum(ious) / len(ious) if ious else 0.0,
    }
