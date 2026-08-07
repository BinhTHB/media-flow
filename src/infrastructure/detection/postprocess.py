"""Post-processing rules for detections.

All magic thresholds live in config or central constants, not scattered in code.
Rules are framework-agnostic and operate on :class:`Detection` objects.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from src.infrastructure.detection.schemas import Detection

#: Minimum box dimension (px) below which a box is discarded as noise.
MIN_BOX_SIDE_PX: float = 3.0

#: IoU above which two same-class boxes are considered duplicates.
DUPLICATE_IOU_THRESHOLD: float = 0.80

#: Per-class expansion margins (normalized to the smaller frame dimension).
CLASS_EXPANSION_RATIO: Dict[str, float] = {
    "subtitle": 0.04,
    "watermark": 0.04,
    "logo": 0.04,
    "other_burn_in_text": 0.04,
}


def compute_normalized(bbox_xyxy: Sequence[float], width: int, height: int) -> List[float]:
    """Convert absolute pixel box to normalized coordinates (clamped to [0,1])."""
    w = max(1, int(width))
    h = max(1, int(height))
    x1, y1, x2, y2 = (float(v) for v in bbox_xyxy)
    norm = [
        max(0.0, min(1.0, x1 / w)),
        max(0.0, min(1.0, y1 / h)),
        max(0.0, min(1.0, x2 / w)),
        max(0.0, min(1.0, y2 / h)),
    ]
    return norm


def clamp_box(
    bbox_xyxy: Sequence[float], width: int, height: int
) -> Tuple[float, float, float, float]:
    """Clamp a box into frame bounds, returning a positive-area box."""
    x1, y1, x2, y2 = (float(v) for v in bbox_xyxy)
    x1 = max(0.0, min(float(width), x1))
    y1 = max(0.0, min(float(height), y1))
    x2 = max(0.0, min(float(width), x2))
    y2 = max(0.0, min(float(height), y2))
    if x2 <= x1:
        x2 = min(float(width), x1 + 1.0)
    if y2 <= y1:
        y2 = min(float(height), y1 + 1.0)
    return x1, y1, x2, y2


def box_area(bbox_xyxy: Sequence[float]) -> float:
    x1, y1, x2, y2 = (float(v) for v in bbox_xyxy)
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def box_iou(left: Sequence[float], right: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = (float(v) for v in left)
    bx1, by1, bx2, by2 = (float(v) for v in right)
    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = inter_w * inter_h
    area_a = box_area(left)
    area_b = box_area(right)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def filter_by_confidence(
    detections: Iterable[Detection],
    confidence_by_class: Dict[str, float],
) -> List[Detection]:
    """Filter detections whose confidence >= the per-class threshold."""
    out: List[Detection] = []
    for det in detections:
        threshold = confidence_by_class.get(det.class_name, 0.0)
        if det.confidence >= threshold:
            out.append(det)
    return out


def filter_invalid_boxes(
    detections: Iterable[Detection], min_side: float = MIN_BOX_SIDE_PX
) -> List[Detection]:
    """Drop boxes that are too small or have non-positive area."""
    out: List[Detection] = []
    for det in detections:
        if box_area(det.bbox_xyxy) <= 0.0:
            continue
        x1, y1, x2, y2 = det.bbox_xyxy
        if (x2 - x1) < min_side or (y2 - y1) < min_side:
            continue
        out.append(det)
    return out


def remove_duplicates(
    detections: Sequence[Detection], iou_threshold: float = DUPLICATE_IOU_THRESHOLD
) -> List[Detection]:
    """Remove high-IoU duplicate boxes *within the same class*.

    Keeps the higher-confidence box. Detections of different classes are never
    merged even if they overlap.
    """
    kept: List[Detection] = []
    for det in sorted(detections, key=lambda d: d.confidence, reverse=True):
        duplicate = False
        for other in kept:
            if other.class_id != det.class_id:
                continue
            if box_iou(other.bbox_xyxy, det.bbox_xyxy) >= iou_threshold:
                duplicate = True
                break
        if not duplicate:
            kept.append(det)
    return kept


def expand_box(
    bbox_xyxy: Sequence[float],
    class_name: str,
    width: int,
    height: int,
    expansion_ratio: Optional[Dict[str, float]] = None,
) -> List[float]:
    """Expand a box by a per-class ratio of the smaller frame dimension."""
    x1, y1, x2, y2 = (float(v) for v in bbox_xyxy)
    ratio_map = expansion_ratio or CLASS_EXPANSION_RATIO
    ratio = ratio_map.get(class_name, 0.04)
    ref = min(width, height)
    pad_x = ratio * ref
    pad_y = ratio * ref
    return clamp_box((x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y), width, height)


def _merge_boxes(boxes: Sequence[Sequence[float]]) -> List[float]:
    x1 = min(b[0] for b in boxes)
    y1 = min(b[1] for b in boxes)
    x2 = max(b[2] for b in boxes)
    y2 = max(b[3] for b in boxes)
    return [x1, y1, x2, y2]


def group_close_boxes_by_class(
    detections: Sequence[Detection],
    gap_ratio: float = 2.0,
) -> List[Detection]:
    """Merge boxes of the *same class* that are spatially close.

    Unlike duplicate removal (overlap-based), this merges nearby disjoint boxes
    whose separation is below ``gap_ratio`` times the average box size. This is
    useful to build a mask to remove whole subtitle/watermark lines. Boxes of
    different classes are never merged.
    """
    by_class: Dict[int, List[Detection]] = {}
    for det in detections:
        by_class.setdefault(det.class_id, []).append(det)

    merged: List[Detection] = []
    for class_id, items in by_class.items():
        remaining = sorted(
            items, key=lambda d: (d.bbox_xyxy[1], d.bbox_xyxy[0])
        )
        while remaining:
            seed = remaining.pop(0)
            group = [seed]
            seed_box = list(seed.bbox_xyxy)
            changed = True
            while changed:
                changed = False
                to_remove: List[Detection] = []
                for other in remaining:
                    if _boxes_close(seed_box, other.bbox_xyxy, gap_ratio):
                        group.append(other)
                        seed_box = _merge_boxes([seed_box, other.bbox_xyxy])
                        to_remove.append(other)
                        changed = True
                for other in to_remove:
                    if other in remaining:
                        remaining.remove(other)
            if len(group) == 1:
                merged.extend(group)
            else:
                box = _merge_boxes([g.bbox_xyxy for g in group])
                first = sorted(group, key=lambda g: g.confidence, reverse=True)[0]
                merged.append(
                    Detection(
                        class_id=first.class_id,
                        class_name=first.class_name,
                        confidence=max(g.confidence for g in group),
                        bbox_xyxy=box,
                        bbox_normalized=compute_normalized(
                            box, first.frame_width, first.frame_height
                        ),
                        frame_index=first.frame_index,
                        timestamp_seconds=first.timestamp_seconds,
                        frame_width=first.frame_width,
                        frame_height=first.frame_height,
                        region_id=first.region_id,
                        group_id=f"group_{class_id}_{first.frame_index}",
                        track_id=first.track_id,
                    )
                )
    return merged


def _horizontal_gap(left: Sequence[float], right: Sequence[float]) -> float:
    """Horizontal separation between two boxes.

    Returns a positive value when the boxes do not horizontally overlap, and a
    non-positive value when they overlap.
    """
    (a1, a2), (b1, b2) = (left[0], left[2]), (right[0], right[2])
    if a1 <= b1:
        return b1 - a2
    return a1 - b2


def _boxes_close(left: Sequence[float], right: Sequence[float], gap_ratio: float) -> bool:
    lx1, ly1, lx2, ly2 = (float(v) for v in left)
    rx1, ry1, rx2, ry2 = (float(v) for v in right)
    gap = _horizontal_gap(left, right)
    if gap <= 0.0:
        return False  # horizontally overlapping; handled by duplicate removal
    size_l = max((lx2 - lx1), (ly2 - ly1), 1.0)
    h_overlap = min(ly2, ry2) - max(ly1, ry1)
    if h_overlap <= 0.0:
        return False  # disjoint baselines (different rows)
    return gap <= gap_ratio * size_l


def postprocess_frame(
    detections: Sequence[Detection],
    confidence_by_class: Dict[str, float],
    *,
    width: Optional[int] = None,
    height: Optional[int] = None,
    expand: bool = True,
    group_close: bool = True,
    gap_ratio: float = 2.0,
) -> List[Detection]:
    """Run the full post-processing chain for one frame's raw detections.

    Order: per-class confidence filter -> clamp -> invalid-box filter ->
    duplicate removal -> (optional) per-class expansion -> (optional) grouping.
    """
    filtered = filter_by_confidence(detections, confidence_by_class)

    clamped: List[Detection] = []
    for det in filtered:
        fw = width if width is not None else det.frame_width
        fh = height if height is not None else det.frame_height
        c = det._with_bbox(*clamp_box(det.bbox_xyxy, fw, fh))
        clamped.append(c)

    valid = filter_invalid_boxes(clamped)

    if expand:
        expanded: List[Detection] = []
        for det in valid:
            fw = width if width is not None else det.frame_width
            fh = height if height is not None else det.frame_height
            box = expand_box(det.bbox_xyxy, det.class_name, fw, fh)
            expanded.append(det._with_bbox(*box))
        valid = expanded

    deduped = remove_duplicates(valid)

    if group_close:
        return group_close_boxes_by_class(deduped, gap_ratio=gap_ratio)
    return deduped


def filter_by_detect_policy(
    detections: Iterable[Detection],
    policy: Dict[str, Dict[str, bool]],
) -> List[Detection]:
    """Drop detections whose class has ``detect: false`` in the class policy.

    Separates *detection* from *action*: classes with ``detect: false`` are not
    emitted at all, while classes with ``safe_cover: false`` are still detected
    but must not be covered downstream. Default policy keeps every class both
    detected and coverable (legacy behavior).
    """
    out: List[Detection] = []
    for det in detections:
        entry = (policy or {}).get(det.class_name, {})
        if not entry.get("detect", True):
            continue
        out.append(det)
    return out


def class_safe_cover(
    class_name: str,
    policy: Optional[Dict[str, Dict[str, bool]]] = None,
) -> bool:
    """Return whether a class is allowed to be covered under the policy."""
    entry = (policy or {}).get(class_name, {})
    return bool(entry.get("safe_cover", True))
