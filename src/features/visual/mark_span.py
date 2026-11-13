"""Box a watermark, logo, or other burned-in mark occupies on one frame.

A face credit is often detected on only a few frames of a shot. The frames
between two nearby sightings still show the same glyphs, so the box is the
straight line between those sightings.
"""

from __future__ import annotations

from typing import Optional, Sequence

from src.core.models.bbox import BBox
from src.core.models.detection import Detection


def bbox_at_frame(detections: Sequence[Detection], frame_index: int) -> Optional[BBox]:
    """Return the mark box on ``frame_index``, or None outside the sightings."""
    if not detections:
        return None
    ordered = sorted(detections, key=lambda det: det.frame_index)
    exact = [det.bbox for det in ordered if det.frame_index == frame_index]
    if exact:
        return exact[0] if len(exact) == 1 else _union(exact)
    earlier = [det for det in ordered if det.frame_index < frame_index]
    later = [det for det in ordered if det.frame_index > frame_index]
    if not earlier or not later:
        return None
    left = earlier[-1]
    right = later[0]
    span = right.frame_index - left.frame_index
    if span <= 0:
        return left.bbox
    weight = (frame_index - left.frame_index) / float(span)
    return BBox(
        x1=_lerp(left.bbox.x1, right.bbox.x1, weight),
        y1=_lerp(left.bbox.y1, right.bbox.y1, weight),
        x2=_lerp(left.bbox.x2, right.bbox.x2, weight),
        y2=_lerp(left.bbox.y2, right.bbox.y2, weight),
        frame_width=left.bbox.frame_width,
        frame_height=left.bbox.frame_height,
    )


def _lerp(start: float, end: float, weight: float) -> float:
    return start + (end - start) * weight


def _union(boxes: Sequence[BBox]) -> BBox:
    return BBox(
        x1=min(box.x1 for box in boxes),
        y1=min(box.y1 for box in boxes),
        x2=max(box.x2 for box in boxes),
        y2=max(box.y2 for box in boxes),
        frame_width=boxes[0].frame_width,
        frame_height=boxes[0].frame_height,
    )
