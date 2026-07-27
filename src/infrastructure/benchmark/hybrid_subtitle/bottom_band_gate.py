"""Configurable geometry gate for bottom-band dialogue subtitle scope."""
from __future__ import annotations

from typing import Any, Mapping


def track_y1_norm(track: Mapping[str, Any], frame_height: int) -> float | None:
    """Return normalized top edge of a track's representative bbox."""
    if frame_height <= 0:
        return None
    bbox = (
        track.get("median_bbox_xyxy")
        or track.get("temporal_union_bbox_xyxy")
        or []
    )
    if len(bbox) != 4:
        return None
    return float(bbox[1]) / float(frame_height)


def passes_bottom_band(
    track: Mapping[str, Any], frame_height: int, y1_norm_threshold: float
) -> bool:
    """Fail closed when geometry is unavailable; pass only bottom-band tracks."""
    y1_norm = track_y1_norm(track, frame_height)
    return y1_norm is not None and y1_norm >= y1_norm_threshold
