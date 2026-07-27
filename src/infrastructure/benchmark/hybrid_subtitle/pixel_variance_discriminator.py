"""Fail-closed reject-only temporal pixel-variance discriminator."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import cv2
import numpy as np


@dataclass(frozen=True)
class PixelVarianceEvidence:
    temporal_pixel_variance: float | None
    sample_count: int
    rejected: bool
    reason: str


class PixelVarianceDiscriminator:
    """Reject low-variance static tracks; errors remain unconfirmed upstream."""

    def __init__(self, min_temporal_variance: float = 827.12) -> None:
        self.min_temporal_variance = min_temporal_variance

    def evaluate(self, track: Mapping[str, Any], bundle: Any) -> PixelVarianceEvidence:
        try:
            crops = []
            for group in track.get("groups", []):
                frame = bundle.frame_for_index(group["frame_index"]).image
                bbox = group.get("union_bbox_xyxy", [])
                if frame is None or len(bbox) != 4:
                    continue
                height, width = frame.shape[:2]
                x1, y1, x2, y2 = [int(round(v)) for v in bbox]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(width, x2), min(height, y2)
                if x2 - x1 < 2 or y2 - y1 < 2:
                    continue
                crops.append(cv2.resize(frame[y1:y2, x1:x2], (96, 48), interpolation=cv2.INTER_AREA))
            if len(crops) < 2:
                return PixelVarianceEvidence(None, len(crops), True, "insufficient_temporal_crops")
            temporal_variance = float(np.mean(np.var(np.stack(crops).astype(np.float32), axis=0)))
            return PixelVarianceEvidence(
                temporal_variance,
                len(crops),
                temporal_variance < self.min_temporal_variance,
                "low_temporal_pixel_variance" if temporal_variance < self.min_temporal_variance else "passed",
            )
        except Exception:
            return PixelVarianceEvidence(None, 0, True, "pixel_variance_error")
