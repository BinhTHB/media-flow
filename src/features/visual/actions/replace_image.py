"""
Replace Image Action

Overlays a PNG/JPEG asset (with alpha channel support) fitted to a frame region.
"""

from __future__ import annotations

import os
from typing import Any, Optional

from src.core.models.bbox import BBox


class ReplaceImageAction:
    def __init__(
        self,
        image_asset_path: str,
        opacity: float = 1.0,
        fit: str = "contain",
        padding_px: int = 4,
    ):
        self.image_asset_path = image_asset_path
        self.opacity = opacity
        self.fit = fit
        self.padding_px = padding_px
        self._asset_cache: Optional[Any] = None

    def execute(self, frame: Any, bbox: BBox) -> Any:
        import cv2

        if not self.image_asset_path or not os.path.exists(self.image_asset_path):
            return frame

        if self._asset_cache is None:
            self._asset_cache = cv2.imread(self.image_asset_path, cv2.IMREAD_UNCHANGED)

        asset = self._asset_cache
        if asset is None:
            return frame

        x1, y1, x2, y2 = bbox.xyxy_int()
        x1 = max(0, x1 - self.padding_px)
        y1 = max(0, y1 - self.padding_px)
        x2 = min(bbox.frame_width, x2 + self.padding_px)
        y2 = min(bbox.frame_height, y2 + self.padding_px)

        target_w = x2 - x1
        target_h = y2 - y1

        if target_w <= 0 or target_h <= 0:
            return frame

        resized = cv2.resize(asset, (target_w, target_h), interpolation=cv2.INTER_AREA)

        if resized.shape[2] == 4:
            alpha = (resized[:, :, 3] / 255.0) * float(self.opacity)
            for c in range(3):
                frame[y1:y2, x1:x2, c] = (
                    alpha * resized[:, :, c] + (1.0 - alpha) * frame[y1:y2, x1:x2, c]
                )
        else:
            frame[y1:y2, x1:x2] = resized[:, :, :3]

        return frame
