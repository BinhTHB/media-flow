from __future__ import annotations

from typing import Optional

from src.application.services.adaptive_subtitle_region_service import BoundingBox
from src.shared.exceptions import ProviderError


class OpenCVCoarseSubtitleROIProvider:
    """Optional bright-text/edge candidate generator; never selects final masks."""

    def detect(self, image_path: str) -> Optional[BoundingBox]:
        try:
            import cv2
        except ImportError as exc:
            raise ProviderError("opencv-python is not installed") from exc
        image = cv2.imread(image_path)
        if image is None:
            raise ProviderError(f"Failed to load frame image: {image_path}")
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        bright = cv2.inRange(hsv, (0, 0, 175), (179, 100, 255))
        edges = cv2.Sobel(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.CV_8U, 1, 0)
        candidate = cv2.bitwise_and(bright, cv2.threshold(edges, 30, 255, cv2.THRESH_BINARY)[1])
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
        candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, kernel)
        contours = cv2.findContours(candidate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        boxes = []
        height, width = image.shape[:2]
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w >= width * 0.015 and h >= height * 0.008 and w > h:
                boxes.append(BoundingBox(x, y, x + w, y + h))
        return BoundingBox.union(boxes) if boxes else None
