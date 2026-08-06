"""Abstract detector interface (burn-in text detection)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

from src.infrastructure.detection.schemas import DetectionResult


class BurnInDetector(ABC):
    """Interface for a burn-in text detector.

    Concrete implementations (e.g. YOLO) are pluggable. The interface accepts
    either raw frames or a whole-frame batch and returns framework-agnostic
    :class:`DetectionResult` objects.
    """

    @abstractmethod
    def detect_frame(
        self, bgr_image, *, frame_index: int = 0, timestamp_seconds: float = 0.0
    ) -> DetectionResult:
        """Detect burn-in text on a single BGR image."""

    @abstractmethod
    def detect_batch(
        self, bgr_images, *, frame_index_offset: int = 0, timestamps_seconds: Optional[List[float]] = None
    ) -> List[DetectionResult]:
        """Detect burn-in text on a batch of BGR images."""

    @property
    @abstractmethod
    def device(self) -> str:
        """The device the detector runs on (e.g. 'cpu' or 'cuda')."""

    @abstractmethod
    def release(self) -> None:
        """Release native resources, if any."""
