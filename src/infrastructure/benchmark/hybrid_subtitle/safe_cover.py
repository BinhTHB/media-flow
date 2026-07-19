from __future__ import annotations
import logging
from typing import Any, Dict, List
import statistics
from .core import median_box, percentile_box

logger = logging.getLogger(__name__)


class SafeCoverRegionBuilder:
    def __init__(self, glyph_margin: float = 12.0, outline_margin: float = 4.0,
                 shadow_margin: float = 2.0, motion_margin: float = 5.0):
        self._margin = glyph_margin + outline_margin + shadow_margin + motion_margin

    def build_safe_covers(self, bboxes: List[List[float]],
                          strategy: str = "percentile") -> List[float]:
        if not bboxes:
            return [0.0, 0.0, 0.0, 0.0]
        m = self._margin
        if strategy == "per_frame":
            expanded = [[b[0] - m, b[1] - m, b[2] + m, b[3] + m] for b in bboxes]
            return list(median_box(expanded))
        if strategy == "median":
            x1, y1, x2, y2 = median_box(bboxes)
            return [x1 - m, y1 - m, x2 + m, y2 + m]
        x1, y1, x2, y2 = percentile_box(bboxes, 5, 95)
        return [x1 - m, y1 - m, x2 + m, y2 + m]

    def check_review_flag(self, bboxes: List[List[float]],
                          verdict_color: str = "green",
                          person_overlap: bool = False) -> bool:
        if not bboxes:
            return False
        if len(bboxes) < 2:
            return False
        x1_var = statistics.variance([b[0] for b in bboxes])
        y1_var = statistics.variance([b[1] for b in bboxes])
        if x1_var > 1500.0 or y1_var > 1500.0:
            return True
        if person_overlap or verdict_color == "orange":
            return True
        return False
