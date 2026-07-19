from __future__ import annotations
import logging
from typing import Any, Dict, List
import numpy as np
import statistics

logger = logging.getLogger(__name__)


class SafeCoverRegionBuilder:
    def __init__(self, glyph_margin: float = 12.0, outline_margin: float = 4.0,
                 shadow_margin: float = 2.0, motion_margin: float = 5.0):
        self._margin = glyph_margin + outline_margin + shadow_margin + motion_margin

    def build_safe_covers(self, bboxes: List[List[float]],
                          strategy: str = "percentile") -> List[float]:
        if not bboxes:
            return [0.0, 0.0, 0.0, 0.0]
        x1s = [b[0] for b in bboxes]
        y1s = [b[1] for b in bboxes]
        x2s = [b[2] for b in bboxes]
        y2s = [b[3] for b in bboxes]

        if strategy == "per_frame":
            exp_x1s = [x - self._margin for x in x1s]
            exp_y1s = [y - self._margin for y in y1s]
            exp_x2s = [x + self._margin for x in x2s]
            exp_y2s = [y + self._margin for y in y2s]
            return [statistics.median(exp_x1s), statistics.median(exp_y1s),
                    statistics.median(exp_x2s), statistics.median(exp_y2s)]
        elif strategy == "median":
            return [
                statistics.median(x1s) - self._margin,
                statistics.median(y1s) - self._margin,
                statistics.median(x2s) + self._margin,
                statistics.median(y2s) + self._margin,
            ]
        else:
            return [
                float(np.percentile(x1s, 5)) - self._margin,
                float(np.percentile(y1s, 5)) - self._margin,
                float(np.percentile(x2s, 95)) + self._margin,
                float(np.percentile(y2s, 95)) + self._margin,
            ]

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
