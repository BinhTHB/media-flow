from __future__ import annotations
from typing import Any, Dict, List
from .core import group_frame_candidates


class TextCandidateGrouper:
    """Compatibility wrapper. Nguon su that: core.group_frame_candidates."""

    def __init__(self, vertical_overlap_thresh: float = 0.3,
                 horizontal_gap_ratio: float = 2.0):
        self._h_gap_ratio = horizontal_gap_ratio

    def group_frame(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not detections:
            return []
        return group_frame_candidates(detections, self._h_gap_ratio)
