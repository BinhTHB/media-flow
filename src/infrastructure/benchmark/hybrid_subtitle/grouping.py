from __future__ import annotations
import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class TextCandidateGrouper:
    def __init__(self, vertical_overlap_thresh: float = 0.3,
                 horizontal_gap_ratio: float = 2.0):
        self._v_thresh = vertical_overlap_thresh
        self._h_gap_ratio = horizontal_gap_ratio

    def group_frame(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not detections:
            return []
        boxes = sorted(detections, key=lambda d: (d["bbox_xyxy"][1], d["bbox_xyxy"][0]))
        groups: List[List[Dict]] = []
        used: set = set()

        for i, det in enumerate(boxes):
            if i in used:
                continue
            group = [det]
            used.add(i)
            gi0, gi1, gi2, gi3 = det["bbox_xyxy"]
            for j, other in enumerate(boxes[i+1:], i+1):
                if j in used:
                    continue
                oj0, oj1, oj2, oj3 = other["bbox_xyxy"]
                ov = max(0.0, min(gi3, oj3) - max(gi1, oj1))
                avg_h = (gi3 - gi1 + oj3 - oj1) / 2.0 + 1e-6
                gap = max(0, oj0 - gi2)
                if ov / avg_h >= self._v_thresh and gap < self._h_gap_ratio * avg_h:
                    group.append(other)
                    used.add(j)
                    gi2 = max(gi2, oj2)
                    gi3 = max(gi3, oj3)
            groups.append(group)

        results = []
        for gi, group in enumerate(groups):
            all_ids = [d["candidate_id"] for d in group]
            union = [
                min(d["bbox_xyxy"][0] for d in group),
                min(d["bbox_xyxy"][1] for d in group),
                max(d["bbox_xyxy"][2] for d in group),
                max(d["bbox_xyxy"][3] for d in group),
            ]
            results.append({
                "candidate_group_id": f"G{gi+1:03d}",
                "member_candidate_ids": all_ids,
                "union_bbox_xyxy": union,
                "member_count": len(group),
            })
        return results
