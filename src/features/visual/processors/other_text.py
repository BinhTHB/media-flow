"""
Other Text Processor

Manages other_burn_in_text-specific processing. Default action: BlurAction (or PreserveAction depending on policy).
Uses shared visual actions without duplicating pixel algorithms.
"""

from __future__ import annotations

from typing import Any, List, Optional

from src.core.models.bbox import BBox
from src.core.models.track import Track
from src.features.visual.actions.blur import BlurAction
from src.features.visual.mark_span import bbox_at_frame
from src.features.visual.actions.preserve import PreserveAction
from src.features.visual.actions.solid_cover import SolidCoverAction


def _get_suppression_bbox(bbox: BBox) -> BBox:
    x1, y1, x2, y2 = bbox.xyxy_int()
    fw = bbox.frame_width
    fh = bbox.frame_height

    pad_x = max(16, int((x2 - x1) * 0.08))
    pad_y = max(12, int((y2 - y1) * 0.15))

    new_x1 = 0 if x1 <= 40 else max(0, x1 - pad_x)
    new_x2 = fw if x2 >= fw - 40 else min(fw, x2 + pad_x)
    new_y1 = 0 if y1 <= 30 else max(0, y1 - pad_y)
    new_y2 = fh if y2 >= fh - 30 else min(fh, y2 + pad_y)

    return BBox(x1=new_x1, y1=new_y1, x2=new_x2, y2=new_y2, frame_width=fw, frame_height=fh)


class OtherTextProcessor:
    def __init__(
        self,
        action_name: str = "preserve",
        blur_action: Optional[BlurAction] = None,
        solid_cover_action: Optional[SolidCoverAction] = None,
        padding_px: int = 16,
    ):
        self.action_name = action_name
        self.blur_action = blur_action or BlurAction(strength=45.0, padding_px=0)
        self.preserve_action = PreserveAction()
        self.solid_cover_action = solid_cover_action or SolidCoverAction(padding_px=0)
        self.padding_px = padding_px

    def process_frame_tracks(self, frame: Any, frame_index: int, tracks: List[Track]) -> Any:
        if self.action_name == "preserve":
            return frame

        for track in tracks:
            if not track.detections:
                continue

            box = bbox_at_frame(track.detections, frame_index)
            if box is None:
                continue

            supp_bbox = _get_suppression_bbox(box)

            if self.action_name == "blur":
                frame = self.blur_action.execute(frame, supp_bbox, padding_px=0)
            elif self.action_name == "solid_cover":
                frame = self.solid_cover_action.execute(frame, supp_bbox, padding_px=0)

        return frame
