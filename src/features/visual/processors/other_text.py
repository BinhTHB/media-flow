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
from src.features.visual.actions.preserve import PreserveAction
from src.features.visual.actions.solid_cover import SolidCoverAction


def _get_suppression_bbox(bbox: BBox) -> BBox:
    x1, y1, x2, y2 = bbox.xyxy_int()
    fw = bbox.frame_width
    fh = bbox.frame_height

    if x1 <= 50:
        x1 = 0
        x2 = max(x2 + 30, 160)
        y1 = max(0, y1 - 20)
        y2 = min(fh, y2 + 20)
    elif x2 >= fw - 50:
        x1 = min(x1 - 30, max(0, fw - 160))
        x2 = fw
        y1 = max(0, y1 - 20)
        y2 = min(fh, y2 + 20)
    else:
        x1 = max(0, x1 - 20)
        y1 = max(0, y1 - 20)
        x2 = min(fw, x2 + 20)
        y2 = min(fh, y2 + 20)

    return BBox(x1=x1, y1=y1, x2=x2, y2=y2, frame_width=fw, frame_height=fh)


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

            det = next((d for d in track.detections if d.frame_index == frame_index), None)
            if not det:
                first_f = track.detections[0].frame_index
                last_f = track.detections[-1].frame_index
                if first_f <= frame_index <= last_f:
                    det = min(track.detections, key=lambda d: abs(d.frame_index - frame_index))

            if not det:
                continue

            supp_bbox = _get_suppression_bbox(det.bbox)

            if self.action_name == "blur":
                frame = self.blur_action.execute(frame, supp_bbox, padding_px=0)
            elif self.action_name == "solid_cover":
                frame = self.solid_cover_action.execute(frame, supp_bbox, padding_px=0)

        return frame
