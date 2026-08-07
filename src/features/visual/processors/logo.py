"""
Logo Processor

Manages logo-specific processing. Default action: BlurAction.
Uses shared visual actions without duplicating pixel algorithms.
"""

from __future__ import annotations

from typing import Any, List, Optional

from src.core.models.track import Track
from src.features.visual.actions.blur import BlurAction
from src.features.visual.actions.preserve import PreserveAction
from src.features.visual.actions.replace_image import ReplaceImageAction
from src.features.visual.actions.solid_cover import SolidCoverAction


class LogoProcessor:
    def __init__(
        self,
        action_name: str = "blur",
        blur_action: Optional[BlurAction] = None,
        replace_image_action: Optional[ReplaceImageAction] = None,
        solid_cover_action: Optional[SolidCoverAction] = None,
    ):
        self.action_name = action_name
        self.blur_action = blur_action or BlurAction()
        self.preserve_action = PreserveAction()
        self.replace_image_action = replace_image_action
        self.solid_cover_action = solid_cover_action or SolidCoverAction()

    def process_frame_tracks(self, frame: Any, frame_index: int, tracks: List[Track]) -> Any:
        if self.action_name == "preserve":
            return frame

        for track in tracks:
            det = next((d for d in track.detections if d.frame_index == frame_index), None)
            if not det:
                continue

            if self.action_name == "blur":
                frame = self.blur_action.execute(frame, det.bbox)
            elif self.action_name == "solid_cover":
                frame = self.solid_cover_action.execute(frame, det.bbox)
            elif self.action_name == "replace_image" and self.replace_image_action:
                frame = self.replace_image_action.execute(frame, det.bbox)

        return frame
