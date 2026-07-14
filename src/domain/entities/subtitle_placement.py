from dataclasses import dataclass
from typing import Dict, Optional

from src.shared.exceptions import ValidationError


@dataclass
class SubtitlePlacement:
    alignment: int
    x: Optional[float] = None
    y: Optional[float] = None
    margin_v: Optional[int] = None
    margin_l: Optional[int] = None
    margin_r: Optional[int] = None
    rendered_bbox: Optional[Dict[str, float]] = None
    mask_bbox: Optional[Dict[str, float]] = None
    is_fallback: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.alignment, int) or not 1 <= self.alignment <= 9:
            raise ValidationError("alignment must be an integer between 1 and 9.")

    def to_dict(self) -> dict:
        result = {
            "alignment": self.alignment,
            "x": self.x,
            "y": self.y,
            "margin_v": self.margin_v,
            "margin_l": self.margin_l,
            "margin_r": self.margin_r,
            "is_fallback": self.is_fallback,
        }
        if self.rendered_bbox is not None:
            result["rendered_bbox"] = self.rendered_bbox
        if self.mask_bbox is not None:
            result["mask_bbox"] = self.mask_bbox
        return result
