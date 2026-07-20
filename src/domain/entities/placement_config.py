from dataclasses import dataclass

from src.shared.exceptions import ValidationError

@dataclass
class PlacementConfig:
    padding_px: int
    preferred_relation: str
    fallback_alignment: int
    fallback_margin_v: int
    min_region_confidence: float
    video_width: int
    video_height: int
    estimated_height_px: int = 50
    placement_mode: str = "replace_source_subtitle"
    safe_zone_top: float = 0.05
    safe_zone_bottom: float = 0.95
    safe_zone_left: float = 0.05
    safe_zone_right: float = 0.95
    min_source_band_width_ratio: float = 0.80
    vertical_start_ratio: float = 0.72

    def __post_init__(self) -> None:
        if not isinstance(self.padding_px, int) or self.padding_px < 0:
            raise ValidationError("padding_px must be a non-negative integer.")
        if self.preferred_relation not in ("above_source", "below_source", "override"):
            raise ValidationError(
                "preferred_relation must be one of: 'above_source', 'below_source', 'override'."
            )
        if not isinstance(self.fallback_alignment, int) or not 1 <= self.fallback_alignment <= 9:
            raise ValidationError("fallback_alignment must be an integer between 1 and 9.")
        if not isinstance(self.fallback_margin_v, int) or self.fallback_margin_v < 0:
            raise ValidationError("fallback_margin_v must be a non-negative integer.")
        if not isinstance(self.min_region_confidence, (int, float)) or not 0.0 <= self.min_region_confidence <= 1.0:
            raise ValidationError("min_region_confidence must be between 0.0 and 1.0.")
        if not isinstance(self.video_width, int) or self.video_width <= 0:
            raise ValidationError("video_width must be a positive integer.")
        if not isinstance(self.video_height, int) or self.video_height <= 0:
            raise ValidationError("video_height must be a positive integer.")
        if not isinstance(self.estimated_height_px, int) or self.estimated_height_px <= 0:
            raise ValidationError("estimated_height_px must be a positive integer.")
        if self.placement_mode not in (
            "avoid_source_subtitle",
            "replace_source_subtitle",
            "fixed_safe_zone",
        ):
            raise ValidationError(
                "placement_mode must be one of: 'avoid_source_subtitle', "
                "'replace_source_subtitle', 'fixed_safe_zone'."
            )
        safe_zone = (
            self.safe_zone_top,
            self.safe_zone_bottom,
            self.safe_zone_left,
            self.safe_zone_right,
        )
        if (
            not all(isinstance(value, (int, float)) for value in safe_zone)
            or not 0.0 <= self.safe_zone_top < self.safe_zone_bottom <= 1.0
            or not 0.0 <= self.safe_zone_left < self.safe_zone_right <= 1.0
        ):
            raise ValidationError("safe zone bounds must be ordered ratios between 0.0 and 1.0.")
        if not 0.0 < self.min_source_band_width_ratio <= 1.0:
            raise ValidationError("min_source_band_width_ratio must be between 0 and 1.")
        if not 0.0 <= self.vertical_start_ratio < 1.0:
            raise ValidationError("vertical_start_ratio must be between 0 and 1.")

    def to_dict(self) -> dict:
        return {
            "padding_px": self.padding_px,
            "preferred_relation": self.preferred_relation,
            "fallback_alignment": self.fallback_alignment,
            "fallback_margin_v": self.fallback_margin_v,
            "min_region_confidence": self.min_region_confidence,
            "video_width": self.video_width,
            "video_height": self.video_height,
            "estimated_height_px": self.estimated_height_px,
            "placement_mode": self.placement_mode,
            "safe_zone_top": self.safe_zone_top,
            "safe_zone_bottom": self.safe_zone_bottom,
            "safe_zone_left": self.safe_zone_left,
            "safe_zone_right": self.safe_zone_right,
            "min_source_band_width_ratio": self.min_source_band_width_ratio,
            "vertical_start_ratio": self.vertical_start_ratio,
        }