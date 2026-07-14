from dataclasses import dataclass
from src.shared.exceptions import ValidationError

@dataclass
class FeatureConfig:
    speech_to_text: bool = True
    ocr: bool = False
    translate: bool = True
    subtitle: bool = True
    render_video: bool = False
    subtitle_placement: bool = False
    subtitle_placement_debug: bool = False
    dubbing: bool = False

    def __post_init__(self):
        for field_name in (
            "speech_to_text",
            "ocr",
            "translate",
            "subtitle",
            "render_video",
            "subtitle_placement",
            "subtitle_placement_debug",
            "dubbing",
        ):
            if not isinstance(getattr(self, field_name), bool):
                raise ValidationError(f"FeatureConfig.{field_name} must be a boolean.")

    def to_dict(self) -> dict:
        return {
            "speech_to_text": self.speech_to_text,
            "ocr": self.ocr,
            "translate": self.translate,
            "subtitle": self.subtitle,
            "render_video": self.render_video,
            "subtitle_placement": self.subtitle_placement,
            "subtitle_placement_debug": self.subtitle_placement_debug,
            "dubbing": self.dubbing,
        }
