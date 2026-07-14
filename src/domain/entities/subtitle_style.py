from dataclasses import dataclass
from src.shared.exceptions import ValidationError

@dataclass
class SubtitleStyle:
    font_family: str = "Arial"
    font_size: int = 24
    primary_color: str = "#FFFFFF"
    outline_color: str = "#000000"
    outline_width: float = 2.0
    shadow: float = 0.0
    alignment: int = 2
    margin_v: int = 15
    margin_l: int = 15
    margin_r: int = 15
    bold: bool = False
    italic: bool = False
    spacing: float = 0.0
    line_height: float = 1.0
    background: str = "outline"
    opacity: float = 1.0

    def __post_init__(self):
        if not isinstance(self.font_family, str) or not self.font_family:
            raise ValidationError("font_family must be a non-empty string.")
        if not isinstance(self.font_size, int) or self.font_size <= 0:
            raise ValidationError("font_size must be a positive integer.")
        if not isinstance(self.primary_color, str):
            raise ValidationError("primary_color must be a string.")
        if not isinstance(self.outline_color, str):
            raise ValidationError("outline_color must be a string.")
        if not isinstance(self.outline_width, (int, float)) or self.outline_width < 0:
            raise ValidationError("outline_width must be a non-negative number.")
        if not isinstance(self.shadow, (int, float)) or self.shadow < 0:
            raise ValidationError("shadow must be a non-negative number.")
        if not isinstance(self.alignment, int) or not (1 <= self.alignment <= 9):
            raise ValidationError("alignment must be an integer between 1 and 9.")
        if not isinstance(self.margin_v, int) or self.margin_v < 0:
            raise ValidationError("margin_v must be a non-negative integer.")
        if not isinstance(self.margin_l, int) or self.margin_l < 0:
            raise ValidationError("margin_l must be a non-negative integer.")
        if not isinstance(self.margin_r, int) or self.margin_r < 0:
            raise ValidationError("margin_r must be a non-negative integer.")
        if not isinstance(self.bold, bool):
            raise ValidationError("bold must be a boolean.")
        if not isinstance(self.italic, bool):
            raise ValidationError("italic must be a boolean.")
        if not isinstance(self.spacing, (int, float)):
            raise ValidationError("spacing must be a number.")
        if not isinstance(self.line_height, (int, float)) or self.line_height <= 0:
            raise ValidationError("line_height must be a positive number.")
        if self.background not in ("outline", "box", "none"):
            raise ValidationError("background must be one of: 'outline', 'box', 'none'.")
        if not isinstance(self.opacity, (int, float)) or not (0.0 <= self.opacity <= 1.0):
            raise ValidationError("opacity must be a number between 0.0 and 1.0.")
