from typing import Dict, Any
from src.domain.entities.subtitle_style import SubtitleStyle
from src.shared.exceptions import ValidationError

class SubtitleStylePreset:
    _PRESETS = {
        "default": {
            "font_family": "Arial",
            "font_size": 24,
            "primary_color": "#FFFFFF",
            "outline_color": "#000000",
            "outline_width": 2.0,
            "shadow": 0.0,
            "alignment": 2,
            "margin_v": 15,
            "margin_l": 15,
            "margin_r": 15,
            "bold": False,
            "italic": False,
            "spacing": 0.0,
            "line_height": 1.0,
            "background": "outline",
            "opacity": 1.0
        },
        "youtube": {
            "font_family": "Arial",
            "font_size": 28,
            "primary_color": "#FFFFFF",
            "outline_color": "#000000",
            "outline_width": 3.0,
            "shadow": 1.5,
            "alignment": 2,
            "margin_v": 15,
            "margin_l": 15,
            "margin_r": 15,
            "bold": True,
            "italic": False,
            "spacing": 0.0,
            "line_height": 1.0,
            "background": "outline",
            "opacity": 1.0
        },
        "tiktok": {
            "font_family": "Arial",
            "font_size": 32,
            "primary_color": "#FFFFFF",
            "outline_color": "#000000",
            "outline_width": 4.0,
            "shadow": 2.0,
            "alignment": 2,
            "margin_v": 45,
            "margin_l": 15,
            "margin_r": 15,
            "bold": True,
            "italic": False,
            "spacing": 0.0,
            "line_height": 1.0,
            "background": "outline",
            "opacity": 1.0
        },
        "netflix": {
            "font_family": "Arial",
            "font_size": 28,
            "primary_color": "#FFFF00",
            "outline_color": "#000000",
            "outline_width": 2.0,
            "shadow": 0.0,
            "alignment": 2,
            "margin_v": 25,
            "margin_l": 15,
            "margin_r": 15,
            "bold": False,
            "italic": False,
            "spacing": 0.0,
            "line_height": 1.0,
            "background": "outline",
            "opacity": 1.0
        },
        "minimal": {
            "font_family": "Arial",
            "font_size": 22,
            "primary_color": "#EAEAEA",
            "outline_color": "#000000",
            "outline_width": 1.0,
            "shadow": 0.0,
            "alignment": 2,
            "margin_v": 15,
            "margin_l": 15,
            "margin_r": 15,
            "bold": False,
            "italic": True,
            "spacing": 0.0,
            "line_height": 1.0,
            "background": "outline",
            "opacity": 1.0
        }
    }

    @classmethod
    def get(cls, name: str, overrides: Dict[str, Any] = None) -> SubtitleStyle:
        if name not in cls._PRESETS:
            raise ValidationError(f"Unknown subtitle style preset: {name}")
        
        style_data = dict(cls._PRESETS[name])
        if overrides:
            style_data.update(overrides)
            
        return SubtitleStyle(**style_data)

    @classmethod
    def presets(cls) -> Dict[str, Dict[str, Any]]:
        return cls._PRESETS
