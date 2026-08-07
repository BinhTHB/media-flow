"""
Processing Policy Configuration Domain Entities

Defines composable multi-action policy structures:
- background: Single background action (preserve | blur | solid_cover)
- overlays: List of overlay actions (replace_text | replace_image)
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


class BackgroundActionType(str, Enum):
    PRESERVE = "preserve"
    BLUR = "blur"
    SOLID_COVER = "solid_cover"

    @classmethod
    def from_str(cls, value: str) -> "BackgroundActionType":
        normalized = value.strip().lower()
        for action in cls:
            if action.value == normalized:
                return action
        valid = [a.value for a in cls]
        raise ValueError(f"Invalid BackgroundActionType '{value}'. Valid actions are: {valid}")


class OverlayActionType(str, Enum):
    REPLACE_TEXT = "replace_text"
    REPLACE_IMAGE = "replace_image"

    @classmethod
    def from_str(cls, value: str) -> "OverlayActionType":
        normalized = value.strip().lower()
        for action in cls:
            if action.value == normalized:
                return action
        valid = [a.value for a in cls]
        raise ValueError(f"Invalid OverlayActionType '{value}'. Valid actions are: {valid}")


@dataclass
class BackgroundActionConfig:
    action: str = "blur"
    blur_strength: float = 25.0
    padding_px: int = 4
    solid_color_hex: str = "#000000"

    def __post_init__(self):
        BackgroundActionType.from_str(self.action)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BackgroundActionConfig":
        if not data:
            return cls()
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class OverlayActionConfig:
    action: str = "replace_text"
    # Text Overlay Parameters
    text_source: str = "ocr"  # ocr | stt | hybrid
    target_language: str = "vi"
    font: Optional[str] = None
    font_size: Optional[int] = None
    stroke_width: int = 2
    placement: str = "source_bbox"  # source_bbox | auto
    # Image Overlay Parameters
    image_asset_path: Optional[str] = None
    scale: float = 1.0
    opacity: float = 1.0
    fit: str = "contain"  # contain | cover | stretch
    padding_px: int = 4
    z_order: int = 0

    def __post_init__(self):
        OverlayActionType.from_str(self.action)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OverlayActionConfig":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class ClassPolicyConfig:
    enabled: bool = True
    confidence_threshold: float = 0.20
    background: BackgroundActionConfig = field(default_factory=BackgroundActionConfig)
    overlays: List[OverlayActionConfig] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "confidence_threshold": self.confidence_threshold,
            "background": self.background.to_dict(),
            "overlays": [o.to_dict() for o in self.overlays],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ClassPolicyConfig":
        if not data:
            return cls()
        enabled = data.get("enabled", True)
        confidence_threshold = float(data.get("confidence_threshold", 0.20))

        # Legacy backward compatibility mapping if action is specified at top-level
        legacy_action = data.get("action")
        bg_data = data.get("background", {})
        overlays_data = data.get("overlays", [])

        if legacy_action and not bg_data and not overlays_data:
            if legacy_action in ("blur", "preserve", "solid_cover"):
                bg_data = {"action": legacy_action, "blur_strength": data.get("blur_strength", 25.0), "padding_px": data.get("padding_px", 4)}
            elif legacy_action == "replace_text":
                bg_data = {"action": data.get("background_action", "blur")}
                overlays_data = [{
                    "action": "replace_text",
                    "text_source": data.get("text_source", "ocr"),
                    "target_language": data.get("target_language", "vi"),
                    "font": data.get("font"),
                    "font_size": data.get("font_size"),
                }]
            elif legacy_action == "replace_image":
                bg_data = {"action": data.get("background_action", "blur")}
                overlays_data = [{
                    "action": "replace_image",
                    "image_asset_path": data.get("image_asset_path"),
                    "fit": data.get("fit", "contain"),
                    "opacity": data.get("opacity", 1.0),
                }]

        bg_config = BackgroundActionConfig.from_dict(bg_data)
        overlay_configs = [OverlayActionConfig.from_dict(o) for o in overlays_data]

        return cls(
            enabled=enabled,
            confidence_threshold=confidence_threshold,
            background=bg_config,
            overlays=overlay_configs,
        )


@dataclass
class ProcessingPolicyConfig:
    tracking_iou: float = 0.20
    max_gap_frames: int = 5
    horizontal_gap_ratio: float = 2.0

    subtitle: ClassPolicyConfig = field(
        default_factory=lambda: ClassPolicyConfig(
            enabled=True,
            confidence_threshold=0.15,
            background=BackgroundActionConfig(action="blur"),
            overlays=[],
        )
    )
    watermark: ClassPolicyConfig = field(
        default_factory=lambda: ClassPolicyConfig(
            enabled=True,
            confidence_threshold=0.20,
            background=BackgroundActionConfig(action="blur"),
            overlays=[],
        )
    )
    logo: ClassPolicyConfig = field(
        default_factory=lambda: ClassPolicyConfig(
            enabled=True,
            confidence_threshold=0.20,
            background=BackgroundActionConfig(action="blur"),
            overlays=[],
        )
    )
    other_burn_in_text: ClassPolicyConfig = field(
        default_factory=lambda: ClassPolicyConfig(
            enabled=True,
            confidence_threshold=0.20,
            background=BackgroundActionConfig(action="preserve"),
            overlays=[],
        )
    )

    def get_class_policy(self, class_name: str) -> ClassPolicyConfig:
        normalized = class_name.strip().lower()
        if hasattr(self, normalized):
            return getattr(self, normalized)
        return self.other_burn_in_text

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tracking_iou": self.tracking_iou,
            "max_gap_frames": self.max_gap_frames,
            "horizontal_gap_ratio": self.horizontal_gap_ratio,
            "subtitle": self.subtitle.to_dict(),
            "watermark": self.watermark.to_dict(),
            "logo": self.logo.to_dict(),
            "other_burn_in_text": self.other_burn_in_text.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProcessingPolicyConfig":
        kwargs: Dict[str, Any] = {}
        if "tracking_iou" in data:
            kwargs["tracking_iou"] = float(data["tracking_iou"])
        if "max_gap_frames" in data:
            kwargs["max_gap_frames"] = int(data["max_gap_frames"])
        if "horizontal_gap_ratio" in data:
            kwargs["horizontal_gap_ratio"] = float(data["horizontal_gap_ratio"])
        kwargs["subtitle"] = ClassPolicyConfig.from_dict(data.get("subtitle", {}))
        kwargs["watermark"] = ClassPolicyConfig.from_dict(data.get("watermark", {}))
        kwargs["logo"] = ClassPolicyConfig.from_dict(data.get("logo", {}))
        kwargs["other_burn_in_text"] = ClassPolicyConfig.from_dict(
            data.get("other_burn_in_text", {})
        )
        return cls(**kwargs)

    @classmethod
    def from_yaml_file(cls, yaml_path: Union[str, Path]) -> "ProcessingPolicyConfig":
        import yaml

        path = Path(yaml_path)
        if not path.exists():
            raise FileNotFoundError(f"Policy YAML file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        policy_data = raw.get("classes", raw)
        return cls.from_dict(policy_data)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


# Presets
PRESET_DEFAULT = ProcessingPolicyConfig(
    subtitle=ClassPolicyConfig(enabled=True, confidence_threshold=0.15, background=BackgroundActionConfig(action="blur"), overlays=[]),
    watermark=ClassPolicyConfig(enabled=True, confidence_threshold=0.20, background=BackgroundActionConfig(action="blur"), overlays=[]),
    logo=ClassPolicyConfig(enabled=True, confidence_threshold=0.20, background=BackgroundActionConfig(action="blur"), overlays=[]),
    other_burn_in_text=ClassPolicyConfig(enabled=True, confidence_threshold=0.20, background=BackgroundActionConfig(action="preserve"), overlays=[]),
)

PRESET_TRANSLATE_SUBTITLE = ProcessingPolicyConfig(
    subtitle=ClassPolicyConfig(
        enabled=True,
        confidence_threshold=0.15,
        background=BackgroundActionConfig(action="blur"),
        overlays=[OverlayActionConfig(action="replace_text", text_source="ocr", target_language="vi")],
    ),
    watermark=ClassPolicyConfig(enabled=True, confidence_threshold=0.20, background=BackgroundActionConfig(action="preserve"), overlays=[]),
    logo=ClassPolicyConfig(enabled=True, confidence_threshold=0.20, background=BackgroundActionConfig(action="preserve"), overlays=[]),
    other_burn_in_text=ClassPolicyConfig(enabled=True, confidence_threshold=0.20, background=BackgroundActionConfig(action="preserve"), overlays=[]),
)

PRESET_BLUR_ALL = ProcessingPolicyConfig(
    subtitle=ClassPolicyConfig(enabled=True, confidence_threshold=0.15, background=BackgroundActionConfig(action="blur"), overlays=[]),
    watermark=ClassPolicyConfig(enabled=True, confidence_threshold=0.20, background=BackgroundActionConfig(action="blur"), overlays=[]),
    logo=ClassPolicyConfig(enabled=True, confidence_threshold=0.20, background=BackgroundActionConfig(action="blur"), overlays=[]),
    other_burn_in_text=ClassPolicyConfig(enabled=True, confidence_threshold=0.20, background=BackgroundActionConfig(action="blur"), overlays=[]),
)

PRESET_PRESERVE_ALL = ProcessingPolicyConfig(
    subtitle=ClassPolicyConfig(enabled=True, background=BackgroundActionConfig(action="preserve"), overlays=[]),
    watermark=ClassPolicyConfig(enabled=True, background=BackgroundActionConfig(action="preserve"), overlays=[]),
    logo=ClassPolicyConfig(enabled=True, background=BackgroundActionConfig(action="preserve"), overlays=[]),
    other_burn_in_text=ClassPolicyConfig(enabled=True, background=BackgroundActionConfig(action="preserve"), overlays=[]),
)


def get_preset_policy(preset_name: str) -> ProcessingPolicyConfig:
    normalized = preset_name.strip().lower().replace("_", "-")
    if normalized in ("default", "blur-default"):
        return PRESET_DEFAULT
    elif normalized in ("blur-all",):
        return PRESET_BLUR_ALL
    elif normalized in ("translate-subtitle", "translate"):
        return PRESET_TRANSLATE_SUBTITLE
    elif normalized in ("preserve-all", "preserve"):
        return PRESET_PRESERVE_ALL
    else:
        raise ValueError(
            f"Unknown preset '{preset_name}'. Available presets: default, blur-all, translate-subtitle, preserve-all"
        )
