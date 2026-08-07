"""
Processing Policy Configuration Domain Entities

Defines action types, per-class policy rules, frontend-tunable parameters,
and pre-built policy presets for the policy-driven media processing engine.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional, Union


class ActionType(str, Enum):
    PRESERVE = "preserve"
    BLUR = "blur"
    SOLID_COVER = "solid_cover"
    REPLACE_TEXT = "replace_text"
    REPLACE_IMAGE = "replace_image"

    @classmethod
    def from_str(cls, value: str) -> "ActionType":
        normalized = value.strip().lower()
        for action in cls:
            if action.value == normalized:
                return action
        valid = [a.value for a in cls]
        raise ValueError(f"Invalid ActionType '{value}'. Valid actions are: {valid}")


@dataclass
class ClassPolicyConfig:
    enabled: bool = True
    action: str = "blur"
    confidence_threshold: float = 0.20
    blur_strength: float = 25.0
    padding_px: int = 4
    solid_color_hex: str = "#000000"
    # Text Replacement Parameters
    text_source: str = "ocr"  # ocr | stt | hybrid
    target_language: str = "vi"
    font: Optional[str] = None
    font_size: Optional[int] = None
    stroke_width: int = 2
    placement: str = "source_bbox"  # source_bbox | auto
    background_action: str = "blur"  # blur | solid_cover | preserve
    # Image Replacement Parameters
    image_asset_path: Optional[str] = None
    scale: float = 1.0
    opacity: float = 1.0
    fit: str = "contain"  # contain | cover | stretch
    z_order: int = 0

    def get_action_enum(self) -> ActionType:
        return ActionType.from_str(self.action)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ClassPolicyConfig":
        if "action" in data and isinstance(data["action"], str):
            ActionType.from_str(data["action"])
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class ProcessingPolicyConfig:
    # Tracker parameters (frontend overridable)
    tracking_iou: float = 0.20
    max_gap_frames: int = 5
    horizontal_gap_ratio: float = 2.0

    # Per-class policies
    subtitle: ClassPolicyConfig = field(
        default_factory=lambda: ClassPolicyConfig(
            action="blur", confidence_threshold=0.15
        )
    )
    watermark: ClassPolicyConfig = field(
        default_factory=lambda: ClassPolicyConfig(
            action="blur", confidence_threshold=0.20
        )
    )
    logo: ClassPolicyConfig = field(
        default_factory=lambda: ClassPolicyConfig(
            action="blur", confidence_threshold=0.20
        )
    )
    other_burn_in_text: ClassPolicyConfig = field(
        default_factory=lambda: ClassPolicyConfig(
            action="blur", confidence_threshold=0.20
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


# Pre-built Presets
PRESET_BLUR_ALL = ProcessingPolicyConfig(
    subtitle=ClassPolicyConfig(action="blur", confidence_threshold=0.15),
    watermark=ClassPolicyConfig(action="blur", confidence_threshold=0.20),
    logo=ClassPolicyConfig(action="blur", confidence_threshold=0.20),
    other_burn_in_text=ClassPolicyConfig(action="blur", confidence_threshold=0.20),
)

PRESET_TRANSLATE_SUBTITLE = ProcessingPolicyConfig(
    subtitle=ClassPolicyConfig(
        action="replace_text", text_source="ocr", target_language="vi", confidence_threshold=0.15
    ),
    watermark=ClassPolicyConfig(action="preserve", confidence_threshold=0.20),
    logo=ClassPolicyConfig(action="preserve", confidence_threshold=0.20),
    other_burn_in_text=ClassPolicyConfig(action="preserve", confidence_threshold=0.20),
)

PRESET_PRESERVE_ALL = ProcessingPolicyConfig(
    subtitle=ClassPolicyConfig(action="preserve"),
    watermark=ClassPolicyConfig(action="preserve"),
    logo=ClassPolicyConfig(action="preserve"),
    other_burn_in_text=ClassPolicyConfig(action="preserve"),
)


def get_preset_policy(preset_name: str) -> ProcessingPolicyConfig:
    normalized = preset_name.strip().lower().replace("_", "-")
    if normalized in ("blur-all", "default"):
        return PRESET_BLUR_ALL
    elif normalized in ("translate-subtitle", "translate"):
        return PRESET_TRANSLATE_SUBTITLE
    elif normalized in ("preserve-all", "preserve"):
        return PRESET_PRESERVE_ALL
    else:
        raise ValueError(
            f"Unknown preset '{preset_name}'. Available presets: blur-all, translate-subtitle, preserve-all"
        )
