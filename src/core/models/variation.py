"""Variation profile for re-uploaded videos.

A re-upload of a finished video is not a byte-identical copy: platforms
fingerprint media, and an identical file is trivially matched to its source.
A ``VariationProfile`` describes *how* one publish attempt differs from the
master file — it is a declarative, serializable contract, not a filter chain.

The profile is deliberately conservative: every knob defaults to "leave it
alone" so an empty profile is a byte-for-byte copy. Callers opt into specific
transformations, which keeps the behaviour auditable and testable.

Units are explicit:
- ``speed`` is a multiplier (>1 faster, <1 slower);
- ``crop_ratio`` is the fraction removed from each edge (0.02 = 2%);
- ``rotate_degrees`` is clockwise;
- ``pitch_semitones`` shifts audio pitch independently of speed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

#: Hard bounds. Outside these, output stops looking like the source and the
#: upload is more likely to be rejected for quality than accepted as new.
MIN_SPEED, MAX_SPEED = 0.90, 1.10
MAX_CROP_RATIO = 0.10
MAX_ROTATE_DEGREES = 3.0
MAX_PITCH_SEMITONES = 2.0


@dataclass(frozen=True)
class VariationProfile:
    """One deterministic set of transformations applied to a master video."""

    speed: float = 1.0
    mirror: bool = False
    crop_ratio: float = 0.0
    rotate_degrees: float = 0.0
    brightness: float = 0.0
    contrast: float = 1.0
    saturation: float = 1.0
    pitch_semitones: float = 0.0
    trim_start_seconds: float = 0.0
    trim_end_seconds: float = 0.0
    #: Free-form label recorded in the publication state for auditing.
    label: str = ""

    def __post_init__(self) -> None:
        if not (MIN_SPEED <= float(self.speed) <= MAX_SPEED):
            raise ValueError(
                f"VariationProfile.speed must be within [{MIN_SPEED}, {MAX_SPEED}], got {self.speed}"
            )
        if not (0.0 <= float(self.crop_ratio) <= MAX_CROP_RATIO):
            raise ValueError(
                f"VariationProfile.crop_ratio must be within [0, {MAX_CROP_RATIO}], got {self.crop_ratio}"
            )
        if abs(float(self.rotate_degrees)) > MAX_ROTATE_DEGREES:
            raise ValueError(
                f"VariationProfile.rotate_degrees must be within ±{MAX_ROTATE_DEGREES}, "
                f"got {self.rotate_degrees}"
            )
        if abs(float(self.pitch_semitones)) > MAX_PITCH_SEMITONES:
            raise ValueError(
                f"VariationProfile.pitch_semitones must be within ±{MAX_PITCH_SEMITONES}, "
                f"got {self.pitch_semitones}"
            )
        if float(self.trim_start_seconds) < 0.0 or float(self.trim_end_seconds) < 0.0:
            raise ValueError("VariationProfile trim values must not be negative")

    @property
    def is_identity(self) -> bool:
        """True when applying this profile changes nothing."""
        return (
            self.speed == 1.0
            and not self.mirror
            and self.crop_ratio == 0.0
            and self.rotate_degrees == 0.0
            and self.brightness == 0.0
            and self.contrast == 1.0
            and self.saturation == 1.0
            and self.pitch_semitones == 0.0
            and self.trim_start_seconds == 0.0
            and self.trim_end_seconds == 0.0
        )

    def to_dict(self) -> dict:
        return {
            "speed": round(float(self.speed), 4),
            "mirror": bool(self.mirror),
            "crop_ratio": round(float(self.crop_ratio), 4),
            "rotate_degrees": round(float(self.rotate_degrees), 3),
            "brightness": round(float(self.brightness), 4),
            "contrast": round(float(self.contrast), 4),
            "saturation": round(float(self.saturation), 4),
            "pitch_semitones": round(float(self.pitch_semitones), 3),
            "trim_start_seconds": round(float(self.trim_start_seconds), 3),
            "trim_end_seconds": round(float(self.trim_end_seconds), 3),
            "label": self.label,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "VariationProfile":
        known = {
            "speed", "mirror", "crop_ratio", "rotate_degrees", "brightness",
            "contrast", "saturation", "pitch_semitones", "trim_start_seconds",
            "trim_end_seconds", "label",
        }
        return cls(**{k: v for k, v in data.items() if k in known})
