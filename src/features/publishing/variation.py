"""Deterministic variation policy for re-uploads.

The policy decides *which* ``VariationProfile`` a given video gets. Two rules
drive the design:

1. **Deterministic, not random.** The same (item, attempt) must always produce
   the same profile. A retry that silently re-randomizes would upload a second
   distinct file and defeat idempotency.
2. **Bounded.** Profiles stay inside ``VariationProfile``'s safety bounds so a
   "varied" upload is still a good video, not a distorted one.

The policy is pure: it takes an identity and returns a profile, so it is
trivially testable and has no I/O.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from typing import Optional

from src.core.models.variation import VariationProfile

#: How much of each knob's range to use. Kept small on purpose: a 2-5% crop
#: and a ~1.03x speed change is enough to break an exact fingerprint match
#: without being visible to a viewer.
DEFAULT_LEVEL = "balanced"

_LEVELS = {
    # (speed_span, crop_span, rotate_span, pitch_span, mirror_allowed)
    "off": (0.0, 0.0, 0.0, 0.0, False),
    "light": (0.015, 0.015, 0.5, 0.0, False),
    "balanced": (0.030, 0.030, 1.0, 0.4, True),
    "aggressive": (0.050, 0.050, 2.0, 1.0, True),
}


@dataclass(frozen=True)
class VariationPolicy:
    """Maps an item identity onto a bounded variation profile."""

    level: str = DEFAULT_LEVEL
    #: Force a specific profile instead of deriving one (explicit override).
    fixed_profile: Optional[VariationProfile] = None

    def __post_init__(self) -> None:
        if self.fixed_profile is None and self.level not in _LEVELS:
            allowed = ", ".join(sorted(_LEVELS))
            raise ValueError(f"Unknown variation level {self.level!r}; expected one of: {allowed}")

    def profile_for(self, item_id: str, attempt: int = 1) -> VariationProfile:
        """Return the profile for one publish attempt of one item.

        ``attempt`` participates in the seed so a genuine re-publish (e.g. the
        operator wants a second variant) can differ, while a plain retry of the
        same attempt stays byte-identical.
        """
        if self.fixed_profile is not None:
            return self.fixed_profile

        speed_span, crop_span, rotate_span, pitch_span, mirror_allowed = _LEVELS[self.level]
        if speed_span == 0.0 and crop_span == 0.0:
            return VariationProfile(label=self.level)

        seed = _seed(item_id, attempt)
        # Each knob draws from its own slice of the digest so the values are
        # independent rather than all moving together.
        speed = 1.0 + _signed_unit(seed, 0) * speed_span
        # Crop has a floor: a 0.01% crop would be a no-op, so every non-off
        # level guarantees at least half its span is actually removed.
        crop = (crop_span / 2.0) + abs(_signed_unit(seed, 1)) * (crop_span / 2.0)
        rotate = _signed_unit(seed, 2) * rotate_span
        pitch = _signed_unit(seed, 3) * pitch_span
        mirror = mirror_allowed and _bit(seed, 4)

        return VariationProfile(
            speed=round(speed, 4),
            crop_ratio=round(crop, 4),
            rotate_degrees=round(rotate, 2),
            pitch_semitones=round(pitch, 3),
            mirror=mirror,
            label=self.level,
        )


def post_render_profile(
    profile: VariationProfile,
    *,
    mirrored_during_render: bool,
    drawn_captions: bool,
) -> VariationProfile:
    """FFmpeg pass after the visual render.

    Mirror is already in the pixels when the visual pass flipped each frame
    before drawing text. This pass still applies crop, speed, rotate, and pitch.
    """
    if mirrored_during_render:
        return replace(profile, mirror=False)
    return keep_overlay_text_readable(profile, drawn_captions=drawn_captions)


def keep_overlay_text_readable(profile: VariationProfile, *, drawn_captions: bool) -> VariationProfile:
    """Post-render hflip mirrors drawn captions and makes them unreadable.

    Crop/speed/rotate stay. Mirror is only skipped when replacement text was
    painted onto the frames.
    """
    if drawn_captions and profile.mirror:
        return replace(profile, mirror=False)
    return profile


def _seed(item_id: str, attempt: int) -> bytes:
    return hashlib.sha256(f"{item_id}|{int(attempt)}".encode("utf-8")).digest()


def _signed_unit(seed: bytes, slot: int) -> float:
    """Deterministic float in [-1, 1] for one slot of the digest."""
    raw = seed[slot * 2] << 8 | seed[slot * 2 + 1]
    return (raw / 65535.0) * 2.0 - 1.0


def _bit(seed: bytes, slot: int) -> bool:
    return bool(seed[slot * 2] & 0x01)
