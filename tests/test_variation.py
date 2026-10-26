"""Tests for the re-upload variation layer.

Two properties matter most and are asserted directly:

1. **Determinism** — the same (item, attempt) always yields the same profile,
   so a retry uploads identical bytes instead of a surprise second variant.
2. **Bounded output** — profiles stay inside the safety limits, and the ffmpeg
   filter chain never produces odd dimensions (libx264 rejects them).
"""

from __future__ import annotations

import pytest

from src.core.models.variation import (
    MAX_CROP_RATIO,
    MAX_PITCH_SEMITONES,
    MAX_ROTATE_DEGREES,
    MAX_SPEED,
    MIN_SPEED,
    VariationProfile,
)
from src.features.publishing.variation import VariationPolicy
from src.infrastructure.video.variation import FFmpegVariationAdapter


# --------------------------------------------------------------------------
# VariationProfile
# --------------------------------------------------------------------------

def test_identity_profile_is_the_default() -> None:
    assert VariationProfile().is_identity


def test_profile_rejects_out_of_range_values() -> None:
    with pytest.raises(ValueError, match="speed"):
        VariationProfile(speed=MAX_SPEED + 0.01)
    with pytest.raises(ValueError, match="speed"):
        VariationProfile(speed=MIN_SPEED - 0.01)
    with pytest.raises(ValueError, match="crop_ratio"):
        VariationProfile(crop_ratio=MAX_CROP_RATIO + 0.01)
    with pytest.raises(ValueError, match="rotate_degrees"):
        VariationProfile(rotate_degrees=MAX_ROTATE_DEGREES + 0.1)
    with pytest.raises(ValueError, match="pitch_semitones"):
        VariationProfile(pitch_semitones=MAX_PITCH_SEMITONES + 0.1)
    with pytest.raises(ValueError, match="trim"):
        VariationProfile(trim_start_seconds=-1.0)


def test_profile_roundtrips_through_dict() -> None:
    profile = VariationProfile(speed=1.03, mirror=True, crop_ratio=0.02, label="balanced")
    assert VariationProfile.from_dict(profile.to_dict()) == profile


def test_profile_from_dict_ignores_unknown_keys() -> None:
    profile = VariationProfile.from_dict({"speed": 1.02, "not_a_field": 1})
    assert profile.speed == 1.02


# --------------------------------------------------------------------------
# VariationPolicy
# --------------------------------------------------------------------------

def test_policy_is_deterministic_per_item_and_attempt() -> None:
    policy = VariationPolicy(level="balanced")
    first = policy.profile_for("item-abc", 1)
    second = policy.profile_for("item-abc", 1)
    assert first == second


def test_policy_varies_between_attempts_and_items() -> None:
    policy = VariationPolicy(level="balanced")
    assert policy.profile_for("item-abc", 1) != policy.profile_for("item-abc", 2)
    assert policy.profile_for("item-abc", 1) != policy.profile_for("item-xyz", 1)


def test_off_level_is_identity() -> None:
    assert VariationPolicy(level="off").profile_for("anything", 1).is_identity


@pytest.mark.parametrize("level", ["light", "balanced", "aggressive"])
def test_non_off_levels_actually_change_something(level: str) -> None:
    profile = VariationPolicy(level=level).profile_for("item", 1)
    assert not profile.is_identity
    # Crop has a floor so a "varied" upload is never a near no-op.
    assert profile.crop_ratio > 0


@pytest.mark.parametrize("level", ["light", "balanced", "aggressive"])
def test_policy_stays_inside_profile_bounds(level: str) -> None:
    policy = VariationPolicy(level=level)
    for index in range(50):
        profile = policy.profile_for(f"item-{index}", 1)
        assert MIN_SPEED <= profile.speed <= MAX_SPEED
        assert 0.0 <= profile.crop_ratio <= MAX_CROP_RATIO
        assert abs(profile.rotate_degrees) <= MAX_ROTATE_DEGREES
        assert abs(profile.pitch_semitones) <= MAX_PITCH_SEMITONES


def test_fixed_profile_overrides_the_level() -> None:
    fixed = VariationProfile(speed=1.05, label="manual")
    assert VariationPolicy(level="off", fixed_profile=fixed).profile_for("x", 1) is fixed


def test_unknown_level_is_rejected() -> None:
    with pytest.raises(ValueError, match="variation level"):
        VariationPolicy(level="nonsense")


# --------------------------------------------------------------------------
# FFmpeg filter construction (no encoding)
# --------------------------------------------------------------------------

def test_identity_profile_builds_no_filters() -> None:
    assert FFmpegVariationAdapter.build_filters(VariationProfile()) == ("", "")


def test_speed_produces_inverse_setpts_and_matching_atempo() -> None:
    vf, af = FFmpegVariationAdapter.build_filters(VariationProfile(speed=1.05))
    assert "setpts=0.952381*PTS" in vf
    assert "atempo=1.0500" in af


def test_crop_restores_exact_source_dimensions() -> None:
    vf, _ = FFmpegVariationAdapter.build_filters(
        VariationProfile(crop_ratio=0.03), frame_width=1920, frame_height=1080
    )
    # Restore targets the exact (even) source size, not a ratio-derived guess.
    assert "scale=1920:1080" in vf
    assert "crop=trunc(iw*0.9400/2)*2:trunc(ih*0.9400/2)*2" in vf


def test_crop_rounds_to_even_without_known_dimensions() -> None:
    vf, _ = FFmpegVariationAdapter.build_filters(VariationProfile(crop_ratio=0.03))
    # Every dimension expression must include an even-rounding trunc()/2)*2.
    for segment in vf.split(","):
        if segment.startswith(("crop=", "scale=")):
            assert "/2)*2" in segment


def test_rotate_crops_back_to_source_size() -> None:
    vf, _ = FFmpegVariationAdapter.build_filters(
        VariationProfile(rotate_degrees=1.5), frame_width=1080, frame_height=1920
    )
    assert "rotate=1.500*PI/180" in vf
    assert "crop=1080:1920" in vf


def test_mirror_and_colour_are_independent_filters() -> None:
    vf, _ = FFmpegVariationAdapter.build_filters(
        VariationProfile(mirror=True, brightness=0.02, contrast=1.05, saturation=1.03)
    )
    assert "hflip" in vf
    assert "eq=brightness=0.020:contrast=1.050:saturation=1.030" in vf


def test_pitch_shifts_audio_without_changing_duration_math() -> None:
    vf, af = FFmpegVariationAdapter.build_filters(VariationProfile(pitch_semitones=1.0))
    # A whole semitone is a 2**(1/12) rate ratio, corrected back by atempo.
    assert "asetrate=48000*1.059463" in af
    assert "atempo=0.943874" in af
    assert "aresample=48000" in af
