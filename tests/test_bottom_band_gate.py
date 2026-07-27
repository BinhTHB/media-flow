"""Tests for the bottom-band gate module."""
from __future__ import annotations

import pytest

from src.infrastructure.benchmark.hybrid_subtitle.bottom_band_gate import (
    passes_bottom_band,
    track_y1_norm,
)


def _make_track(y1: float, y2: float) -> dict:
    return {
        "track_id": "T001",
        "median_bbox_xyxy": [100.0, y1, 400.0, y2],
        "groups": [{"union_bbox_xyxy": [100.0, y1, 400.0, y2]}],
    }


def test_typical_bottom_subtitle_y1_norm():
    assert track_y1_norm(_make_track(949.0, 1040.0), 1280) == 949.0 / 1280.0


def test_mid_screen_hud_y1_norm():
    assert track_y1_norm(_make_track(500.0, 600.0), 1280) == pytest.approx(0.3906, abs=0.001)


def test_bottom_subtitle_passes_selected_threshold():
    assert passes_bottom_band(_make_track(949.0, 1040.0), 1280, 0.68) is True


def test_mid_hud_rejected_before_classifier():
    assert passes_bottom_band(_make_track(500.0, 600.0), 1280, 0.68) is False


def test_t042_mid_negative_rejected():
    assert passes_bottom_band(_make_track(857.0, 925.0), 1280, 0.68) is False


def test_t036_bottom_negative_requires_later_discriminator():
    assert passes_bottom_band(_make_track(1100.0, 1259.0), 1280, 0.68) is True


def test_missing_bbox_fails_closed():
    assert passes_bottom_band({"track_id": "T001", "groups": []}, 1280, 0.68) is False
