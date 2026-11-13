"""Checks a finished reup job against the failures that block publishing.

The finished file may already include the delivery variation (speed, crop,
mirror, rotate). Sample times and boxes are mapped onto that file so the
gate judges the video that will be uploaded.

A passing job has covered source captions, separate caption rows, no
translated face credit, and Vietnamese speech on the dubbed lines. The
functions here are pure so a test can fail them without a video file.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from src.features.translation.text_quality import (
    detect_script,
    is_production_credit,
    looks_vietnamese,
)
from src.features.visual.text_classifier import (
    CREDIT_MAX_CENTER_Y_RATIO,
    CREDIT_MAX_HEIGHT_RATIO,
    CREDIT_MAX_WIDTH_RATIO,
)

#: Mean absolute pixel difference below this means the output still shows
#: the source crop. A painted panel or a new caption sits well above it.
SOURCE_UNCHANGED_MAD = 12.0
#: A gap row this dark across the wider caption is one painted slab.
SLAB_DARK_LEVEL = 40
SLAB_DARK_RATIO = 0.85
#: Share of speech lines that must come back as Vietnamese TTS.
DUB_COVERAGE_MIN = 0.8


def delivery_sample_time(master_time: float, speed: float) -> float:
    """Map a master timestamp onto the sped-up delivery file."""
    factor = float(speed) if speed else 1.0
    if factor <= 0.0:
        factor = 1.0
    return float(master_time) / factor


def box_on_delivery_frame(
    box: Sequence[float],
    frame_width: float,
    frame_height: float,
    crop_ratio: float = 0.0,
    mirror: bool = False,
    rotate_degrees: float = 0.0,
) -> tuple[float, float, float, float]:
    """Map a master-frame box onto the delivery frame.

    The delivery filter crops each edge, scales back to the original size,
    optionally flips horizontally, then rotates about the center. The returned
    box is the axis-aligned bounds of that region, clipped to the frame.
    """
    width = float(frame_width)
    height = float(frame_height)
    if width <= 1.0 or height <= 1.0:
        return (float(box[0]), float(box[1]), float(box[2]), float(box[3]))

    crop = min(max(float(crop_ratio), 0.0), 0.49)
    keep = 1.0 - 2.0 * crop
    corners = (
        (float(box[0]), float(box[1])),
        (float(box[2]), float(box[1])),
        (float(box[2]), float(box[3])),
        (float(box[0]), float(box[3])),
    )
    mapped = [_map_delivery_point(x, y, width, height, crop, keep, mirror, rotate_degrees) for x, y in corners]
    xs = [point[0] for point in mapped]
    ys = [point[1] for point in mapped]
    return (
        max(0.0, min(xs)),
        max(0.0, min(ys)),
        min(width, max(xs)),
        min(height, max(ys)),
    )


def _map_delivery_point(
    x: float,
    y: float,
    width: float,
    height: float,
    crop: float,
    keep: float,
    mirror: bool,
    rotate_degrees: float,
) -> tuple[float, float]:
    mapped_x = (x - crop * width) / keep
    mapped_y = (y - crop * height) / keep
    if mirror:
        mapped_x = width - mapped_x
    if rotate_degrees:
        theta = math.radians(float(rotate_degrees))
        dx = mapped_x - width / 2.0
        dy = mapped_y - height / 2.0
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        mapped_x = width / 2.0 + dx * cos_t - dy * sin_t
        mapped_y = height / 2.0 + dx * sin_t + dy * cos_t
    return mapped_x, mapped_y


def source_still_visible(source_crop: np.ndarray, output_crop: np.ndarray) -> bool:
    """True when the output crop is still the source image."""
    if source_crop.shape != output_crop.shape or source_crop.size == 0:
        return False
    mad = float(np.mean(np.abs(source_crop.astype(np.int16) - output_crop.astype(np.int16))))
    return mad < SOURCE_UNCHANGED_MAD


def gap_row_is_one_panel(frame: np.ndarray, upper: Sequence[float], lower: Sequence[float]) -> bool:
    """True when two stacked rows were painted as one black rectangle.

    ``upper`` and ``lower`` are ``(x1, y1, x2, y2)``. A real gap shows the
    costume. A merged plate stays dark across the wider row.
    """
    upper_bottom = int(upper[3])
    lower_top = int(lower[1])
    if lower_top - upper_bottom > 24 or lower_top <= upper_bottom:
        return False
    mid = (upper_bottom + lower_top) // 2
    if mid < 0 or mid >= frame.shape[0]:
        return False
    x1 = int(min(upper[0], lower[0]))
    x2 = int(max(upper[2], lower[2]))
    x1 = max(0, x1)
    x2 = min(frame.shape[1], x2)
    if x2 - x1 < 8:
        return False
    row = frame[mid, x1:x2]
    dark = np.max(row, axis=-1) < SLAB_DARK_LEVEL
    return float(np.mean(dark)) >= SLAB_DARK_RATIO


def translated_face_credit(
    box: Sequence[float],
    frame_width: float,
    frame_height: float,
    status: str,
) -> bool:
    """True when a thin box above the caption band was translated and drawn."""
    if status != "READY":
        return False
    width = float(box[2]) - float(box[0])
    height = float(box[3]) - float(box[1])
    center_y = (float(box[1]) + float(box[3])) / 2.0
    return (
        height < CREDIT_MAX_HEIGHT_RATIO * frame_height
        and width < CREDIT_MAX_WIDTH_RATIO * frame_width
        and center_y < CREDIT_MAX_CENTER_Y_RATIO * frame_height
    )


def translated_production_credit(status: str, source_text: str) -> bool:
    """True when a credit or disclaimer was drawn as a translated caption."""
    return status == "READY" and is_production_credit(source_text)


def dub_coverage(segments: Sequence[dict]) -> tuple[bool, str]:
    """True when speech lines were spoken in Vietnamese.

    A line with no source text is silence, not a missed dub. A dubbed line
    that is still CJK, or a batch where most lines never received a voice,
    fails.
    """
    speech = [item for item in segments if str(item.get("source_text") or "").strip()]
    if not speech:
        return False, "no speech segments"
    dubbed = []
    for item in speech:
        spoken = str(item.get("spoken_text") or "").strip()
        provider = str(item.get("tts_provider") or "").strip()
        if not spoken or not provider:
            continue
        if detect_script(spoken) == "cjk":
            return False, f"dub still CJK: {spoken}"
        if looks_vietnamese(spoken):
            dubbed.append(item)
    ratio = len(dubbed) / len(speech)
    if ratio < DUB_COVERAGE_MIN:
        return False, f"vietnamese dub coverage {ratio:.0%} of {len(speech)} lines"
    return True, f"{len(dubbed)}/{len(speech)} lines dubbed in Vietnamese"
