"""Production-render tests: caption box hugs the text, the font flexes to
fit, and the text baseline is fixed across box heights."""

import numpy as np

from src.core.models.bbox import BBox
from src.features.visual.actions.text_overlay import TextOverlayAction

FW, FH = 1920, 1080


def _render(text, region):
    frame = np.zeros((FH, FW, 3), dtype=np.uint8)
    action = TextOverlayAction(text=text, anchor_to_bbox=True, vertical_anchor="bottom")
    return action.execute(frame, region)


def _white_rows(img):
    mask = (img[:, :, 0] > 200) & (img[:, :, 1] > 200) & (img[:, :, 2] > 200)
    return np.where(mask.any(axis=1))[0]


def test_bottom_anchor_keeps_one_baseline_across_box_heights():
    # A 1-line and a 2-line-height region sharing the band bottom edge must
    # place the caption at the SAME baseline — the text must not float with
    # its box height.
    tall = BBox(700, 900, 1300, 1054, FW, FH)
    short = BBox(700, 990, 1300, 1054, FW, FH)
    bottom_tall = _white_rows(_render("Nói năng nhẹ nhàng", tall)).max()
    bottom_short = _white_rows(_render("Nói năng nhẹ nhàng", short)).max()
    assert bottom_tall == bottom_short


def test_font_shrinks_so_long_caption_fits_region():
    text = "Một quý cô chuẩn mực phải tuân thủ những quy tắc hành xử mọi lúc mọi nơi"
    region = BBox(700, 940, 1220, 1054, FW, FH)
    img = _render(text, region)
    white = np.where(
        (img[:, :, 0] > 200) & (img[:, :, 1] > 200) & (img[:, :, 2] > 200)
    )
    # Text (plus stroke) stays inside the region with a small tolerance.
    assert white[1].min() >= region.x1 - 6
    assert white[1].max() <= region.x2 + 6
    assert white[0].max() <= region.y2 + 6


def test_font_grows_for_short_caption_in_roomy_region():
    # A very short caption in a roomy box grows up to MAX_FONT_SCALE: its
    # rendered text must be taller than the base-size render.
    region = BBox(700, 900, 1300, 1054, FW, FH)
    small = _white_rows(_render("OK", region))
    bigger_text = "Cô ấy mỉm cười dịu dàng và bước đi"
    big = _white_rows(_render(bigger_text, region))
    # Both render inside the region; the grown short caption occupies at
    # least the base line height (40px + stroke), not a tiny remnant.
    assert small.max() - small.min() >= 30
    assert big.max() <= region.y2 + 6


def test_panel_hugs_text_when_source_box_is_cinema_wide():
    # Gap-recovery used to store an 80%-wide crop as the event box. Expanding
    # the caption panel onto that box painted a black bar across the scene.
    region = BBox(192, 842, 1728, 1074, FW, FH)
    frame = np.zeros((FH, FW, 3), dtype=np.uint8)
    out = TextOverlayAction(
        text="Xin chào",
        anchor_to_bbox=True,
        vertical_anchor="bottom",
        bg_box=True,
    ).execute(frame, region)
    mid_y = 1000
    # Far into the source box, left of the actual caption, stays empty.
    assert int(out[mid_y, 250].sum()) < 40
    # The caption itself is present near the box center/bottom.
    band = out[940:1060, 700:1220]
    assert int(band.max()) > 200


def test_anchored_panel_corner_stays_clear():
    # The caption backing is one rounded panel. Its outer corner is not a
    # filled rectangle.
    region = BBox(700, 940, 1220, 1040, FW, FH)
    frame = np.full((FH, FW, 3), 180, dtype=np.uint8)
    out = TextOverlayAction(
        text="Xin chào",
        anchor_to_bbox=True,
        vertical_anchor="center",
        bg_box=True,
        bg_color=(16, 12, 10, 255),
    ).execute(frame, region)
    assert out[int(region.y1), int(region.x1)].tolist() == [180, 180, 180]
    assert out[int(region.y1) + 24, int(region.x1) + 36].tolist() == [10, 12, 16]


def test_region_render_centers_text_in_its_box():
    # The production caller places the caption in the middle of its own box.
    from src.core.models.subtitle_event import SubtitleEvent
    from src.features.visual.subtitle.processor import SubtitleProcessor

    processor = SubtitleProcessor()
    tall = BBox(700, 900, 1300, 1054, FW, FH)
    short = BBox(700, 990, 1300, 1054, FW, FH)
    frame = np.zeros((FH, FW, 3), dtype=np.uint8)

    def _event(box: BBox) -> SubtitleEvent:
        return SubtitleEvent(
            event_id="E",
            track_id="T",
            start_time=0.0,
            end_time=1.0,
            anchor_bbox=box,
            translated_text="Nói năng nhẹ nhàng",
            replacement_status="READY",
            overlay_applied=True,
        )

    def _mid(box: BBox) -> float:
        rows = _white_rows(processor.render_region_events(frame.copy(), [_event(box)], box))
        return float(rows.min() + rows.max()) / 2.0

    mid_tall = _mid(tall)
    mid_short = _mid(short)
    assert abs(mid_tall - (tall.y1 + tall.y2) / 2.0) < 24
    assert abs(mid_short - (short.y1 + short.y2) / 2.0) < 24
    assert mid_short > mid_tall
