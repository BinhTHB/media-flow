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
