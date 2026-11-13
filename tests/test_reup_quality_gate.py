"""The reup gate must fail on a leaked caption, a merged slab, a face credit, and a missing dub."""

import numpy as np

from src.features.visual.reup_gate import (
    box_on_delivery_frame,
    delivery_sample_time,
    dub_coverage,
    gap_row_is_one_panel,
    source_still_visible,
    translated_face_credit,
    translated_production_credit,
)


def test_unchanged_crop_is_a_leak_and_a_painted_panel_is_not():
    source = np.full((40, 120, 3), 220, dtype=np.uint8)
    assert source_still_visible(source, source.copy())
    painted = np.full_like(source, 12)
    assert not source_still_visible(source, painted)


def test_costume_gap_is_two_rows_and_a_dark_span_is_one_slab():
    frame = np.full((200, 400, 3), 180, dtype=np.uint8)
    upper = (40, 20, 360, 80)
    lower = (100, 96, 300, 150)
    assert not gap_row_is_one_panel(frame, upper, lower)
    frame[80:96, :] = 10
    assert gap_row_is_one_panel(frame, upper, lower)


def test_face_credit_translation_fails_and_a_low_name_passes():
    face = (675, 481, 836, 518)
    name = (888, 860, 1064, 946)
    assert translated_face_credit(face, 1920, 1080, "READY")
    assert not translated_face_credit(name, 1920, 1080, "READY")
    assert not translated_face_credit(face, 1920, 1080, "FRAGMENT")
    assert translated_production_credit("READY", "※人形です")
    assert not translated_production_credit("CREDIT", "※人形です")
    assert not translated_production_credit("READY", "海军总部大将")


def test_delivery_frame_maps_speed_crop_and_mirror():
    assert delivery_sample_time(2.0, 1.25) == 1.6
    unchanged = box_on_delivery_frame((10, 20, 40, 50), 100, 80)
    assert unchanged == (10, 20, 40, 50)
    flipped = box_on_delivery_frame((10, 20, 40, 50), 100, 80, mirror=True)
    assert flipped == (60, 20, 90, 50)
    cropped = box_on_delivery_frame((100, 100, 200, 150), 1000, 500, crop_ratio=0.1)
    assert cropped[0] == 0
    assert cropped[2] == 125
    assert cropped[1] == 62.5
    assert cropped[3] == 125


def test_dub_coverage_requires_vietnamese_speech():
    ok, _detail = dub_coverage([
        {"source_text": "你好", "spoken_text": "Xin chào", "tts_provider": "edge-tts"},
        {"source_text": "再见", "spoken_text": "Tạm biệt", "tts_provider": "edge-tts"},
    ])
    assert ok
    missed, detail = dub_coverage([
        {"source_text": "你好", "spoken_text": "", "tts_provider": ""},
        {"source_text": "再见", "spoken_text": "Tạm biệt", "tts_provider": "edge-tts"},
    ])
    assert not missed
    assert "coverage" in detail
    cjk, cjk_detail = dub_coverage([
        {"source_text": "你好", "spoken_text": "你好", "tts_provider": "edge-tts"},
    ])
    assert not cjk
    assert "CJK" in cjk_detail
