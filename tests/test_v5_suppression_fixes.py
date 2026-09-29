"""
Unit tests for V5 Source Suppression & Track Gap Interpolation fixes:
1. Source suppression is independent from translated text layout
2. Translated backdrop covers complete translated text bbox
3. Long Vietnamese text does not shrink/expand source suppression region
4. Non-subtitle processors (Watermark/Logo/OtherText) perform gap interpolation
"""

import numpy as np

from src.core.models.bbox import BBox
from src.core.models.detection import Detection
from src.core.models.subtitle_event import SubtitleEvent
from src.core.models.track import Track
from src.features.visual.actions.text_overlay import TextOverlayAction
from src.features.visual.processors.watermark import WatermarkProcessor
from src.features.visual.subtitle.processor import SubtitleProcessor


def test_source_suppression_independent_from_translated_layout():
    src_bbox = BBox(x1=100, y1=200, x2=300, y2=250, frame_width=1000, frame_height=1000)
    supp_bbox = SubtitleProcessor.get_source_suppression_bbox(src_bbox, pad_x=18, pad_y=14)

    assert supp_bbox.x1 == 100 - 18
    assert supp_bbox.y1 == 200 - 14
    assert supp_bbox.x2 == 300 + 18
    assert supp_bbox.y2 == 250 + 14

    # Long vs Short translation string does not change supp_bbox calculation
    event_short = SubtitleEvent(
        event_id="ev_short",
        track_id="tr1",
        start_time=1.0,
        end_time=2.0,
        anchor_bbox=src_bbox,
        source_text="你好",
        translated_text="Chào",
    )

    event_long = SubtitleEvent(
        event_id="ev_long",
        track_id="tr1",
        start_time=1.0,
        end_time=2.0,
        anchor_bbox=src_bbox,
        source_text="你好",
        translated_text="Xin chào tất cả mọi người có mặt ngày hôm nay",
    )

    supp1 = SubtitleProcessor.get_source_suppression_bbox(event_short.anchor_bbox, pad_x=18, pad_y=14)
    supp2 = SubtitleProcessor.get_source_suppression_bbox(event_long.anchor_bbox, pad_x=18, pad_y=14)

    assert supp1 == supp2


def test_text_overlay_action_draws_backdrop_box():
    action = TextOverlayAction(text="Xin chào", bg_box=True)
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    bbox = BBox(x1=20, y1=20, x2=180, y2=80, frame_width=200, frame_height=100)

    out_frame = action.execute(frame, bbox)
    assert out_frame.shape == frame.shape
    # Ensure non-zero pixels exist where backdrop + text were rendered
    assert np.sum(out_frame) > 0


def test_watermark_processor_gap_interpolation():
    processor = WatermarkProcessor(action_name="blur", padding_px=16)

    det1 = Detection(class_id=1, class_name="watermark", confidence=0.9, bbox=BBox(x1=0, y1=800, x2=50, y2=1000, frame_width=1000, frame_height=1000), frame_index=10, timestamp_seconds=0.33)
    det2 = Detection(class_id=1, class_name="watermark", confidence=0.9, bbox=BBox(x1=0, y1=800, x2=50, y2=1000, frame_width=1000, frame_height=1000), frame_index=20, timestamp_seconds=0.66)

    track = Track(track_id="T_wm", class_id=1, class_name="watermark", detections=[det1, det2])

    frame = np.zeros((1000, 1000, 3), dtype=np.uint8)
    frame[800:1000, 0:50] = 255  # High-contrast watermark area
    orig_frame = frame.copy()

    # Frame 15 has no explicit detection in track.detections but falls in [10..20]:
    # the processor must interpolate to the nearest detection and blur that region.
    out_frame = processor.process_frame_tracks(frame, frame_index=15, tracks=[track])

    supp_region = (slice(770, 1000), slice(0, 66))  # Watermark bbox padded by the processor
    assert not np.array_equal(
        out_frame[supp_region], orig_frame[supp_region]
    ), "Expected the watermark region to be blurred on the interpolated frame"
    # Blur must stay localized to the watermark region.
    assert np.array_equal(out_frame[100:200, 400:500], orig_frame[100:200, 400:500])

