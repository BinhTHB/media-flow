from __future__ import annotations

import cv2
import numpy as np
import pytest

from src.core.models.bbox import BBox
from src.core.models.subtitle_event import SubtitleEvent
from src.features.visual.subtitle.boundary_refiner import (
    BoundaryRefinementConfig,
    SubtitleBoundaryRefiner,
)
from src.features.visual.subtitle.processor import SubtitleProcessor


def _bbox(x1=100, y1=120, x2=220, y2=155):
    return BBox(x1=x1, y1=y1, x2=x2, y2=y2, frame_width=320, frame_height=180)


def test_boundary_refiner_recovers_adjacent_missed_frames(tmp_path):
    path = tmp_path / "timing.avi"
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30.0, (320, 180)
    )
    assert writer.isOpened()
    for frame_index in range(10):
        frame = np.zeros((180, 320, 3), dtype=np.uint8)
        if 3 <= frame_index <= 6:
            cv2.putText(
                frame,
                "HELLO",
                (105, 145),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
        writer.write(frame)
    writer.release()

    event = SubtitleEvent(
        event_id="E1",
        track_id="T1",
        start_time=4 / 30.0,
        end_time=5 / 30.0,
        anchor_bbox=_bbox(),
        bbox_trajectory=[_bbox(), _bbox()],
        bbox_frame_indices=[4, 5],
    )
    refiner = SubtitleBoundaryRefiner(
        BoundaryRefinementConfig(
            scan_back_seconds=0.2,
            scan_forward_seconds=0.2,
            present_threshold=0.42,
            absence_confirm_frames=2,
        )
    )

    refined = refiner.refine([event], str(path))[0]

    assert refined.start_time == pytest.approx(3 / 30.0, abs=1e-3)
    assert refined.end_time == pytest.approx(6 / 30.0, abs=1e-3)


def test_boundary_refiner_does_not_blindly_extend_into_different_content(tmp_path):
    path = tmp_path / "content-change.avi"
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30.0, (320, 180)
    )
    assert writer.isOpened()
    for frame_index in range(8):
        frame = np.zeros((180, 320, 3), dtype=np.uint8)
        label = "OLD" if frame_index <= 4 else "NEW"
        cv2.putText(
            frame,
            label,
            (110, 145),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        writer.write(frame)
    writer.release()

    event = SubtitleEvent(
        event_id="E1",
        track_id="T1",
        start_time=2 / 30.0,
        end_time=4 / 30.0,
        anchor_bbox=_bbox(),
        bbox_trajectory=[_bbox(), _bbox(), _bbox()],
        bbox_frame_indices=[2, 3, 4],
    )
    refined = SubtitleBoundaryRefiner().refine([event], str(path))[0]

    # NEW starts at frame 5, so the OLD event must not be blindly padded into it.
    assert refined.end_time <= (4 / 30.0) + 1e-3


def test_bilingual_source_selector_prefers_cjk_row_over_duplicate_english():
    text = "热浪扭曲着空气却还不够让人燥热\nHeat waves distort the air, yet the heat is not sweltering enough."
    assert SubtitleProcessor._select_bilingual_source_text(text) == "热浪扭曲着空气却还不够让人燥热"


def test_parallel_bilingual_events_merge_into_one_union_event():
    top = SubtitleEvent(
        event_id="TOP",
        track_id="T_TOP",
        start_time=1.0,
        end_time=3.0,
        anchor_bbox=_bbox(90, 135, 230, 155),
    )
    bottom = SubtitleEvent(
        event_id="BOTTOM",
        track_id="T_BOTTOM",
        start_time=1.1,
        end_time=2.9,
        anchor_bbox=_bbox(85, 160, 235, 175),
    )
    merged = SubtitleProcessor._merge_parallel_bilingual_events([top, bottom])
    assert len(merged) == 1
    event = merged[0]
    assert event.split_reason == "parallel_bilingual_merge"
    assert event.start_time == pytest.approx(1.0)
    assert event.end_time == pytest.approx(3.0)
    assert event.anchor_bbox.y1 == 135
    assert event.anchor_bbox.y2 == 175


def test_parallel_bilingual_merge_does_not_drop_long_event_coverage():
    long_top = SubtitleEvent(
        event_id="TOP_LONG",
        track_id="T_TOP",
        start_time=2.3,
        end_time=5.3,
        anchor_bbox=_bbox(90, 135, 230, 155),
    )
    short_bottom = SubtitleEvent(
        event_id="BOTTOM_SHORT",
        track_id="T_BOTTOM",
        start_time=4.5,
        end_time=5.3,
        anchor_bbox=_bbox(85, 160, 235, 175),
    )
    merged = SubtitleProcessor._merge_parallel_bilingual_events([long_top, short_bottom])
    assert len(merged) == 2
    assert any(event.event_id == "TOP_LONG" and event.start_time == pytest.approx(2.3) for event in merged)


def test_trajectory_bbox_interpolates_for_missing_detection_frame():
    left = _bbox(100, 120, 180, 150)
    right = _bbox(120, 120, 200, 150)
    event = SubtitleEvent(
        event_id="E1",
        track_id="T1",
        start_time=0.0,
        end_time=1.0,
        anchor_bbox=left,
        bbox_trajectory=[left, right],
        bbox_frame_indices=[10, 12],
    )

    mid = SubtitleProcessor.get_event_bbox(event, 11)

    assert mid.x1 == pytest.approx(110.0)
    assert mid.x2 == pytest.approx(190.0)
