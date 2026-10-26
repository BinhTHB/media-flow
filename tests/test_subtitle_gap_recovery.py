from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from src.core.models.bbox import BBox
from src.core.models.subtitle_event import SubtitleEvent
from src.features.visual.subtitle.processor import SubtitleProcessor


class _RepeatedOCR:
    def __init__(self, text: str) -> None:
        self.text = text

    def read_crops(self, crops):
        return [self.text for _ in crops]


def _event(event_id: str, start: float, end: float, *, w: int, h: int) -> SubtitleEvent:
    return SubtitleEvent(
        event_id=event_id,
        track_id="T001_subtitle",
        start_time=start,
        end_time=end,
        anchor_bbox=BBox(
            x1=w * 0.30,
            y1=h * 0.84,
            x2=w * 0.70,
            y2=h * 0.94,
            frame_width=w,
            frame_height=h,
        ),
    )


def _write_video(path: Path, *, w: int = 640, h: int = 360, fps: int = 10, seconds: float = 2.0) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))
    assert writer.isOpened()
    for _ in range(int(fps * seconds)):
        writer.write(np.zeros((h, w, 3), dtype=np.uint8))
    writer.release()


def test_recover_bottom_lane_gap_requires_repeated_ocr(tmp_path: Path):
    video = tmp_path / "gap.avi"
    _write_video(video)
    processor = SubtitleProcessor(ocr_provider=_RepeatedOCR("那年夏天很热\nIt was hot that summer"))
    events = [_event("A", 0.0, 0.4, w=640, h=360), _event("B", 1.4, 1.8, w=640, h=360)]

    recovered = processor._recover_bottom_lane_gap_events(events, str(video))
    synthetic = [event for event in recovered if event.split_reason == "bottom_lane_gap_recovery"]

    assert len(synthetic) == 1
    assert synthetic[0].source_text == "那年夏天很热"
    assert synthetic[0].start_time == 0.4
    assert synthetic[0].end_time == 1.4


def test_source_panel_covers_long_offcenter_bilingual_rows():
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    # Reproduce a wide source row whose right edge exceeds the stable panel.
    bbox = BBox(40, 610, 1250, 704, 1280, 720)
    frame[610:704, 40:1250] = 255
    result = SubtitleProcessor.suppress_source_subtitles(frame, [bbox])
    assert int(result[615:700, 45:1245].max()) < 30
    assert np.array_equal(result[:500], frame[:500])


def test_snug_plate_covers_given_box_and_stays_snug():
    # New contract: the plate hugs the boxes the caller passes. Coverage of a
    # caption's full extent is the CALLER's job (per-event trajectory union),
    # so the plate must cover the given box plus its edge pad — and must NOT
    # span a fixed frame-wide panel anymore.
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    frame[620:700, 900:1010] = 255
    full_row = BBox(20, 615, 1030, 704, 1280, 720)
    result = SubtitleProcessor.suppress_source_subtitles(frame, [full_row])
    assert int(result[620:700, 900:1010].max()) < 30  # covered: box + pad reach it
    # Snug: a single narrow box no longer expands to the old 0.72W panel.
    frame2 = np.zeros((720, 1280, 3), dtype=np.uint8)
    frame2[620:700, 500:560] = 255
    narrow = BBox(500, 615, 560, 704, 1280, 720)
    result2 = SubtitleProcessor.suppress_source_subtitles(frame2, [narrow])
    assert int(result2[630:695, 505:555].max()) < 30  # covered
    assert result2[640, 200].tolist() == [0, 0, 0]  # far left stays untouched


def test_parallel_bilingual_merge_preserves_union_temporal_coverage():
    top = SubtitleEvent(
        event_id="TOP",
        track_id="T001_subtitle",
        start_time=4.40,
        end_time=5.70,
        anchor_bbox=BBox(470, 615, 806, 680, 1280, 720),
    )
    bottom = SubtitleEvent(
        event_id="BOTTOM",
        track_id="T002_subtitle",
        start_time=4.53,
        end_time=5.33,
        anchor_bbox=BBox(496, 672, 767, 704, 1280, 720),
    )

    merged = SubtitleProcessor._merge_parallel_bilingual_events([top, bottom])

    assert len(merged) == 1
    assert merged[0].start_time == 4.40
    assert merged[0].end_time == 5.70


def test_recover_bottom_lane_gap_rejects_single_unstable_guess(tmp_path: Path):
    video = tmp_path / "gap.avi"
    _write_video(video)

    class _UnstableOCR:
        def read_crops(self, crops):
            values = ["那年夏天很热", "", "不同文本", ""]
            return [values[i % len(values)] for i in range(len(crops))]

    processor = SubtitleProcessor(ocr_provider=_UnstableOCR())
    events = [_event("A", 0.0, 0.4, w=640, h=360), _event("B", 1.4, 1.8, w=640, h=360)]

    recovered = processor._recover_bottom_lane_gap_events(events, str(video))
    assert all(event.split_reason != "bottom_lane_gap_recovery" for event in recovered)
