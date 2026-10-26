"""Regression tests: translated captions must render at ONE stable position
per event. Layout geometry comes from the event's averaged trajectory box;
per-frame detections feed suppression only."""

import logging
from unittest.mock import MagicMock

import pytest

from src.core.models.bbox import BBox
from src.core.models.detection import Detection
from src.core.models.subtitle_event import SubtitleEvent
from src.features.visual.pipeline import (
    VisualPipeline,
    VisualRequest,
    _layout_box_for_event,
    _layout_boxes_for_event,
    _stable_layout_box,
)
from src.features.visual.subtitle.processor import SubtitleProcessor

FW, FH = 1920, 1080


def _box(x1, y1, x2, y2):
    return BBox(x1=x1, y1=y1, x2=x2, y2=y2, frame_width=FW, frame_height=FH)


def _event(trajectory, event_id="E1"):
    anchor = trajectory[len(trajectory) // 2] if trajectory else _box(100, 900, 800, 980)
    return SubtitleEvent(
        event_id=event_id,
        track_id="T1",
        start_time=0.0,
        end_time=2.0,
        anchor_bbox=anchor,
        bbox_trajectory=list(trajectory),
        bbox_frame_indices=list(range(len(trajectory))),
        source_text="你好",
        translated_text="Xin chào",
        replacement_status="READY",
        blur_applied=True,
        overlay_applied=True,
    )


def _det(frame_index, box, confidence=0.9):
    return Detection(
        class_id=0, class_name="subtitle", confidence=confidence,
        bbox=box, frame_index=frame_index, timestamp_seconds=frame_index / 30.0,
    )


class _LayoutRecordingProcessor:
    """Records the exact region render_region_events receives."""

    def __init__(self, frame_boxes=None):
        self.render_calls = []
        self.suppress_calls = []
        self.frame_boxes = frame_boxes or {}

    def suppress_source_subtitles(self, frame, regions):
        self.suppress_calls.append(list(regions))
        return frame

    def render_region_events(self, frame, events, region, bg_box=True):
        self.render_calls.append(region)
        return frame

    select_render_events = staticmethod(SubtitleProcessor.select_render_events)

    @staticmethod
    def get_source_suppression_bbox(bbox, pad_x=18, pad_y=14):
        return bbox

    def get_event_bbox(self, event, frame_index):
        # Simulates per-frame trajectory interpolation of the legacy path.
        return self.frame_boxes.get(frame_index, event.anchor_bbox)


def test_stable_layout_box_averages_and_clamps():
    traj = [_box(100, 900, 800, 980), _box(104, 902, 796, 978), _box(108, 904, 792, 976)]
    box = _stable_layout_box(traj, FW, FH)
    assert box.x1 == pytest.approx(104.0)
    assert box.y1 == pytest.approx(902.0)
    assert box.x2 == pytest.approx(796.0)
    assert box.y2 == pytest.approx(978.0)
    # BBox validates its own bounds at construction (0 <= x1 < x2 <= frame),
    # so a mean of valid boxes is always in-bounds; the clamp call stays as
    # defense for float drift at the edges.
    assert 0 <= box.x1 < box.x2 <= FW
    assert 0 <= box.y1 < box.y2 <= FH


def test_layout_box_for_event_falls_back_to_anchor_when_trajectory_empty():
    anchor = _box(200, 910, 900, 990)
    event = _event([], event_id="merged")
    event.anchor_bbox = anchor
    box = _layout_box_for_event(event)
    assert (box.x1, box.y1, box.x2, box.y2) == (anchor.x1, anchor.y1, anchor.x2, anchor.y2)


def test_render_position_identical_across_frames_with_different_detections():
    processor = _LayoutRecordingProcessor()
    pipeline = VisualPipeline(subtitle_processor=processor)
    pipeline._region_planner = MagicMock()

    ev = _event([_box(660, 940, 1260, 1020)])
    layout = {ev.event_id: _layout_box_for_event(ev)}
    request = VisualRequest(subtitle_enabled=True, subtitle_mode="translate_replace", subtitle_layout="in-place")

    # Word-dropout: frame 45 sees a wide union, frame 46 a narrow one.
    pipeline._apply_subtitle_actions(
        object(), 45, request,
        {45: [_det(45, _box(640, 935, 1280, 1025))]},
        {45: [ev]},
        event_layout_boxes=layout,
    )
    pipeline._apply_subtitle_actions(
        object(), 46, request,
        {46: [_det(46, _box(700, 945, 1100, 1015))]},
        {46: [ev]},
        event_layout_boxes=layout,
    )

    assert len(processor.render_calls) == 2
    first, second = processor.render_calls
    assert (first.x1, first.y1, first.x2, first.y2) == (second.x1, second.y1, second.x2, second.y2)
    # Suppression uses the SAME stable box (padded by LAYOUT_BOX_PAD_RATIO,
    # 0.015 x frame height = 16.2px at 1080p) — never the per-frame detections,
    # whose unions made the inpainted band shimmer.
    pad = 0.015 * FH
    expected = (layout[ev.event_id].x1 - pad, layout[ev.event_id].y1 - pad,
                layout[ev.event_id].x2 + pad, layout[ev.event_id].y2 + pad)
    assert len(processor.suppress_calls) == 2
    for regions in processor.suppress_calls:
        assert len(regions) == 1
        assert (regions[0].x1, regions[0].y1, regions[0].x2, regions[0].y2) == pytest.approx(expected)


def test_flag_false_restores_per_frame_behavior():
    processor = _LayoutRecordingProcessor(frame_boxes={
        45: _box(600, 930, 1300, 1030),
        46: _box(760, 950, 1080, 1010),
    })
    pipeline = VisualPipeline(subtitle_processor=processor)
    pipeline._region_planner = MagicMock()
    # Legacy path consults region_for; return the per-frame anchor itself
    # (no lane match) so same_lane arithmetic operates on a real BBox.
    pipeline._region_planner.region_for.side_effect = lambda frame_index, anchor: anchor

    ev = _event([_box(660, 940, 1260, 1020)])
    request = VisualRequest(
        subtitle_enabled=True, subtitle_mode="translate_replace",
        subtitle_layout="in-place", stable_subtitle_layout=False,
    )

    pipeline._apply_subtitle_actions(
        object(), 45, request,
        {45: [_det(45, _box(600, 930, 1300, 1030))]},
        {45: [ev]},
        event_layout_boxes={},
    )
    pipeline._apply_subtitle_actions(
        object(), 46, request,
        {46: [_det(46, _box(760, 950, 1080, 1010))]},
        {46: [ev]},
        event_layout_boxes={},
    )

    assert len(processor.render_calls) == 2
    first, second = processor.render_calls
    # Legacy tight_box follows the per-frame detections: positions differ.
    assert (first.x1, first.y1, first.x2, first.y2) != (second.x1, second.y1, second.x2, second.y2)


def test_large_trajectory_spread_logs_warning(caplog):
    # 50/50 two-position trajectory: no dominant cluster exists, so the mean
    # box sits between the positions and a warning must say so.
    far_boxes = [_box(100, 900, 500, 980), _box(1300, 200, 1800, 280)]
    event = _event(far_boxes, event_id="bimodal")
    with caplog.at_level(logging.WARNING, logger="src.features.visual.pipeline"):
        _layout_box_for_event(event)
    assert any("no dominant cluster" in record.message for record in caplog.records)


def test_spread_trajectory_renders_dominant_cluster_and_suppresses_rest():
    # Real case (T100): the same caption appeared at two positions, 70/30.
    # The mean box sat between them (huge, shifted); now the render box is
    # the dominant cluster's mean and the leftover position is suppressed.
    boxes = [_box(100, 900, 500, 980)] * 7 + [_box(1300, 200, 1800, 280)] * 3
    event = _event(boxes, event_id="T100")
    layout, plate, leftovers = _layout_boxes_for_event(event)
    # Dominant cluster mean == the 7 identical boxes.
    assert (layout.x1, layout.y1, layout.x2, layout.y2) == pytest.approx((100, 900, 500, 980))
    assert len(leftovers) == 1
    assert (leftovers[0].x1, leftovers[0].y1, leftovers[0].x2, leftovers[0].y2) == pytest.approx(
        (1300, 200, 1800, 280)
    )


def test_ocr_empty_events_never_redacted_failed_ones_do():
    # Evidence rule: OCR_EMPTY means no readable caption was found (detector
    # hallucination on scenery, watermark edges) — redacting painted empty
    # boxes over the frame (seen at 19.5s, 20.3s, 21.2s). They must never
    # redact, at any span. Translation failures keep the fail-safe redaction:
    # text WAS there, only the translation is missing.
    from src.features.visual.pipeline import _build_suppress_only_map

    empty_ev = _event([_box(456, 716, 936, 843)], event_id="hallucination")
    empty_ev.start_time, empty_ev.end_time = 19.50, 19.77
    empty_ev.replacement_status = "OCR_EMPTY"
    failed_ev = _event([_box(660, 940, 1260, 1020)], event_id="failed")
    failed_ev.start_time, failed_ev.end_time = 21.0, 22.0
    failed_ev.replacement_status = "TRANSLATION_FAILED"
    layouts = {
        "hallucination": _box(456, 716, 936, 843),
        "failed": _box(660, 940, 1260, 1020),
    }

    mapping = _build_suppress_only_map([empty_ev, failed_ev], layouts, {}, 29.97)

    assert not any(583 <= f <= 594 for f in mapping)  # 19.5-19.77s: nothing painted
    assert 628 in mapping and 661 in mapping  # failed caption: redacted (21.0-22.0s ± 2)


def test_band_normalization_snaps_bottom_and_center():
    # Real data (last run): consecutive caption bottoms wobbled 1045-1058 and
    # x-centers shifted -193..+439 between events. Near-band READY captions
    # snap to one shared bottom edge and, when close, one shared center.
    from src.features.visual.pipeline import _apply_band_normalization

    evs = {}
    boxes = {}
    for i, (x1, x2) in enumerate([(660, 1260), (700, 1310), (620, 1210)]):
        ev = _event([_box(x1, 990, x2, 1052 + i * 3)], event_id=f"e{i}")
        evs[ev.event_id] = ev
        boxes[ev.event_id] = _box(x1, 990, x2, 1052 + i * 3)

    normalized = _apply_band_normalization(boxes, evs)

    y2s = {round(b.y2) for b in normalized.values()}
    assert len(y2s) == 1  # one shared bottom edge
    cxs = {round((b.x1 + b.x2) / 2) for b in normalized.values()}
    assert len(cxs) == 1  # x-centers within tolerance re-centered


def test_band_normalization_keeps_off_band_event_untouched():
    from src.features.visual.pipeline import _apply_band_normalization

    evs = {}
    boxes = {}
    for i, (x1, x2) in enumerate([(660, 1260), (700, 1310), (620, 1210)]):
        ev = _event([_box(x1, 990, x2, 1052)], event_id=f"e{i}")
        evs[ev.event_id] = ev
        boxes[ev.event_id] = _box(x1, 990, x2, 1052)
    # A top caption (upper row of a bilingual pair) must keep its own band.
    top = _event([_box(844, 924, 1076, 985)], event_id="top")
    evs["top"] = top
    boxes["top"] = _box(844, 924, 1076, 985)

    normalized = _apply_band_normalization(boxes, evs)

    assert (normalized["top"].x1, normalized["top"].y1) == (844, 924)
    # Bottom events snapped together; the top one is far from the band.
    y2s = {round(normalized[f"e{i}"].y2) for i in range(3)}
    assert len(y2s) == 1


def test_selection_stable_when_short_fragment_overlaps_long_event():
    # Real regression: a short duplicate fragment (0.4s, CJK-preferred) stole
    # the render mid-span from the full-length caption, so the caption
    # switched text/position at the fragment's edges. Duplicate tracks of the
    # SAME visual line (vertically overlapping boxes) collapse to one; the
    # longest event wins deterministically.
    long_ev = _event([_box(660, 940, 1260, 1020)], event_id="T112_E03")
    long_ev.start_time, long_ev.end_time = 12.07, 12.87
    long_ev.source_text = "soft eyes"

    frag = _event([_box(700, 955, 1100, 1005)], event_id="T111_E05")
    frag.start_time, frag.end_time = 12.23, 12.63
    frag.source_text = "眼神柔和"

    at_12_3 = SubtitleProcessor.select_render_events([long_ev, frag])
    assert [e.event_id for e in at_12_3] == ["T112_E03"]


def test_selection_keeps_distinct_stacked_rows():
    # A persistent title row above a content row has a disjoint y-range and
    # a different meaning: both must survive selection.
    title = _event([_box(688, 759, 1260, 849)], event_id="title")
    title.start_time, title.end_time = 21.8, 26.8
    content = _event([_box(713, 860, 1206, 1043)], event_id="content")
    content.start_time, content.end_time = 21.8, 24.7

    chosen = SubtitleProcessor.select_render_events([title, content])
    assert [e.event_id for e in chosen] == ["title", "content"]


def test_incoming_event_preroll_suppresses_early_but_does_not_render():
    # Real regression seen on T098: the NEXT caption's pre-roll started
    # rendering 3 frames early at a different layout box, so the current
    # caption jumped before the new one actually appeared.
    processor = _LayoutRecordingProcessor()
    pipeline = VisualPipeline(subtitle_processor=processor)
    pipeline._region_planner = MagicMock()

    current_ev = _event([_box(660, 940, 1260, 1020)], event_id="current")
    current_ev.start_time, current_ev.end_time = 0.0, 1.0
    next_ev = _event([_box(300, 700, 900, 780)], event_id="next")
    next_ev.start_time, next_ev.end_time = 1.1, 2.0

    layout = {
        "current": _layout_box_for_event(current_ev),
        "next": _layout_box_for_event(next_ev),
    }
    # Frame 32 is inside the next event's pre-roll (its span starts at 33).
    request = VisualRequest(subtitle_enabled=True, subtitle_mode="translate_replace", subtitle_layout="in-place")
    pipeline._apply_subtitle_actions(
        object(), 32, request,
        {32: [_det(32, _box(660, 940, 1260, 1020))]},
        {32: [current_ev, next_ev]},
        event_layout_boxes=layout,
        event_render_spans={"current": (0, 45), "next": (33, 60)},
    )

    # Text still renders ONLY the current event's stable box (padded)...
    assert len(processor.render_calls) == 1
    pad = 0.015 * FH
    expected_render = (layout["current"].x1 - pad, layout["current"].y1 - pad,
                       layout["current"].x2 + pad, layout["current"].y2 + pad)
    assert (processor.render_calls[0].x1, processor.render_calls[0].y1,
            processor.render_calls[0].x2, processor.render_calls[0].y2) == pytest.approx(expected_render)
    # ...while suppression already covers the incoming caption too (padded).
    pad = 0.015 * FH
    suppressed = [(r.x1, r.y1, r.x2, r.y2) for r in processor.suppress_calls[-1]]
    assert len(suppressed) == 2
    assert suppressed[0] == pytest.approx((layout["current"].x1 - pad, layout["current"].y1 - pad,
                                          layout["current"].x2 + pad, layout["current"].y2 + pad))
    assert suppressed[1] == pytest.approx((layout["next"].x1 - pad, layout["next"].y1 - pad,
                                          layout["next"].x2 + pad, layout["next"].y2 + pad))


def test_failed_event_still_suppresses_source_pixels():
    # Real regression: TRANSLATION_FAILED events were absent from the render
    # map, so their original captions stayed fully readable — reported twice
    # as the "missing subtitle" leak. A failed event must contribute its
    # layout box to the suppression map (redaction, never fake text).
    processor = _LayoutRecordingProcessor()
    pipeline = VisualPipeline(subtitle_processor=processor)
    pipeline._region_planner = MagicMock()

    failed_ev = _event([_box(660, 940, 1260, 1020)], event_id="failed")
    failed_ev.start_time, failed_ev.end_time = 0.0, 1.0
    failed_ev.replacement_status = "TRANSLATION_FAILED"
    failed_ev.blur_applied = False
    failed_ev.overlay_applied = False
    failed_layout = _layout_box_for_event(failed_ev)

    request = VisualRequest(subtitle_enabled=True, subtitle_mode="translate_replace", subtitle_layout="in-place")
    pipeline._apply_subtitle_actions(
        object(), 10, request,
        {10: [_det(10, _box(660, 940, 1260, 1020))]},
        {10: []},  # No READY events on this frame.
        event_layout_boxes={"failed": failed_layout},
        event_render_spans={"failed": (0, 30)},
        suppress_only_by_frame={10: [failed_layout]},
    )

    assert processor.render_calls == []  # No text is faked.
    pad = 0.015 * FH
    assert len(processor.suppress_calls) == 1
    assert len(processor.suppress_calls[0]) == 1
    assert (processor.suppress_calls[0][0].x1, processor.suppress_calls[0][0].y1,
            processor.suppress_calls[0][0].x2, processor.suppress_calls[0][0].y2) == pytest.approx(
        (failed_layout.x1 - pad, failed_layout.y1 - pad, failed_layout.x2 + pad, failed_layout.y2 + pad)
    )


def test_duplicate_translation_in_two_lanes_renders_once():
    # Real regression: the same bilingual caption existed as two lane events
    # (upper CJK row + lower Latin row); both drew the identical Vietnamese
    # string at two positions, and the band jumped when the shorter one ended.
    processor = _LayoutRecordingProcessor()
    pipeline = VisualPipeline(subtitle_processor=processor)
    pipeline._region_planner = MagicMock()

    lower = _event([_box(844, 990, 1076, 1056)], event_id="lower")
    lower.start_time, lower.end_time = 12.07, 12.87
    lower.translated_text = "Ánh mắt dịu dàng"
    upper = _event([_box(850, 924, 1071, 985)], event_id="upper")
    upper.start_time, upper.end_time = 12.23, 12.63
    upper.translated_text = "Ánh mắt dịu dàng"

    layout = {e.event_id: _layout_box_for_event(e) for e in (lower, upper)}
    request = VisualRequest(subtitle_enabled=True, subtitle_mode="translate_replace", subtitle_layout="in-place")
    pipeline._apply_subtitle_actions(
        object(), 375, request,
        {375: [_det(375, _box(850, 924, 1071, 985))]},
        {375: [lower, upper]},
        event_layout_boxes=layout,
        event_render_spans={"lower": (362, 386), "upper": (367, 379)},
    )

    # The identical translation renders exactly once per frame.
    assert len(processor.render_calls) == 1
    kept_box = layout["lower"]  # longer event wins
    pad = 0.015 * FH
    expected = (kept_box.x1 - pad, kept_box.y1 - pad, kept_box.x2 + pad, kept_box.y2 + pad)
    assert (processor.render_calls[0].x1, processor.render_calls[0].y1,
            processor.render_calls[0].x2, processor.render_calls[0].y2) == pytest.approx(expected)
    # Suppression still covers BOTH rows while both events are mapped.
    assert len(processor.suppress_calls[-1]) == 2


def test_preroll_suppression_skipped_in_caption_free_gap():
    # Real case (21.2s transition): the incoming caption's pre-roll panel was
    # painted 3 frames before its text appeared, over a scene where the
    # original caption had already ended — a flashing empty black panel. With
    # raw per-frame detections supplied and nothing detected in the gap, the
    # pre-roll suppression must be skipped (only the current caption's box,
    # already in-span, is suppressed).
    processor = _LayoutRecordingProcessor()
    pipeline = VisualPipeline(subtitle_processor=processor)
    pipeline._region_planner = MagicMock()

    current_ev = _event([_box(660, 940, 1260, 1020)], event_id="current")
    current_ev.start_time, current_ev.end_time = 0.0, 1.0
    next_ev = _event([_box(300, 700, 900, 780)], event_id="next")
    next_ev.start_time, next_ev.end_time = 1.1, 2.0

    layout = {
        "current": _layout_box_for_event(current_ev),
        "next": _layout_box_for_event(next_ev),
    }
    request = VisualRequest(subtitle_enabled=True, subtitle_mode="translate_replace", subtitle_layout="in-place")
    pipeline._apply_subtitle_actions(
        object(), 32, request,
        {32: [_det(32, _box(660, 940, 1260, 1020))]},
        {32: [current_ev, next_ev]},
        event_layout_boxes=layout,
        event_render_spans={"current": (0, 30), "next": (33, 60)},
        raw_sub_dets_by_frame={32: [_det(32, _box(660, 940, 1260, 1020))]},  # current lane only
    )

    suppressed = [(r.x1, r.y1, r.x2, r.y2) for r in processor.suppress_calls[-1]]
    assert len(suppressed) == 1  # next caption's pre-roll panel skipped
    # Frame 32 sits in the gap: the previous caption ended (its post-roll box
    # is kept because a raw detection still shows in its lane), the next one
    # has not started — nothing renders.
    assert len(processor.render_calls) == 0


def test_preroll_suppression_kept_while_original_caption_still_detected():
    # Contiguous captions: the previous caption's text is still visible during
    # the incoming caption's pre-roll, so suppression must stay on.
    processor = _LayoutRecordingProcessor()
    pipeline = VisualPipeline(subtitle_processor=processor)
    pipeline._region_planner = MagicMock()

    current_ev = _event([_box(660, 940, 1260, 1020)], event_id="current")
    current_ev.start_time, current_ev.end_time = 0.0, 1.0
    next_ev = _event([_box(300, 700, 900, 780)], event_id="next")
    next_ev.start_time, next_ev.end_time = 1.1, 2.0

    layout = {
        "current": _layout_box_for_event(current_ev),
        "next": _layout_box_for_event(next_ev),
    }
    request = VisualRequest(subtitle_enabled=True, subtitle_mode="translate_replace", subtitle_layout="in-place")
    pipeline._apply_subtitle_actions(
        object(), 32, request,
        {32: [_det(32, _box(660, 940, 1260, 1020))]},
        {32: [current_ev, next_ev]},
        event_layout_boxes=layout,
        event_render_spans={"current": (0, 30), "next": (33, 60)},
        raw_sub_dets_by_frame={32: [
            _det(32, _box(660, 940, 1260, 1020)),
            _det(32, _box(300, 700, 900, 780)),  # incoming caption's source text still visible
        ]},
    )

    assert len(processor.suppress_calls[-1]) == 2


def test_near_duplicate_translations_render_once():
    # OCR verification pass recovered BOTH rows of a bilingual pair; their
    # Vietnamese differs by one word ('Điều thứ nhất...' vs 'Thứ nhất...').
    # Different lanes bypass select_render_events, so a single render here
    # proves the near-duplicate dedupe (Jaccard >= 0.6) did the work.
    processor = _LayoutRecordingProcessor()
    pipeline = VisualPipeline(subtitle_processor=processor)
    pipeline._region_planner = MagicMock()

    lower = _event([_box(844, 990, 1076, 1056)], event_id="lower")
    lower.start_time, lower.end_time = 6.93, 7.20
    lower.translated_text = "Điều thứ nhất: Phải dịu dàng"
    upper = _event([_box(850, 824, 1071, 885)], event_id="upper")
    upper.start_time, upper.end_time = 6.93, 7.20
    upper.translated_text = "Thứ nhất: Phải dịu dàng"

    layout = {e.event_id: _layout_box_for_event(e) for e in (lower, upper)}
    request = VisualRequest(subtitle_enabled=True, subtitle_mode="translate_replace", subtitle_layout="in-place")
    pipeline._apply_subtitle_actions(
        object(), 210, request,
        {210: [_det(210, _box(850, 924, 1071, 985))]},
        {210: [lower, upper]},
        event_layout_boxes=layout,
        event_render_spans={"lower": (207, 216), "upper": (207, 216)},
    )

    assert len(processor.render_calls) == 1


def test_distinct_translations_in_different_lanes_still_render_both():
    # Different captions sharing a few words (Jaccard 0.5 < 0.6) must NOT be
    # deduped: both lanes render their own caption.
    processor = _LayoutRecordingProcessor()
    pipeline = VisualPipeline(subtitle_processor=processor)
    pipeline._region_planner = MagicMock()

    first = _event([_box(844, 990, 1076, 1056)], event_id="first")
    first.start_time, first.end_time = 21.0, 23.0
    first.translated_text = "Điều thứ hai: Phải đúng mực"
    second = _event([_box(850, 824, 1071, 885)], event_id="second")
    second.start_time, second.end_time = 21.0, 23.0
    second.translated_text = "Thứ hai: Phải thật chừng mực"

    layout = {e.event_id: _layout_box_for_event(e) for e in (first, second)}
    request = VisualRequest(subtitle_enabled=True, subtitle_mode="translate_replace", subtitle_layout="in-place")
    pipeline._apply_subtitle_actions(
        object(), 650, request,
        {650: [_det(650, _box(850, 924, 1071, 985))]},
        {650: [first, second]},
        event_layout_boxes=layout,
        event_render_spans={"first": (630, 690), "second": (630, 690)},
    )

    assert len(processor.render_calls) == 2


def test_twin_rows_suppress_union_plate_through_longer_span():
    # Real case (12.6-12.9s): the Chinese row's event ended at 12.63 while
    # '眼神柔和' stayed on screen until 12.87 — per-event suppression let the
    # whole row leak above the lower plate. Twin unification gives the kept
    # event the union plate and union suppress span, so the frame at 12.8s
    # (only the kept event mapped) still paints a plate covering BOTH rows.
    from src.features.visual.pipeline import _unify_twin_events

    processor = _LayoutRecordingProcessor()
    pipeline = VisualPipeline(subtitle_processor=processor)
    pipeline._region_planner = MagicMock()

    lower = _event([_box(844, 990, 1076, 1056)], event_id="lower")
    lower.start_time, lower.end_time = 12.07, 12.87
    lower.translated_text = "Ánh mắt dịu dàng"
    upper = _event([_box(850, 924, 1071, 985)], event_id="upper")
    upper.start_time, upper.end_time = 12.23, 12.63
    upper.translated_text = "Ánh mắt dịu dàng"

    events = [lower, upper]
    layout = {e.event_id: _layout_box_for_event(e) for e in events}
    plates = {e.event_id: _layout_boxes_for_event(e)[1] for e in events}
    fps = 29.97
    suppress_spans, twin_members = _unify_twin_events(events, plates, fps)

    # Union suppress span covers both rows' lifetimes.
    assert suppress_spans["lower"] == (362, 386)
    # Union plate covers the upper row too.
    assert plates["lower"].y1 <= 924

    request = VisualRequest(subtitle_enabled=True, subtitle_mode="translate_replace", subtitle_layout="in-place")
    pipeline._apply_subtitle_actions(
        object(), 384, request,  # 12.81s: upper's own span ended at 12.63
        {384: [_det(384, _box(844, 990, 1076, 1056))]},
        {384: [lower]},
        event_layout_boxes=layout,
        event_render_spans={"lower": (362, 386), "upper": (367, 379)},
        event_plate_boxes=plates,
        suppress_spans=suppress_spans,
    )

    pad = 0.015 * FH
    assert len(processor.suppress_calls[-1]) == 1
    painted = processor.suppress_calls[-1][0]
    assert painted.y1 <= 924 - pad  # upper row's glyphs covered
    assert painted.y2 >= 1056 + pad


def test_stacked_distinct_rows_render_at_their_own_boxes():
    # One Piece case: a persistent title row above a rotating content row.
    # Different meanings sharing one lane must BOTH render, each anchored to
    # its own stable box — not collapse into one overlay that drops the
    # other row's text.
    processor = _LayoutRecordingProcessor()
    pipeline = VisualPipeline(subtitle_processor=processor)
    pipeline._region_planner = MagicMock()

    title = _event([_box(688, 759, 1260, 849)], event_id="title")
    title.start_time, title.end_time = 21.8, 26.8
    title.translated_text = "Đô đốc Tổng bộ Hải quân-"
    content = _event([_box(713, 860, 1206, 1043)], event_id="content")
    content.start_time, content.end_time = 21.8, 24.7
    content.translated_text = "Aokiji\nngười đang trấn giữ phía dưới đài hành hình là"

    events = [title, content]
    layout = {e.event_id: _layout_box_for_event(e) for e in events}
    pipeline._render_stable_layout(
        object(), 690, events, layout,
        event_render_spans={"title": (654, 804), "content": (654, 741)},
        hide_source=False,
    )

    assert len(processor.render_calls) == 2
    title_box = min(processor.render_calls, key=lambda r: r.y1)
    content_box = max(processor.render_calls, key=lambda r: r.y2)
    assert title_box.y1 <= 759 and title_box.y2 <= 880
    assert content_box.y2 >= 1043


def test_twin_rows_with_drifted_translation_collapse_to_one():
    # Bilingual rows of ONE caption: their per-row translations drift more
    # than unrelated captions would (Jaccard 0.5), so twins collapse at the
    # looser 0.4 threshold — the same meaning never renders twice. Without
    # twin membership the same pair would render at both boxes.
    from src.features.visual.pipeline import _unify_twin_events

    processor = _LayoutRecordingProcessor()
    pipeline = VisualPipeline(subtitle_processor=processor)
    pipeline._region_planner = MagicMock()

    lower = _event([_box(844, 990, 1076, 1056)], event_id="lower")
    lower.start_time, lower.end_time = 12.07, 12.87
    lower.translated_text = "Hãy dịu dàng"
    upper = _event([_box(850, 924, 1071, 985)], event_id="upper")
    upper.start_time, upper.end_time = 12.23, 12.63
    upper.translated_text = "Hãy dịu dàng với mọi người"

    events = [lower, upper]
    layout = {e.event_id: _layout_box_for_event(e) for e in events}
    plates = {e.event_id: _layout_boxes_for_event(e)[1] for e in events}
    suppress_spans, twin_members = _unify_twin_events(events, plates, 29.97)
    assert twin_members == {"lower": ["upper"]}

    pipeline._render_stable_layout(
        object(), 380, events, layout,
        event_render_spans={"lower": (362, 386), "upper": (367, 379)},
        hide_source=False,
        suppress_spans=suppress_spans,
        twin_members=twin_members,
    )
    assert len(processor.render_calls) == 1
