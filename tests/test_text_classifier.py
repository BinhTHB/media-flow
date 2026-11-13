"""Unit tests for the generic-text detection classifier.

The generic YOLO model emits a single class "text"; the classifier must
re-label detections into the canonical burn-in classes using per-track
geometry and temporal properties so standard class routing keeps working.
"""

from typing import List

from src.core.models.bbox import BBox
from src.core.models.detection import Detection
from src.features.visual.text_classifier import (
    TextClassifyingDetector,
    classify_generic_text_detections,
    demote_upper_credits_labeled_subtitle,
    promote_subtitle_rows_labeled_watermark,
)

FW, FH = 1920, 1080
FPS = 30.0


def _det(frame_index: int, x1: float, y1: float, x2: float, y2: float, class_name: str = "text") -> Detection:
    return Detection(
        class_id=0,
        class_name=class_name,
        confidence=0.9,
        bbox=BBox(x1=x1, y1=y1, x2=x2, y2=y2, frame_width=FW, frame_height=FH),
        frame_index=frame_index,
        timestamp_seconds=frame_index / FPS,
    )


def _run(detections: List[Detection]) -> List[str]:
    return [d.class_name for d in classify_generic_text_detections(detections)]


def _contiguous(frames: range, box) -> List[Detection]:
    return [_det(i, *box) for i in frames]


def test_second_caption_row_labeled_watermark_becomes_subtitle():
    # Real frame at 9s: the name is subtitle, the line under it is watermark.
    name = _det(270, 541, 807, 1382, 897, class_name="subtitle")
    line = _det(270, 681, 987, 1265, 1042, class_name="watermark")
    out = promote_subtitle_rows_labeled_watermark([name, line])
    assert [d.class_name for d in out] == ["subtitle", "subtitle"]


def test_narrow_corner_mark_stays_watermark():
    subtitle = _det(10, 600, 900, 1300, 980, class_name="subtitle")
    mark = _det(10, 1500, 20, 1880, 70, class_name="watermark")
    out = promote_subtitle_rows_labeled_watermark([subtitle, mark])
    assert out[1].class_name == "watermark"


def test_watermark_far_below_a_subtitle_stays_watermark():
    subtitle = _det(10, 600, 400, 1300, 480, class_name="subtitle")
    mark = _det(10, 600, 980, 1300, 1040, class_name="watermark")
    out = promote_subtitle_rows_labeled_watermark([subtitle, mark])
    assert out[1].class_name == "watermark"


def test_opening_line_labeled_watermark_joins_the_subtitle_track():
    # The first second of douyin_new_30s: the line is on screen at frame 5
    # labeled watermark, and the subtitle class only appears at frame 22.
    from src.features.visual.tracker import ClassAwareTracker

    dets = [
        _det(5, 570, 982, 1365, 1046, class_name="watermark"),
        _det(12, 570, 981, 1369, 1047, class_name="watermark"),
    ]
    dets += [_det(frame, 596, 987, 1253, 1045, class_name="subtitle") for frame in range(22, 40)]
    promoted = promote_subtitle_rows_labeled_watermark(dets)
    assert all(det.class_name == "subtitle" for det in promoted)
    tracks = [t for t in ClassAwareTracker().track(promoted) if t.class_name == "subtitle"]
    assert len(tracks) == 1
    assert tracks[0].start_frame == 5


def test_two_long_captions_in_the_same_place_stay_separate_tracks():
    from src.features.visual.tracker import ClassAwareTracker

    dets = [_det(frame, 705, 809, 1218, 892, class_name="subtitle") for frame in range(0, 60)]
    dets += [_det(frame, 705, 810, 1214, 896, class_name="subtitle") for frame in range(70, 120)]
    tracks = [t for t in ClassAwareTracker().track(dets) if t.class_name == "subtitle"]
    assert len(tracks) == 2


def test_narrow_bottom_line_labeled_watermark_becomes_subtitle():
    # 大将们 at 19.8s is ~155px wide, so the 18% width gate rejects it.
    # It shares the bottom row with the subtitle class 0.7s later.
    early = _det(594, 881, 986, 1036, 1047, class_name="watermark")
    later = _det(615, 881, 986, 1036, 1047, class_name="subtitle")
    out = promote_subtitle_rows_labeled_watermark([early, later])
    assert out[0].class_name == "subtitle"


def test_face_credit_labeled_subtitle_becomes_watermark():
    # 西瓜君AI制作 sits on the chin: thin, narrow, above the caption band.
    credit = _det(594, 675, 481, 836, 518, class_name="subtitle")
    name = _det(655, 888, 860, 1064, 946, class_name="subtitle")
    out = demote_upper_credits_labeled_subtitle([credit, name])
    assert [d.class_name for d in out] == ["watermark", "subtitle"]


def test_watermark_more_than_a_second_before_a_subtitle_stays_watermark():
    early = _det(0, 570, 982, 1365, 1046, class_name="watermark")
    later = _det(40, 596, 987, 1253, 1045, class_name="subtitle")
    out = promote_subtitle_rows_labeled_watermark([early, later])
    assert out[0].class_name == "watermark"


def test_bottom_caption_four_seconds_in_ten_second_video_is_subtitle():
    # 10s video = 300 frames; a rock-stable 4s bottom caption spans 120 frames,
    # above the static frame threshold. Geometry-first ordering must classify
    # it as subtitle, never logo.
    detections = _contiguous(range(0, 120), (660, 940, 1260, 1020))
    assert set(_run(detections)) == {"subtitle"}


def test_short_bottom_caption_is_subtitle():
    detections = _contiguous(range(0, 30), (660, 940, 1260, 1020))
    assert set(_run(detections)) == {"subtitle"}


def test_static_small_corner_box_is_watermark():
    # Top-right account handle: small, static, hugging the corner.
    detections = _contiguous(range(0, 200), (1500, 30, 1880, 75))
    assert set(_run(detections)) == {"watermark"}


def test_bottom_right_corner_watermark_is_not_subtitle():
    # Centered is mandatory for subtitle; a small bottom-right overlay must
    # not be swallowed by the bottom-band signal.
    detections = _contiguous(range(0, 200), (1400, 980, 1900, 1030))
    assert set(_run(detections)) == {"watermark"}


def test_real_world_top_corner_handle_is_watermark_not_subtitle():
    # Real regression from the 30s dubbing video: the Douyin handle
    # (bbox ~[1297, 25, 1877, 74], cx≈0.826, full-video span, conf 0.87) was
    # classified subtitle because the wide center window (0.15-0.85) admitted
    # it and it is line-like. The line-like path must demand strict centering.
    detections = _contiguous(range(0, 300), (1297, 25, 1877, 74))
    assert set(_run(detections)) == {"watermark"}


def test_strictly_centered_top_caption_is_subtitle():
    # A genuinely centered top-of-frame caption still qualifies via the
    # line-like path even though it never enters the bottom band.
    detections = _contiguous(range(0, 40), (760, 60, 1160, 110))
    assert set(_run(detections)) == {"subtitle"}


def test_static_large_center_box_is_logo():
    detections = _contiguous(range(0, 200), (660, 380, 1260, 700))
    assert set(_run(detections)) == {"logo"}


def test_transient_offcenter_text_is_other_burn_in_text():
    # Off-center transient text fails the mandatory centered signal, so it can
    # never be promoted to subtitle. (A centered line-like caption in the
    # middle of the frame IS subtitle-shaped by design: top-centered captions
    # exist, and geometry alone cannot tell them apart — accepted v1 tradeoff.)
    detections = _contiguous(range(0, 20), (20, 400, 500, 470))
    assert set(_run(detections)) == {"other_burn_in_text"}


def test_out_and_back_track_is_not_static():
    # The box leaves its position mid-track and returns: endpoint IoU stays
    # high but center jitter disqualifies the static classes.
    detections = []
    for i in range(0, 60):
        detections.append(_det(i, 660, 940, 1260, 1020))
    for i in range(60, 120):
        detections.append(_det(i, 100, 200, 700, 280))
    for i in range(120, 180):
        detections.append(_det(i, 660, 940, 1260, 1020))
    names = set(_run(detections))
    assert "logo" not in names
    assert "watermark" not in names


def test_mixed_tracks_get_independent_labels():
    detections = [
        *_contiguous(range(0, 200), (1500, 30, 1880, 75)),   # corner watermark
        *_contiguous(range(0, 30), (660, 940, 1260, 1020)),  # bottom caption
        *_contiguous(range(0, 15), (20, 400, 500, 470)),     # transient off-center text
    ]
    counts: dict[str, int] = {}
    for name in _run(detections):
        counts[name] = counts.get(name, 0) + 1
    assert counts == {"watermark": 200, "subtitle": 30, "other_burn_in_text": 15}


def test_wrapper_relabels_both_class_id_and_class_name():
    class _GenericStub:
        def detect_video(self, video_path: str, stride: int = 1) -> List[Detection]:
            return _contiguous(range(0, 40), (660, 940, 1260, 1020))

    relabeled = TextClassifyingDetector(_GenericStub()).detect_video("unused.mp4")

    assert relabeled
    assert {d.class_name for d in relabeled} == {"subtitle"}
    assert {d.class_id for d in relabeled} == {0}
    # Geometry, confidence, and frame metadata are preserved.
    first = relabeled[0]
    assert (first.bbox.x1, first.bbox.y1, first.bbox.x2, first.bbox.y2) == (660, 940, 1260, 1020)
    assert first.confidence == 0.9
    assert first.frame_index == 0


def test_non_generic_detections_pass_through_untouched():
    detections = [_det(0, 660, 940, 1260, 1020, class_name="subtitle")]
    result = classify_generic_text_detections(detections)
    assert result is not detections
    assert len(result) == 1
    assert result[0].class_name == "subtitle"
    assert result[0].class_id == 0


def test_missing_timestamps_still_classify():
    # Frame-based thresholds must not depend on trustworthy timestamps.
    detections = []
    for i in range(0, 30):
        det = _det(i, 660, 940, 1260, 1020)
        det.timestamp_seconds = 0.0
        detections.append(det)
    assert set(_run(detections)) == {"subtitle"}


def test_wide_banner_of_thin_centered_chips_is_not_subtitle():
    # Real case (T141): a wide in-scene banner (94% of frame width, 28% tall)
    # was detected as several thin centered chips; chip-by-chip each passed
    # the line-like test and the banner rendered as a mid-screen "subtitle".
    # The subtitle rule must judge the per-frame UNION of the chips.
    detections = []
    for i in range(0, 25):  # 25 frames: transient, far below static gates
        detections.append(_det(i, 108, 301, 700, 609))
        detections.append(_det(i, 700, 301, 1300, 609))
        detections.append(_det(i, 1300, 301, 1914, 609))
    assert set(_run(detections)) == {"other_burn_in_text"}


def test_narrow_top_left_account_banner_is_not_subtitle():
    # Real case (T122): a top-left account banner at cx~0.26 slipped through
    # the old line-like window (0.25-0.75). Strict centering must reject it.
    detections = _contiguous(range(0, 25), (322, 310, 657, 371))
    assert set(_run(detections)) != {"subtitle"}


def test_tall_bottom_region_box_is_not_subtitle():
    # Real case (second 7): the detector hallucinated a 420x280 box over dark
    # scenery (conf 0.23-0.42, height 26% of frame). It passed the old
    # bottom-band path (which had no height cap), got merged into the caption
    # track, and made the rendered panel jump size mid-video. Caption-height
    # cap on the bottom band rejects it; real two-line blocks (~12%) pass.
    detections = _contiguous(range(0, 25), (96, 801, 516, 1080))
    assert set(_run(detections)) != {"subtitle"}


def test_two_line_bottom_caption_still_subtitle():
    detections = _contiguous(range(0, 30), (765, 922, 1169, 1054))
    assert set(_run(detections)) == {"subtitle"}
