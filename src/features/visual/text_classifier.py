"""
Generic-text detection classifier.

The generic YOLO text model (single class "text") does not distinguish
subtitle / watermark / logo / other burn-in text. This module re-labels
generic detections into the canonical classes using per-track geometry and
temporal properties so the standard class-aware tracking and routing
downstream stay unchanged.

Policy-only module: imports core models and the feature-layer tracker; no
infrastructure dependencies.
"""

from __future__ import annotations

import logging
import statistics
from dataclasses import replace
from typing import Dict, List

from src.core.contracts import DetectorPort
from src.core.models.bbox import BBox
from src.core.models.detection import CANONICAL_CLASS_NAMES, Detection
from src.features.visual.tracker import ClassAwareTracker

logger = logging.getLogger(__name__)

#: Class name emitted by the generic single-class text model.
GENERIC_TEXT_CLASS_NAME = "text"

CLASS_IDS: Dict[str, int] = {name: cid for cid, name in CANONICAL_CLASS_NAMES.items()}

# Geometry signals, all as fractions of the frame. Subtitle requires the
# centered signal plus at least one of bottom-band / line-like, so corner
# watermarks never fall into the subtitle class.
BOTTOM_BAND_CENTER_Y = 0.62       # mean center-y >= this counts as bottom placement
MAX_SUBTITLE_WIDTH_RATIO = 0.80   # centered line width cap
MAX_SUBTITLE_HEIGHT_RATIO = 0.12  # line-like height cap
#: Bottom-band captions are at most a tall two-line block; anything higher is
#: a hallucinated region box (dark-textured scenery), not a caption.
MAX_BAND_CAPTION_HEIGHT_RATIO = 0.20
MIN_LINE_ASPECT = 3.0             # width/height of a text line
# The bottom-band path admits off-center captions (cx 0.15-0.85); the
# line-like path (top-of-frame captions) demands strict centering so a
# corner handle at cx~0.83 — or a narrow top-left account banner at
# cx~0.26 — cannot enter the subtitle class.
CENTER_X_WINDOW = (0.15, 0.85)    # horizontally centered window, bottom-band path
STRICT_CENTER_X_WINDOW = (0.30, 0.70)  # line-like path (real captions are centered)

# Temporal signals. Thresholds are frame counts, deliberately not seconds: the
# sampled detection stream carries no trustworthy fps, and frame spans are
# stride-invariant because track spans and the video span come from the same
# sampling.
MIN_STATIC_FRAMES = 75
MIN_STATIC_SPAN_RATIO = 0.30
MAX_STATIC_JITTER = 0.01          # normalized center std for a static track
MAX_STATIC_SIZE_JITTER = 0.02     # normalized width/height std
CORNER_MARGIN = 0.22              # center outside this margin counts as corner
SMALL_HEIGHT_RATIO = 0.08         # small box height for watermark
#: A second caption row the 4-class model labeled watermark. It must be a
#: wide line sitting just under a subtitle; a corner mark is too narrow.
CAPTION_ROW_MIN_WIDTH_RATIO = 0.18
CAPTION_ROW_MAX_GAP_RATIO = 0.12
CAPTION_ROW_OVERLAP_RATIO = 0.45
#: A watermark box on the same glyphs as a subtitle, up to about one second
#: before the model also emits the subtitle class.
SAME_LINE_FRAME_WINDOW = 30
SAME_LINE_VERTICAL_OVERLAP = 0.50
#: A short bottom line (for example 大将们) is still a caption when it sits
#: on the same glyphs as a subtitle. A face credit is thin, narrow, and
#: above the caption band.
BOTTOM_LINE_CENTER_Y = 0.72
CREDIT_MAX_WIDTH_RATIO = 0.15
CREDIT_MAX_HEIGHT_RATIO = 0.06
CREDIT_MAX_CENTER_Y_RATIO = 0.70


class TextClassifyingDetector(DetectorPort):
    """DetectorPort wrapper that re-labels generic "text" detections into the
    canonical burn-in classes before tracking and class routing."""

    def __init__(self, inner: DetectorPort):
        self._inner = inner

    def detect_video(self, video_path: str, stride: int = 1) -> List[Detection]:
        return classify_generic_text_detections(
            self._inner.detect_video(video_path, stride=stride)
        )


def promote_subtitle_rows_labeled_watermark(detections: List[Detection]) -> List[Detection]:
    """Re-label a watermark that is really a caption line.

    The 4-class model sometimes calls a caption watermark: the second line
    under a subtitle, or the same line for up to a second before it also
    emits the subtitle class. A wide second row qualifies on width. A short
    bottom line qualifies when it shares glyphs with a nearby subtitle.
    A corner mark stays a watermark.
    """
    by_frame: Dict[int, List[Detection]] = {}
    for det in detections:
        by_frame.setdefault(det.frame_index, []).append(det)
    promoted: List[Detection] = []
    for det in detections:
        if det.class_name != "watermark":
            promoted.append(det)
            continue
        frame_w = float(det.bbox.frame_width)
        frame_h = float(det.bbox.frame_height)
        wide = det.bbox.width >= CAPTION_ROW_MIN_WIDTH_RATIO * frame_w
        center_y = (det.bbox.y1 + det.bbox.y2) / 2.0
        bottom_line = center_y >= BOTTOM_LINE_CENTER_Y * frame_h
        if not wide and not bottom_line:
            promoted.append(det)
            continue
        matched = False
        for other in _nearby_subtitles(by_frame, det.frame_index):
            if wide and _is_second_caption_row(det, other, frame_h):
                matched = True
                break
            if (wide or bottom_line) and _is_same_caption_line(det, other):
                matched = True
                break
        if not matched:
            promoted.append(det)
            continue
        promoted.append(replace(det, class_id=CLASS_IDS["subtitle"], class_name="subtitle"))
    return promoted


def demote_upper_credits_labeled_subtitle(detections: List[Detection]) -> List[Detection]:
    """Re-label a thin subtitle box above the caption band as a watermark.

    Face credits such as 西瓜君AI制作 are a narrow, short box on the face.
    A name on the costume is taller or sits in the lower band and stays a
    subtitle, so the translated name is still drawn.
    """
    demoted: List[Detection] = []
    for det in detections:
        if det.class_name != "subtitle":
            demoted.append(det)
            continue
        frame_w = float(det.bbox.frame_width)
        frame_h = float(det.bbox.frame_height)
        center_y = (det.bbox.y1 + det.bbox.y2) / 2.0
        thin = det.bbox.height < CREDIT_MAX_HEIGHT_RATIO * frame_h
        narrow = det.bbox.width < CREDIT_MAX_WIDTH_RATIO * frame_w
        high = center_y < CREDIT_MAX_CENTER_Y_RATIO * frame_h
        if thin and narrow and high:
            demoted.append(replace(det, class_id=CLASS_IDS["watermark"], class_name="watermark"))
            continue
        demoted.append(det)
    return demoted


def _nearby_subtitles(by_frame: Dict[int, List[Detection]], frame_index: int) -> List[Detection]:
    found: List[Detection] = []
    for frame in range(frame_index - SAME_LINE_FRAME_WINDOW, frame_index + SAME_LINE_FRAME_WINDOW + 1):
        for other in by_frame.get(frame, []):
            if other.class_name == "subtitle":
                found.append(other)
    return found


def _horizontal_overlap_ratio(a: BBox, b: BBox) -> float:
    overlap = min(a.x2, b.x2) - max(a.x1, b.x1)
    return overlap / max(1.0, min(a.width, b.width))


def _is_second_caption_row(watermark: Detection, subtitle: Detection, frame_h: float) -> bool:
    """True when the watermark is the next line under this subtitle."""
    if watermark.frame_index != subtitle.frame_index:
        return False
    gap = watermark.bbox.y1 - subtitle.bbox.y2
    if gap < -0.03 * frame_h or gap > CAPTION_ROW_MAX_GAP_RATIO * frame_h:
        return False
    return _horizontal_overlap_ratio(watermark.bbox, subtitle.bbox) >= CAPTION_ROW_OVERLAP_RATIO


def _is_same_caption_line(watermark: Detection, subtitle: Detection) -> bool:
    """True when both boxes sit on one row of glyphs."""
    overlap_y = min(watermark.bbox.y2, subtitle.bbox.y2) - max(watermark.bbox.y1, subtitle.bbox.y1)
    shorter = min(watermark.bbox.height, subtitle.bbox.height)
    if overlap_y < SAME_LINE_VERTICAL_OVERLAP * max(1.0, shorter):
        return False
    return _horizontal_overlap_ratio(watermark.bbox, subtitle.bbox) >= CAPTION_ROW_OVERLAP_RATIO


def classify_generic_text_detections(detections: List[Detection]) -> List[Detection]:
    """Re-label generic "text" detections into canonical classes.

    Non-generic detections pass through untouched; relabeled detections are
    new copies (caller's objects are never mutated).
    """
    generic = [d for d in detections if d.class_name == GENERIC_TEXT_CLASS_NAME]
    others = [d for d in detections if d.class_name != GENERIC_TEXT_CLASS_NAME]
    if not generic:
        return list(detections)

    tracker = ClassAwareTracker()
    tracks = tracker.track(generic)
    video_frame_span = _video_frame_span(tracks)

    relabeled: List[Detection] = []
    for track in tracks:
        class_id = _classify_track(track, video_frame_span=video_frame_span)
        for det in track.detections:
            relabeled.append(
                Detection(
                    class_id=class_id,
                    class_name=CANONICAL_CLASS_NAMES[class_id],
                    confidence=det.confidence,
                    bbox=det.bbox,
                    frame_index=det.frame_index,
                    timestamp_seconds=det.timestamp_seconds,
                    candidate_id=det.candidate_id,
                )
            )
    return sorted([*others, *relabeled], key=lambda d: (d.frame_index, -d.confidence))


def _video_frame_span(tracks) -> int:
    frames = [frame for track in tracks for frame in (track.start_frame, track.end_frame)]
    return max(frames) - min(frames) if frames else 0


def _classify_track(track, video_frame_span: int) -> int:
    detections = sorted(track.detections, key=lambda d: d.frame_index)
    boxes = [d.bbox for d in detections]
    frame_w = float(boxes[0].frame_width or 1)
    frame_h = float(boxes[0].frame_height or 1)

    # Subtitle placement is judged on the per-frame UNION of the track's
    # boxes, not on individual chips: a wide banner detected as several thin
    # centered chips passes the line-like test chip-by-chip even though the
    # visual line it forms is far too wide/tall for a caption.
    union_boxes = _per_frame_union_boxes(detections)
    widths = [b.x2 - b.x1 for b in union_boxes]
    heights = [b.y2 - b.y1 for b in union_boxes]
    centers_x = [(b.x1 + b.x2) / 2 / frame_w for b in union_boxes]
    centers_y = [(b.y1 + b.y2) / 2 / frame_h for b in union_boxes]

    mean_w = sum(widths) / len(widths)
    mean_h = sum(heights) / len(heights)
    mean_cx = sum(centers_x) / len(centers_x)
    mean_cy = sum(centers_y) / len(centers_y)

    bottom_band = mean_cy >= BOTTOM_BAND_CENTER_Y
    band_height_ok = mean_h / frame_h <= MAX_BAND_CAPTION_HEIGHT_RATIO
    centered = (
        CENTER_X_WINDOW[0] <= mean_cx <= CENTER_X_WINDOW[1]
        and mean_w / frame_w <= MAX_SUBTITLE_WIDTH_RATIO
    )
    strictly_centered = (
        STRICT_CENTER_X_WINDOW[0] <= mean_cx <= STRICT_CENTER_X_WINDOW[1]
        and mean_w / frame_w <= MAX_SUBTITLE_WIDTH_RATIO
    )
    line_like = (
        mean_h / frame_h <= MAX_SUBTITLE_HEIGHT_RATIO
        and mean_w / max(mean_h, 1.0) >= MIN_LINE_ASPECT
    )

    # 1. Geometry-first subtitle check: an on-screen caption is placed like a
    #    caption regardless of how long it stays, so short videos never push
    #    long captions into the static classes. Bottom-band captions may sit
    #    off-center but stay caption-height (a tall box over dark scenery is
    #    a detector hallucination, and merging one into a caption track makes
    #    the rendered panel jump size mid-video); a top caption qualifies only
    #    when strictly centered, so corner handles never leak into the class.
    if (bottom_band and centered and band_height_ok) or (line_like and strictly_centered):
        return CLASS_IDS["subtitle"]

    # 2. Static overlay classes (watermark / logo). Multi-piece overlays (e.g.
    #    a handle built from several text chips) mix positions inside one
    #    track; evaluate the whole track first, then its modal position
    #    cluster — the stable piece — when the mixture fails the jitter gate.
    #    Static and cluster rules keep the RAW per-chip geometry they were
    #    tuned on.
    frame_span = track.end_frame - track.start_frame
    raw_widths = [b.x2 - b.x1 for b in boxes]
    raw_heights = [b.y2 - b.y1 for b in boxes]
    raw_centers_x = [(b.x1 + b.x2) / 2 / frame_w for b in boxes]
    raw_centers_y = [(b.y1 + b.y2) / 2 / frame_h for b in boxes]
    if _is_static_layout(
        frame_span, video_frame_span, raw_centers_x, raw_centers_y,
        raw_widths, raw_heights, frame_w, frame_h,
    ):
        return _static_class(
            sum(raw_centers_x) / len(raw_centers_x),
            sum(raw_centers_y) / len(raw_centers_y),
            sum(raw_heights) / len(raw_heights),
            frame_h,
        )

    cluster = _modal_position_cluster(detections, frame_w, frame_h)
    if cluster is not None:
        c_cx, c_cy, c_w, c_h, c_frames = cluster
        # The cluster's center/size are means, so its jitter is zero by
        # construction; only the frame-coverage gates remain.
        if video_frame_span > 0 and c_frames >= max(
            MIN_STATIC_FRAMES, MIN_STATIC_SPAN_RATIO * video_frame_span
        ):
            return _static_class(c_cx, c_cy, c_h * frame_h, frame_h)

    # 3. Transient text outside the caption placement is burn-in text.
    return CLASS_IDS["other_burn_in_text"]


def _per_frame_union_boxes(detections) -> List:
    """Union of all track boxes sharing a frame_index, ordered by frame.

    A frame normally holds one grouped detection from the tracker; the union
    makes the shape metric describe the whole visual line when several chips
    of the same overlay share a frame.
    """
    by_frame: Dict[int, list] = {}
    for det in detections:
        by_frame.setdefault(det.frame_index, []).append(det.bbox)
    unions = []
    for frame_index in sorted(by_frame):
        group = by_frame[frame_index]
        if len(group) == 1:
            unions.append(group[0])
        else:
            unions.append(
                type(group[0])(
                    x1=min(b.x1 for b in group),
                    y1=min(b.y1 for b in group),
                    x2=max(b.x2 for b in group),
                    y2=max(b.y2 for b in group),
                    frame_width=group[0].frame_width,
                    frame_height=group[0].frame_height,
                )
            )
    return unions


def _is_static_layout(frame_span: int, video_frame_span: int, centers_x, centers_y, widths, heights, frame_w: float, frame_h: float) -> bool:
    cx_jitter = statistics.pstdev(centers_x) if len(centers_x) > 1 else 0.0
    cy_jitter = statistics.pstdev(centers_y) if len(centers_y) > 1 else 0.0
    width_jitter = (statistics.pstdev(widths) / frame_w) if len(widths) > 1 else 0.0
    height_jitter = (statistics.pstdev(heights) / frame_h) if len(heights) > 1 else 0.0
    return bool(
        frame_span >= MIN_STATIC_FRAMES
        and video_frame_span > 0
        and frame_span >= MIN_STATIC_SPAN_RATIO * video_frame_span
        and cx_jitter <= MAX_STATIC_JITTER
        and cy_jitter <= MAX_STATIC_JITTER
        and width_jitter <= MAX_STATIC_SIZE_JITTER
        and height_jitter <= MAX_STATIC_SIZE_JITTER
    )


def _static_class(mean_cx: float, mean_cy: float, mean_h: float, frame_h: float) -> int:
    corner = (
        mean_cx <= CORNER_MARGIN
        or mean_cx >= 1.0 - CORNER_MARGIN
        or mean_cy <= CORNER_MARGIN
        or mean_cy >= 1.0 - CORNER_MARGIN
    )
    small = mean_h / frame_h <= SMALL_HEIGHT_RATIO
    if corner and small:
        return CLASS_IDS["watermark"]
    return CLASS_IDS["logo"]


def _modal_position_cluster(detections, frame_w: float, frame_h: float):
    """Find the largest position bucket of a track's detections.

    Returns (mean_cx, mean_cy, mean_w, mean_h, distinct_frames) for the modal
    bucket, or None when no bucket holds at least a quarter of the track.
    """
    buckets: Dict[tuple, list] = {}
    for det in detections:
        b = det.bbox
        cx = (b.x1 + b.x2) / 2 / frame_w
        cy = (b.y1 + b.y2) / 2 / frame_h
        key = (int(cx * 20), int(cy * 20))
        buckets.setdefault(key, []).append(det)
    if not buckets:
        return None
    modal = max(buckets.values(), key=len)
    if len(modal) * 4 < len(detections):
        return None
    c_cx = sum((d.bbox.x1 + d.bbox.x2) / 2 / frame_w for d in modal) / len(modal)
    c_cy = sum((d.bbox.y1 + d.bbox.y2) / 2 / frame_h for d in modal) / len(modal)
    c_w = sum(d.bbox.x2 - d.bbox.x1 for d in modal) / len(modal) / frame_w
    c_h = sum(d.bbox.y2 - d.bbox.y1 for d in modal) / len(modal) / frame_h
    c_frames = len({d.frame_index for d in modal})
    return c_cx, c_cy, c_w, c_h, c_frames
