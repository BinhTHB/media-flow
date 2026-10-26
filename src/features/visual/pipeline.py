"""
Independent Visual Pipeline

Executes YOLO 4-class detection, Class-Aware Tracking, Class Routing, Subtitle/Logo/Watermark/Other Processors,
and Video Rendering.
Independent from AudioPipeline: zero audio/ASR/TTS imports.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.core.contracts import DetectorPort, SceneBoundaryPort, VideoUtilPort
from src.core.models.bbox import BBox
from src.features.visual.subtitle.region_planner import SubtitleRegionPlanner, same_lane, union_boxes
from src.core.models.detection import Detection
from src.core.models.subtitle_event import SubtitleEvent
from src.features.visual.processors.logo import LogoProcessor
from src.features.visual.processors.other_text import OtherTextProcessor
from src.features.visual.processors.watermark import WatermarkProcessor
from src.features.visual.subtitle.processor import SubtitleProcessor
from src.features.visual.tracker import ClassAwareTracker

logger = logging.getLogger(__name__)

VALID_SUBTITLE_MODES = {"translate_replace", "blur", "preserve"}

# Layout geometry is intentionally per-event, not per-frame: word-level
# detections jitter between frames, so a caption must be drawn from ONE box
# computed once for the whole event. Suppression stays per-frame.
LAYOUT_BOX_PAD_RATIO = 0.015
LAYOUT_BOX_PAD_MIN_PX = 8.0
LAYOUT_SPREAD_WARN_RATIO = 0.05
#: Suppression bucket grid (5% of the frame) used to split a two-position
#: trajectory into its dominant cluster and the leftover positions.
LAYOUT_CLUSTER_BUCKET = 20
#: Band normalization (see _apply_band_normalization): bottom-edge capture
#: window and the maximum x-center correction for near-band captions.
BAND_Y2_TOLERANCE_RATIO = 0.06
BAND_X_CENTER_MAX_SHIFT_RATIO = 0.08


def _stable_layout_box(trajectory_boxes: List[BBox], frame_width: int, frame_height: int) -> BBox:
    """Average the trajectory boxes per coordinate and clamp into the frame."""
    count = max(1, len(trajectory_boxes))
    mean_x1 = sum(b.x1 for b in trajectory_boxes) / count
    mean_y1 = sum(b.y1 for b in trajectory_boxes) / count
    mean_x2 = sum(b.x2 for b in trajectory_boxes) / count
    mean_y2 = sum(b.y2 for b in trajectory_boxes) / count
    return BBox.clamped(
        mean_x1, mean_y1, mean_x2, mean_y2,
        frame_width=frame_width, frame_height=frame_height,
    )


def _split_trajectory_clusters(trajectory_boxes: List[BBox], frame_width: int, frame_height: int):
    """Bucket a spread trajectory by normalized center (5% grid).

    Returns (dominant_bucket_boxes, other_bucket_boxes_list). The dominant
    bucket must hold at least half the trajectory, otherwise the split is not
    trustworthy and the whole trajectory is returned as one bucket.
    """
    if len(trajectory_boxes) < 3:
        return list(trajectory_boxes), []
    buckets: Dict[tuple, List[BBox]] = {}
    for b in trajectory_boxes:
        cx = (b.x1 + b.x2) / 2 / max(1.0, frame_width)
        cy = (b.y1 + b.y2) / 2 / max(1.0, frame_height)
        key = (int(cx * LAYOUT_CLUSTER_BUCKET), int(cy * LAYOUT_CLUSTER_BUCKET))
        buckets.setdefault(key, []).append(b)
    modal = max(buckets.values(), key=len)
    if len(modal) * 2 < len(trajectory_boxes):
        return list(trajectory_boxes), []
    others = [group for group in buckets.values() if group is not modal]
    return modal, others


def _layout_boxes_for_event(event: SubtitleEvent):
    """Render box, suppression plate and leftover boxes for one event.

    Empty trajectory (merged events, or detection failure) falls back to the
    mandatory anchor box. A trajectory whose centers span more than
    LAYOUT_SPREAD_WARN_RATIO of the frame is split into its dominant position
    cluster: the text renders at the dominant position's MEAN box (stable)
    while every other cluster contributes a suppression-only box, so a
    caption that genuinely appears at two positions neither renders in
    between nor leaks pixels at the secondary position. The PLATE is the
    UNION of the dominant cluster's boxes — the caption's worst-case extent —
    so the snugged background covers every glyph on every frame even when
    individual detections wobble.
    """
    frame_width = int(event.anchor_bbox.frame_width)
    frame_height = int(event.anchor_bbox.frame_height)
    boxes = event.bbox_trajectory or [event.anchor_bbox]
    centers_x = [(b.x1 + b.x2) / 2 / max(1.0, frame_width) for b in boxes]
    centers_y = [(b.y1 + b.y2) / 2 / max(1.0, frame_height) for b in boxes]
    if (
        max(centers_x) - min(centers_x) > LAYOUT_SPREAD_WARN_RATIO
        or max(centers_y) - min(centers_y) > LAYOUT_SPREAD_WARN_RATIO
    ):
        cluster, others = _split_trajectory_clusters(boxes, frame_width, frame_height)
        if others:
            logger.warning(
                "SubtitleEvent %s trajectory spans >%.0f%% of the frame; rendering at the dominant position (%d/%d boxes) and suppressing %d leftover position(s).",
                event.event_id,
                LAYOUT_SPREAD_WARN_RATIO * 100,
                len(cluster),
                len(boxes),
                len(others),
            )
            return (
                _stable_layout_box(cluster, frame_width, frame_height),
                _trajectory_union(cluster, frame_width, frame_height),
                [_stable_layout_box(group, frame_width, frame_height) for group in others],
            )
        logger.warning(
            "SubtitleEvent %s trajectory spans >%.0f%% of the frame with no dominant cluster; the stable layout box may sit between two caption positions.",
            event.event_id,
            LAYOUT_SPREAD_WARN_RATIO * 100,
        )
    return (
        _stable_layout_box(boxes, frame_width, frame_height),
        _trajectory_union(boxes, frame_width, frame_height),
        [],
    )


def _trajectory_union(boxes: List[BBox], frame_width: int, frame_height: int) -> BBox:
    """Worst-case extent of a box group: the union clamped into the frame."""
    return BBox.clamped(
        min(b.x1 for b in boxes),
        min(b.y1 for b in boxes),
        max(b.x2 for b in boxes),
        max(b.y2 for b in boxes),
        frame_width=frame_width,
        frame_height=frame_height,
    )


def _layout_box_for_event(event: SubtitleEvent) -> BBox:
    """Stable layout box only (kept for callers that ignore the plate)."""
    return _layout_boxes_for_event(event)[0]


def _translation_tokens(text: str) -> "frozenset[str]":
    """Normalized word tokens used to recognize near-duplicate translations."""
    import unicodedata

    value = str(text or "").lower().replace("đ", "d")
    value = unicodedata.normalize("NFD", value)
    value = "".join(ch for ch in value if unicodedata.category(ch) != "Mn")
    return frozenset(re.findall(r"[a-z0-9]{2,}", value))


def _unify_twin_events(
    subtitle_events: List[SubtitleEvent],
    event_plate_boxes: Dict[str, BBox],
    fps: float,
) -> Dict[str, tuple]:
    """Merge near-duplicate READY events into one suppression unit.

    A bilingual caption whose rows escaped the OCR merge exists as two
    events with UNEQUAL spans occupying the same caption block (vertically
    stacked plates, overlapping time). The per-frame dedupe keeps one for
    rendering, but each event's suppression used to end with its own span —
    the row whose event ended earlier stayed visible while its source text
    was still on screen (seen at 12.6-12.9s). Grouping is SPATIAL, not
    translation-based: per-row translations of one visual caption can
    diverge, but their plates stay stacked. The kept event inherits the
    union of all group plates and spans for suppression; rendering keeps
    using its own span.
    """
    suppress_spans: Dict[str, tuple] = {}
    twin_members: Dict[str, List[str]] = {}
    groups: List[List[SubtitleEvent]] = []
    ready = sorted(
        (ev for ev in subtitle_events if ev.replacement_status == "READY"),
        key=lambda ev: (ev.start_time, -(ev.end_time - ev.start_time)),
    )
    for ev in ready:
        plate = event_plate_boxes.get(ev.event_id)
        home = None
        for group in groups:
            kept_plate = event_plate_boxes.get(group[0].event_id)
            if plate is None or kept_plate is None:
                continue
            overlap = min(group[0].end_time, ev.end_time) - max(group[0].start_time, ev.start_time)
            if overlap <= 0:
                continue
            v_gap = max(0.0, max(kept_plate.y1, plate.y1) - min(kept_plate.y2, plate.y2))
            h_overlap = min(kept_plate.x2, plate.x2) - max(kept_plate.x1, plate.x1)
            if v_gap <= 24 and h_overlap >= 0.5 * min(
                kept_plate.x2 - kept_plate.x1, plate.x2 - plate.x1
            ):
                home = group
                break
        if home is None:
            groups.append([ev])
        else:
            home.append(ev)

    for group in groups:
        kept = group[0]
        span_start = min(int(round(ev.start_time * fps)) for ev in group)
        span_end = max(int(round(ev.end_time * fps)) for ev in group)
        suppress_spans[kept.event_id] = (span_start, span_end)
        if len(group) > 1:
            twin_members[kept.event_id] = [
                ev.event_id for ev in group if ev.event_id != kept.event_id
            ]
            plate = event_plate_boxes.get(kept.event_id)
            for twin in (ev for ev in group if ev.event_id != kept.event_id):
                twin_plate = event_plate_boxes.get(twin.event_id)
                if plate is None:
                    plate = twin_plate
                elif twin_plate is not None:
                    plate = BBox.clamped(
                        min(plate.x1, twin_plate.x1),
                        min(plate.y1, twin_plate.y1),
                        max(plate.x2, twin_plate.x2),
                        max(plate.y2, twin_plate.y2),
                        frame_width=plate.frame_width,
                        frame_height=plate.frame_height,
                    )
            if plate is not None:
                event_plate_boxes[kept.event_id] = plate
    return suppress_spans, twin_members


def _build_suppress_only_map(
    subtitle_events,
    event_layout_boxes: Dict[str, BBox],
    event_suppress_boxes: Dict[str, List[BBox]],
    fps: float,
    event_plate_boxes: Optional[Dict[str, BBox]] = None,
) -> Dict[int, List[BBox]]:
    """Frame map of boxes that are blurred but never rendered as text.

    Evidence rule: redaction (fail-safe) applies only where TEXT EVIDENCE
    exists. READY events contribute only their off-cluster leftover
    positions. OCR_EMPTY events are never redacted — the OCR pass (including
    its upscale verification) found no readable caption in the region, so a
    redaction would paint an empty box over scenery. OCR_FAILED and
    translation failures keep redaction: text was seen (or the reader
    crashed mid-batch), only the translation is missing. Redaction uses the
    event's PLATE box (worst-case trajectory union) so glyph edges never
    leak at trajectory extremes.
    """
    plates = event_plate_boxes or {}
    suppress_only_by_frame: Dict[int, List[BBox]] = {}
    for ev in subtitle_events:
        span_start = int(round(ev.start_time * fps))
        span_end = int(round(ev.end_time * fps))
        for leftover in event_suppress_boxes.get(ev.event_id, []):
            for f_i in range(span_start, min(span_end, int(fps * 3600)) + 1):
                suppress_only_by_frame.setdefault(f_i, []).append(leftover)
        if ev.replacement_status in {"READY", "OCR_EMPTY"}:
            continue
        box = plates.get(ev.event_id) or event_layout_boxes.get(ev.event_id) or ev.anchor_bbox
        for f_i in range(max(0, span_start - 2), min(span_end + 2, int(fps * 3600)) + 1):
            suppress_only_by_frame.setdefault(f_i, []).append(box)
    return suppress_only_by_frame


def _apply_band_normalization(
    event_layout_boxes: Dict[str, BBox], events_by_id: Dict[str, SubtitleEvent]
) -> Dict[str, BBox]:
    """Snap near-band READY caption boxes to one shared bottom edge/center.

    Consecutive captions of one video share one subtitle band; per-event
    means still wobble a few pixels and that wobble reads as the caption
    "jumping" between lines. The band bottom is the most common y2 bucket of
    the READY events; a box within BAND_Y2_TOLERANCE_RATIO of it is shifted
    to the shared bottom edge (height preserved) and, when its x-center is
    within BAND_X_CENTER_MAX_SHIFT_RATIO of the band center, re-centered
    (width preserved). Events far from the band (top captions, the upper row
    of a bilingual pair) keep their own geometry.
    """
    ready_boxes = [
        (event_id, box)
        for event_id, box in event_layout_boxes.items()
        if events_by_id.get(event_id) is not None
        and events_by_id[event_id].replacement_status == "READY"
    ]
    if len(ready_boxes) < 3:
        return event_layout_boxes
    frame_width = int(ready_boxes[0][1].frame_width)
    frame_height = int(ready_boxes[0][1].frame_height)
    bucket_size = BAND_Y2_TOLERANCE_RATIO / 3.0
    y2_buckets: Dict[int, List[tuple]] = {}
    for entry in ready_boxes:
        y2 = entry[1].y2 / max(1.0, frame_height)
        y2_buckets.setdefault(int(y2 / bucket_size), []).append(entry)
    modal_bucket = max(y2_buckets.values(), key=len)
    if len(modal_bucket) * 3 < len(ready_boxes):
        return event_layout_boxes
    band_y2 = sum(b.y2 for _, b in modal_bucket) / len(modal_bucket) / max(1.0, frame_height)
    band_cx = sum((b.x1 + b.x2) / 2 for _, b in modal_bucket) / len(modal_bucket) / max(1.0, frame_width)

    normalized = dict(event_layout_boxes)
    for event_id, box in ready_boxes:
        y2 = box.y2 / max(1.0, frame_height)
        if abs(y2 - band_y2) > BAND_Y2_TOLERANCE_RATIO:
            continue
        x1, y1, x2 = box.x1, box.y1, box.x2
        height = box.y2 - box.y1
        new_y2 = band_y2 * frame_height
        y1 = new_y2 - height
        y2_px = new_y2
        cx = (x1 + x2) / 2 / max(1.0, frame_width)
        if abs(cx - band_cx) <= BAND_X_CENTER_MAX_SHIFT_RATIO:
            shift = (band_cx - cx) * frame_width
            x1 += shift
            x2 += shift
        normalized[event_id] = BBox.clamped(
            x1, y1, x2, y2_px,
            frame_width=frame_width, frame_height=frame_height,
        )
    return normalized


def merge_same_frame_detections(
    detections_by_frame: Dict[int, List[Any]],
) -> Dict[int, List[Any]]:
    """Collapse same-frame detections into one union box per frame.

    The generic text model emits word-level boxes; blurring each box on its
    own leaks glyph edges and inter-word gaps of the original caption, so
    suppression operates on the caption-area union instead.
    """
    merged: Dict[int, List[Any]] = {}
    for frame_index, dets in detections_by_frame.items():
        if len(dets) <= 1:
            merged[frame_index] = dets
            continue
        template = dets[0]
        union_box = BBox(
            x1=min(d.bbox.x1 for d in dets),
            y1=min(d.bbox.y1 for d in dets),
            x2=max(d.bbox.x2 for d in dets),
            y2=max(d.bbox.y2 for d in dets),
            frame_width=template.bbox.frame_width,
            frame_height=template.bbox.frame_height,
        )
        merged[frame_index] = [
            Detection(
                class_id=template.class_id,
                class_name=template.class_name,
                confidence=max(d.confidence for d in dets),
                bbox=union_box,
                frame_index=template.frame_index,
                timestamp_seconds=template.timestamp_seconds,
            )
        ]
    return merged


@dataclass
class VisualRequest:
    enabled: bool = True
    video_path: str = ""
    subtitle_enabled: bool = True
    subtitle_mode: str = "translate_replace"  # translate_replace | blur | preserve
    subtitle_layout: str = "temporal"  # temporal | bottom | legacy_centered
    subtitle_background: str = "blur"
    target_language: str = "vi"
    hide_source_subtitle: bool = True
    stable_subtitle_layout: bool = True
    logo_enabled: bool = True
    logo_action: str = "blur"
    logo_image_path: Optional[str] = None
    watermark_enabled: bool = True
    watermark_action: str = "blur"
    other_text_enabled: bool = True
    other_text_action: str = "blur"
    segments: Optional[List[Any]] = None


@dataclass
class VisualResult:
    enabled: bool
    status: str  # SUCCESS | SKIPPED | FAILED
    rendered_video_path: Optional[str] = None
    detections_summary: Dict[str, int] = field(default_factory=dict)
    tracks_summary: Dict[str, int] = field(default_factory=dict)
    subtitle_events: List[SubtitleEvent] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    timings: Dict[str, float] = field(default_factory=dict)


class VisualPipeline:
    def __init__(
        self,
        detector: Optional[DetectorPort] = None,
        tracker: Optional[ClassAwareTracker] = None,
        subtitle_processor: Optional[SubtitleProcessor] = None,
        logo_processor: Optional[LogoProcessor] = None,
        watermark_processor: Optional[WatermarkProcessor] = None,
        other_text_processor: Optional[OtherTextProcessor] = None,
        scene_detector: Optional[SceneBoundaryPort] = None,
        video_util: Optional[VideoUtilPort] = None,
    ):
        self.detector = detector
        self.tracker = tracker or ClassAwareTracker()
        self.subtitle_processor = subtitle_processor
        self.logo_processor = logo_processor
        self.watermark_processor = watermark_processor
        self.other_text_processor = other_text_processor
        self.scene_detector = scene_detector
        self.video_util = video_util
        self._region_planner = None

    def _build_events_from_segments(
        self,
        segments: List[Any],
        detections: List[Detection],
        width: int,
        height: int,
        fps: float,
        target_language: str = "vi",
        subtitle_layout: str = "in-place",
    ) -> List[SubtitleEvent]:
        events: List[SubtitleEvent] = []
        subtitle_dets = [d for d in detections if getattr(d, "class_name", "") == "subtitle"]

        for idx, seg in enumerate(segments):
            if isinstance(seg, dict):
                start = float(seg.get("start", seg.get("start_time", 0.0)))
                end = float(seg.get("end", seg.get("end_time", 0.0)))
                source_text = str(seg.get("source_text", "") or "")
                target_text = str(seg.get("target_text") or seg.get("translated_text") or seg.get("text") or "").strip()
                seg_id = str(seg.get("id", seg.get("segment_id", f"seg_ev_{idx}")))
            else:
                start = float(getattr(seg, "start_time", None) if getattr(seg, "start_time", None) is not None else getattr(seg, "start", 0.0))
                end = float(getattr(seg, "end_time", None) if getattr(seg, "end_time", None) is not None else getattr(seg, "end", 0.0))
                source_text = str(getattr(seg, "source_text", "") or "")
                target_text = str(
                    getattr(seg, "translated_text", None)
                    or getattr(seg, "target_text", None)
                    or getattr(seg, "spoken_text", None)
                    or getattr(seg, "text", "")
                    or ""
                ).strip()
                seg_id = str(getattr(seg, "segment_id", None) or getattr(seg, "id", f"seg_ev_{idx}"))

            if not target_text and source_text and self.subtitle_processor and getattr(self.subtitle_processor, "translation_service", None):
                try:
                    res = self.subtitle_processor.translation_service.translate_texts([source_text], target_language=target_language)
                    if res and res[0]:
                        target_text = res[0].strip()
                except Exception:
                    pass

            start_f = int(round(start * fps))
            end_f = int(round(end * fps))

            matched_dets = [
                d for d in subtitle_dets
                if start_f <= getattr(d, "frame_index", -1) <= end_f
            ]

            seg_bbox = seg.get("bbox") if isinstance(seg, dict) else getattr(seg, "bbox", None)

            has_source_bbox = False
            if seg_bbox and isinstance(seg_bbox, dict) and all(k in seg_bbox for k in ("x1", "y1", "x2", "y2")) and subtitle_layout in {"in-place", "temporal"}:
                fw = int(seg_bbox.get("frame_width") or width)
                fh = int(seg_bbox.get("frame_height") or height)
                scale_x = width / fw if fw > 0 else 1.0
                scale_y = height / fh if fh > 0 else 1.0
                anchor = BBox.clamped(
                    x1=float(seg_bbox["x1"]) * scale_x,
                    y1=float(seg_bbox["y1"]) * scale_y,
                    x2=float(seg_bbox["x2"]) * scale_x,
                    y2=float(seg_bbox["y2"]) * scale_y,
                    frame_width=width,
                    frame_height=height,
                )
                trajectory = [anchor]
                indices = [start_f]
                has_source_bbox = True
            elif matched_dets and subtitle_layout in {"in-place", "temporal"}:
                matched_dets.sort(key=lambda d: getattr(d, "frame_index", 0))
                trajectory = [d.bbox for d in matched_dets]
                indices = [d.frame_index for d in matched_dets]
                best_det = max(matched_dets, key=lambda d: getattr(d, "confidence", 0.0))
                anchor = best_det.bbox
                has_source_bbox = True
            else:
                anchor = BBox(
                    x1=int(width * 0.08),
                    y1=int(height * 0.84),
                    x2=int(width * 0.92),
                    y2=int(height * 0.95),
                    frame_width=width,
                    frame_height=height,
                )
                trajectory = [anchor]
                indices = [start_f]
                has_source_bbox = False

            ev = SubtitleEvent(
                event_id=seg_id,
                track_id=f"track_seg_{idx}",
                start_time=start,
                end_time=end,
                anchor_bbox=anchor,
                bbox_trajectory=trajectory,
                bbox_frame_indices=indices,
                source_text=source_text,
                translated_text=target_text,
                split_reason="segment" if has_source_bbox else "no_source_sub",
                replacement_status="READY" if target_text else "TRANSLATION_EMPTY",
                blur_applied=bool(target_text and has_source_bbox),
                overlay_applied=bool(target_text),
            )
            events.append(ev)

        return events

    @staticmethod
    def _validate_request(request: VisualRequest) -> None:
        if request.subtitle_enabled and request.subtitle_mode not in VALID_SUBTITLE_MODES:
            allowed = ", ".join(sorted(VALID_SUBTITLE_MODES))
            raise ValueError(
                f"Unsupported subtitle_mode {request.subtitle_mode!r}; expected one of: {allowed}."
            )

    def _render_stable_layout(
        self,
        frame: Any,
        frame_index: int,
        in_place_events: List[SubtitleEvent],
        event_layout_boxes: Dict[str, BBox],
        event_render_spans: Dict[str, tuple],
        hide_source: bool,
        suppress_only_boxes: Optional[List[BBox]] = None,
        raw_sub_dets_by_frame: Optional[Dict[int, List[Any]]] = None,
        event_plate_boxes: Optional[Dict[str, BBox]] = None,
        suppress_spans: Optional[Dict[str, tuple]] = None,
        twin_members: Optional[Dict[str, List[str]]] = None,
    ) -> Any:
        """Draw translated captions at a per-event stable position.

        Layout geometry comes exclusively from the event's averaged trajectory
        box. Pre/post-roll events suppress early but render only inside their
        REAL time span, so an incoming caption never moves the current one —
        and pre/post-roll suppression only paints while the ORIGINAL caption
        is still detected on the frame, so a caption-free gap never flashes an
        empty panel. Failed (non-READY) events arrive via suppress_only_boxes:
        their region is blurred so the original caption is never left readable.
        Suppression plates use the event's worst-case trajectory union so the
        snugged background covers every glyph on every frame.
        """
        plates = event_plate_boxes or {}
        active_events: List[SubtitleEvent] = []
        # Span gating applies only when the caller supplies real spans; legacy
        # callers (and tests) without spans render every mapped event.
        render_spans = event_render_spans if event_render_spans is not None else None
        for event in in_place_events:
            span = render_spans.get(event.event_id) if render_spans is not None else None
            if render_spans is None or (span and span[0] <= frame_index <= span[1]):
                active_events.append(event)

        # Suppression covers every event mapped to this frame (pre/post-roll
        # included), padded identically to the render box so the text area is
        # always backed by suppression — plus the fail-safe boxes of
        # non-READY events so failed captions are never left readable.
        if hide_source and hasattr(self.subtitle_processor, "suppress_source_subtitles"):
            suppress_boxes = []
            twin_spans = suppress_spans or {}
            for event in in_place_events:
                box = plates.get(event.event_id) or event_layout_boxes.get(event.event_id, event.anchor_bbox)
                span = twin_spans.get(event.event_id)
                if span is None:
                    span = render_spans.get(event.event_id) if render_spans is not None else None
                if span is not None and not (span[0] <= frame_index <= span[1]):
                    # Pre/post-roll frame: the panel must only hide source
                    # pixels that are actually there. Raw per-frame subtitle
                    # detections (no backward fill) tell us whether the
                    # original caption is still visible in this lane.
                    raw_dets = (raw_sub_dets_by_frame or {}).get(frame_index, [])
                    if raw_sub_dets_by_frame is not None and not any(
                        same_lane(det.bbox, box) for det in raw_dets
                    ):
                        continue
                event_pad = max(LAYOUT_BOX_PAD_MIN_PX, LAYOUT_BOX_PAD_RATIO * float(box.frame_height))
                suppress_boxes.append(BBox.clamped(
                    box.x1 - event_pad, box.y1 - event_pad,
                    box.x2 + event_pad, box.y2 + event_pad,
                    frame_width=box.frame_width, frame_height=box.frame_height,
                ))
            for box in (suppress_only_boxes or []):
                event_pad = max(LAYOUT_BOX_PAD_MIN_PX, LAYOUT_BOX_PAD_RATIO * float(box.frame_height))
                suppress_boxes.append(BBox.clamped(
                    box.x1 - event_pad, box.y1 - event_pad,
                    box.x2 + event_pad, box.y2 + event_pad,
                    frame_width=box.frame_width, frame_height=box.frame_height,
                ))
            if suppress_boxes:
                frame = self.subtitle_processor.suppress_source_subtitles(frame, suppress_boxes)

        if not active_events:
            return frame

        # The same bilingual caption can exist as two lane events (upper CJK
        # row + lower Latin row) that escaped the pre-OCR merge; drawing both
        # duplicates the translated string at two positions and makes the band
        # jump when one ends. Keep one event per identical or near-identical
        # translation — the OCR verification pass recovers small rows whose
        # Vietnamese then differs from its twin by one word only. Identical
        # text is just Jaccard 1.0, so one token-overlap test covers both.
        kept_events: List[tuple] = []
        twin_map = twin_members or {}

        def _pair_threshold(a: SubtitleEvent, b: SubtitleEvent) -> float:
            # Unified twin rows share one visual caption, so their per-row
            # translations can drift further apart than unrelated captions
            # would — they collapse at a looser threshold. Unrelated captions
            # collapse only when the wording is essentially the same: distinct
            # captions legitimately share most words ('-Đô đốc Tổng bộ Hải
            # quân-' above '3 Đô đốc Hải quân', Jaccard 0.67) and both must
            # render.
            if b.event_id in twin_map.get(a.event_id, []) or a.event_id in twin_map.get(b.event_id, []):
                return 0.4
            return 0.75

        for event in active_events:
            tokens = _translation_tokens(event.translated_text)
            if not tokens:
                kept_events.append((tokens, event))
                continue
            twin = None
            twin_jaccard = -1.0
            for entry in kept_events:
                if not entry[0]:
                    continue
                jaccard = len(entry[0] & tokens) / max(1, len(entry[0] | tokens))
                if jaccard >= _pair_threshold(entry[1], event) and jaccard > twin_jaccard:
                    twin = entry
                    twin_jaccard = jaccard
            if twin is None:
                kept_events.append((tokens, event))
                continue
            # Near-duplicate: the longer-span event wins the render slot.
            kept_tokens, kept = twin
            if (event.end_time - event.start_time, event.event_id) > (
                kept.end_time - kept.start_time, kept.event_id
            ):
                kept_events[kept_events.index(twin)] = (tokens, event)
        active_events = [ev for _tokens, ev in kept_events]

        groups: List[list] = []
        for event in active_events:
            stable_box = event_layout_boxes.get(event.event_id) or event.anchor_bbox
            group = next((g for g in groups if same_lane(g[0], stable_box)), None)
            if group is None:
                groups.append([stable_box, [event]])
            else:
                group[0] = union_boxes([group[0], stable_box])
                group[1].append(event)

        for lane_box, events in groups:
            chosen = (
                self.subtitle_processor.select_render_events(events)
                if hasattr(self.subtitle_processor, "select_render_events")
                else events
            )
            # A lane can hold distinct caption ROWS (a persistent title row
            # above rotating content rows). Rendering them inside the lane
            # union collapses different meanings into one overlay and drops
            # the other rows' text. Cluster the survivors by vertical box
            # overlap — same-line duplicates were already collapsed by the
            # token dedupe and select_render_events — and give each row its
            # own stable box so its text stays pixel-stable.
            row_clusters: List[list] = []
            for event in chosen:
                box = event_layout_boxes.get(event.event_id) or event.anchor_bbox
                cluster = next(
                    (
                        entry
                        for entry in row_clusters
                        if box.y1 < entry[0].y2 - 2 and entry[0].y1 < box.y2 - 2
                    ),
                    None,
                )
                if cluster is None:
                    row_clusters.append([box, [event]])
                else:
                    cluster[0] = union_boxes([cluster[0], box])
                    cluster[1].append(event)
            for cluster_box, cluster_events in row_clusters:
                pad = max(LAYOUT_BOX_PAD_MIN_PX, LAYOUT_BOX_PAD_RATIO * float(cluster_box.frame_height))
                layout_box = BBox.clamped(
                    cluster_box.x1 - pad,
                    cluster_box.y1 - pad,
                    cluster_box.x2 + pad,
                    cluster_box.y2 + pad,
                    frame_width=cluster_box.frame_width,
                    frame_height=cluster_box.frame_height,
                )
                frame = self.subtitle_processor.render_region_events(
                    frame, cluster_events, layout_box, bg_box=False
                )
        return frame

    def _apply_subtitle_actions(
        self,
        frame: Any,
        frame_index: int,
        request: VisualRequest,
        detections_by_frame: Dict[int, List[Any]],
        events_by_frame: Dict[int, List[SubtitleEvent]],
        event_layout_boxes: Optional[Dict[str, BBox]] = None,
        event_render_spans: Optional[Dict[str, tuple]] = None,
        suppress_only_by_frame: Optional[Dict[int, List[BBox]]] = None,
        raw_sub_dets_by_frame: Optional[Dict[int, List[Any]]] = None,
        event_plate_boxes: Optional[Dict[str, BBox]] = None,
        suppress_spans: Optional[Dict[str, tuple]] = None,
        twin_members: Optional[Dict[str, List[str]]] = None,
    ) -> Any:
        """Apply the selected subtitle policy without destructive fallback."""
        if not request.subtitle_enabled or request.subtitle_mode == "preserve":
            return frame

        if self.subtitle_processor is None:
            has_work = bool(detections_by_frame.get(frame_index)) or bool(
                events_by_frame.get(frame_index)
            )
            if has_work:
                raise RuntimeError(
                    "SubtitleProcessor is required for subtitle blur/replacement."
                )
            return frame

        if request.subtitle_mode == "blur":
            for detection in detections_by_frame.get(frame_index, []):
                bbox = self.subtitle_processor.get_source_suppression_bbox(
                    detection.bbox, pad_x=18, pad_y=14
                )
                frame = self.subtitle_processor.blur_action.execute(
                    frame, bbox, padding_px=0
                )
            return frame

        ready_events = [
            event
            for event in events_by_frame.get(frame_index, [])
            if event.replacement_status == "READY"
            and event.blur_applied
            and event.overlay_applied
        ]
        is_in_place = getattr(request, "subtitle_layout", "temporal") in {"temporal", "in-place"}
        hide_source = getattr(request, "hide_source_subtitle", True)
        if hasattr(self.subtitle_processor, "subtitle_layout"):
            self.subtitle_processor.subtitle_layout = getattr(request, "subtitle_layout", "temporal")
        in_place_events = [e for e in ready_events if e.blur_applied]
        bottom_cinema_events = [e for e in ready_events if not e.blur_applied]

        has_suppress_only = bool((suppress_only_by_frame or {}).get(frame_index))
        if (in_place_events or has_suppress_only) and is_in_place and self._region_planner is not None:
            if getattr(request, "stable_subtitle_layout", True) and event_layout_boxes:
                # Stable layout: the caption position comes from one
                # per-event box; per-frame detections feed suppression only.
                frame = self._render_stable_layout(
                    frame,
                    frame_index,
                    in_place_events,
                    event_layout_boxes,
                    event_render_spans,
                    hide_source=getattr(request, "hide_source_subtitle", True),
                    suppress_only_boxes=(suppress_only_by_frame or {}).get(frame_index, []),
                    raw_sub_dets_by_frame=raw_sub_dets_by_frame,
                    event_plate_boxes=event_plate_boxes,
                    suppress_spans=suppress_spans,
                    twin_members=twin_members,
                )
                return frame
            # Keep independent lanes separate; select one semantic caption only
            # within the same spatial group.
            groups = []
            for event in in_place_events:
                anchor = (
                    self.subtitle_processor.get_event_bbox(event, frame_index)
                    if hasattr(self.subtitle_processor, "get_event_bbox")
                    else event.anchor_bbox
                )
                region = self._region_planner.region_for(frame_index, anchor)
                group = next((g for g in groups if same_lane(g[0], region)), None)
                if group is None:
                    groups.append([region, [event]])
                else:
                    group[0] = union_boxes([group[0], region])
                    group[1].append(event)
            for region, events in groups:
                current = [d.bbox for d in detections_by_frame.get(frame_index, []) if same_lane(region, d.bbox)]
                chosen = (
                    self.subtitle_processor.select_render_events(events)
                    if hasattr(self.subtitle_processor, "select_render_events")
                    else events
                )
                # True in-place: constrain X to active event/detection bboxes only,
                # while keeping Y anchored to the stable scene lane.
                active_boxes = []
                for ev in chosen:
                    box = None
                    if hasattr(self.subtitle_processor, "get_event_bbox"):
                        try:
                            candidate = self.subtitle_processor.get_event_bbox(ev, frame_index)
                            if isinstance(candidate, BBox):
                                box = candidate
                        except Exception:
                            box = None
                    if box is None:
                        box = getattr(ev, "anchor_bbox", None)
                    if isinstance(box, BBox):
                        active_boxes.append(box)

                if current:
                    for d in current:
                        b = getattr(d, "bbox", None)
                        if isinstance(b, BBox):
                            active_boxes.append(b)

                if active_boxes:
                    raw_x1 = min(float(b.x1) for b in active_boxes)
                    raw_x2 = max(float(b.x2) for b in active_boxes)
                    raw_y1 = min(float(b.y1) for b in active_boxes)
                    raw_y2 = max(float(b.y2) for b in active_boxes)

                    ax1 = max(0.0, raw_x1 - 16.0)
                    ax2 = min(float(region.frame_width), raw_x2 + 16.0)
                    ay1 = max(0.0, min(float(region.y1), raw_y1 - 6.0))
                    ay2 = min(float(region.frame_height), max(float(region.y2), raw_y2 + 6.0))
                    tight_box = BBox(x1=ax1, y1=ay1, x2=ax2, y2=ay2, frame_width=region.frame_width, frame_height=region.frame_height)
                else:
                    tight_box = region

                # Suppress source subtitle glyphs first (inpaint / local blur),
                # then overlay translated text with clean stroke outline.
                if hide_source:
                    suppress_boxes = list(current) + [tight_box]
                    if hasattr(self.subtitle_processor, "suppress_source_subtitles"):
                        frame = self.subtitle_processor.suppress_source_subtitles(
                            frame, suppress_boxes
                        )
                frame = self.subtitle_processor.render_region_events(
                    frame, chosen, tight_box, bg_box=False
                )

            # Render non-source cinema subtitles if any without black box
            if bottom_cinema_events and hasattr(self.subtitle_processor, "render_frame_events"):
                frame = self.subtitle_processor.render_frame_events(
                    frame, frame_index, bottom_cinema_events, anchor_to_bbox=False
                )
            return frame

        if ready_events:
            # Only suppress source subtitles when explicitly requested (default True).
            # When False, original subtitles remain untouched for bilingual display.
            if hide_source:
                suppression_regions = list(detections_by_frame.get(frame_index, []))
                get_event_bbox = getattr(self.subtitle_processor, "get_event_bbox", None)
                for event in ready_events:
                    suppression_regions.append(
                        get_event_bbox(event, frame_index)
                        if callable(get_event_bbox)
                        else event.anchor_bbox
                    )
                if hasattr(self.subtitle_processor, "suppress_source_subtitles"):
                    frame = self.subtitle_processor.suppress_source_subtitles(
                        frame, suppression_regions
                    )
            ready_events = (
                self.subtitle_processor.select_render_events(ready_events)
                if hasattr(self.subtitle_processor, "select_render_events")
                else ready_events
            )
            if hasattr(self.subtitle_processor, "render_frame_events"):
                frame = self.subtitle_processor.render_frame_events(
                    frame, frame_index, ready_events
                )
        return frame

    def _open_frame_writer(self, output_path: str, fps: float, width: int, height: int):
        """Prefer the injected VideoUtilPort's single-pass H.264 encoder so the
        artifact is web-compatible without a second full re-encode; fall back
        to OpenCV's mp4v writer when the port cannot provide one."""
        if self.video_util is not None:
            try:
                return self.video_util.open_encoded_frame_writer(output_path, fps, width, height)
            except NotImplementedError:
                logger.info("Visual: VideoUtilPort has no frame writer; falling back to OpenCV mp4v.")
        import cv2
        return cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    def process(self, request: VisualRequest, job_workspace_dir: str, output_video_path: str) -> VisualResult:
        start_time = time.monotonic()
        timings = {}
        warnings = []

        if not request.enabled:
            logger.info("VisualPipeline is disabled in request. Skipping execution.")
            return VisualResult(
                enabled=False,
                status="SKIPPED",
                warnings=["Visual processing disabled in request."],
            )

        self._validate_request(request)

        if self.detector is None:
            raise RuntimeError("YOLO Detector is not configured for enabled VisualPipeline.")

        ws_dir = Path(job_workspace_dir)
        visual_dir = ws_dir / "visual"
        visual_dir.mkdir(parents=True, exist_ok=True)

        # 1. YOLO 4-class Detection
        t0 = time.monotonic()
        logger.info("Visual: running YOLO detection on '%s'...", Path(request.video_path).name)
        detections = self.detector.detect_video(request.video_path, stride=1)
        timings["detection"] = round(time.monotonic() - t0, 3)

        class_counts: Dict[str, int] = {}
        for d in detections:
            class_counts[d.class_name] = class_counts.get(d.class_name, 0) + 1
        logger.info(
            "Visual: detection done (%.1fs) — %d boxes %s",
            timings["detection"], len(detections), class_counts,
        )

        # 2. Class-Aware Tracking
        t0 = time.monotonic()
        tracks = self.tracker.track(detections)
        timings["tracking"] = round(time.monotonic() - t0, 3)

        track_summary: Dict[str, int] = {}
        for tr in tracks:
            track_summary[tr.class_name] = track_summary.get(tr.class_name, 0) + 1
        logger.info("Visual: tracking done — %d tracks %s", len(tracks), track_summary)

        # Route tracks by class
        subtitle_tracks = [t for t in tracks if t.class_name == "subtitle"]
        logo_tracks = [t for t in tracks if t.class_name == "logo"]
        watermark_tracks = [t for t in tracks if t.class_name == "watermark"]
        other_tracks = [t for t in tracks if t.class_name == "other_burn_in_text"]

        self._region_planner = None
        if request.subtitle_enabled and request.subtitle_mode == 'translate_replace' and request.subtitle_layout in {'temporal', 'in-place'}:
            import json
            cuts = self.scene_detector.detect_cuts(request.video_path) if self.scene_detector else []
            self._region_planner = SubtitleRegionPlanner(detections, cuts)
            (visual_dir / 'subtitle_regions.json').write_text(json.dumps(self._region_planner.to_dict(), indent=2), encoding='utf-8')

        # 3. Subtitle Processing
        t0 = time.monotonic()
        subtitle_events = []
        if (
            request.subtitle_enabled
            and request.subtitle_mode == "translate_replace"
        ):
            if subtitle_tracks and self.subtitle_processor is not None:
                subtitle_events = self.subtitle_processor.process_tracks(
                    tracks=subtitle_tracks,
                    target_language=request.target_language,
                    video_path=request.video_path,
                )
                # If caller provided segments (e.g. edited translations from Web UI),
                # enrich the YOLO events with matching user translations
                if request.segments and subtitle_events:
                    for ev in subtitle_events:
                        best_seg = None
                        best_overlap = 0.0
                        for seg in request.segments:
                            s_start = float(seg.get("start", seg.get("start_time", 0.0)) if isinstance(seg, dict) else getattr(seg, "start_time", getattr(seg, "start", 0.0)))
                            s_end = float(seg.get("end", seg.get("end_time", 0.0)) if isinstance(seg, dict) else getattr(seg, "end_time", getattr(seg, "end", 0.0)))
                            overlap = max(0.0, min(ev.end_time, s_end) - max(ev.start_time, s_start))
                            if overlap > best_overlap:
                                best_overlap = overlap
                                best_seg = seg
                        if best_seg and best_overlap > 0.3:
                            t_text = (best_seg.get("target_text") or best_seg.get("translated_text") or best_seg.get("text")) if isinstance(best_seg, dict) else (getattr(best_seg, "translated_text", None) or getattr(best_seg, "target_text", None) or getattr(best_seg, "text", None))
                            if t_text and str(t_text).strip():
                                ev.translated_text = str(t_text).strip()
                                ev.replacement_status = "READY"
                                ev.blur_applied = True
                                ev.overlay_applied = True
            elif request.segments:
                import cv2
                _probe_cap = cv2.VideoCapture(request.video_path)
                _fps = _probe_cap.get(cv2.CAP_PROP_FPS) or 30.0
                _w = int(_probe_cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
                _h = int(_probe_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
                _probe_cap.release()
                subtitle_events = self._build_events_from_segments(
                    segments=request.segments,
                    detections=detections,
                    width=_w,
                    height=_h,
                    fps=_fps,
                    target_language=request.target_language,
                    subtitle_layout=request.subtitle_layout,
                )
            elif subtitle_tracks:
                if self.subtitle_processor is None:
                    raise RuntimeError(
                        "SubtitleProcessor is required for translate_replace mode."
                    )
                subtitle_events = self.subtitle_processor.process_tracks(
                    tracks=subtitle_tracks,
                    target_language=request.target_language,
                    video_path=request.video_path,
                )
        timings["subtitle_processing"] = round(time.monotonic() - t0, 3)
        logger.info(
            "Visual: subtitle processing done — %d events (%d READY) from %d subtitle tracks",
            len(subtitle_events),
            sum(1 for ev in subtitle_events if ev.replacement_status == "READY"),
            len(subtitle_tracks),
        )

        # Per-event stable layout geometry (see _stable_layout_box): computed
        # once per event, reused for every frame the event covers. Suppressed
        # entirely when the caller opts out via stable_subtitle_layout=False.
        # A spread trajectory renders at its dominant position cluster and the
        # leftover positions become suppression-only boxes; finally READY
        # caption boxes near the shared bottom band snap to one common edge
        # so consecutive captions stop wobbling.
        event_layout_boxes: Dict[str, BBox] = {}
        event_suppress_boxes: Dict[str, List[BBox]] = {}
        event_plate_boxes: Dict[str, BBox] = {}
        if request.stable_subtitle_layout:
            for ev in subtitle_events:
                box, plate, leftovers = _layout_boxes_for_event(ev)
                event_layout_boxes[ev.event_id] = box
                event_plate_boxes[ev.event_id] = plate
                event_suppress_boxes[ev.event_id] = leftovers
            events_by_id = {ev.event_id: ev for ev in subtitle_events}
            event_layout_boxes = _apply_band_normalization(event_layout_boxes, events_by_id)
        else:
            suppress_spans = {}

        # Persist subtitle-event audit so a detector-only success can never be
        # mistaken for a successful replacement render. This is especially
        # important because translate_replace is intentionally fail-safe and
        # preserves source pixels when OCR/translation is unavailable.
        if request.subtitle_enabled and request.subtitle_mode == "translate_replace":
            import json
            audit_rows = []
            status_counts: Dict[str, int] = {}
            for ev in subtitle_events:
                status_counts[ev.replacement_status] = status_counts.get(ev.replacement_status, 0) + 1
                layout_box = event_layout_boxes.get(ev.event_id)
                audit_rows.append({
                    "event_id": ev.event_id,
                    "track_id": ev.track_id,
                    "start_time": ev.start_time,
                    "end_time": ev.end_time,
                    "replacement_status": ev.replacement_status,
                    "source_text": ev.source_text,
                    "translated_text": ev.translated_text,
                    "blur_applied": ev.blur_applied,
                    "overlay_applied": ev.overlay_applied,
                    "split_reason": ev.split_reason,
                    "anchor_bbox": [ev.anchor_bbox.x1, ev.anchor_bbox.y1, ev.anchor_bbox.x2, ev.anchor_bbox.y2],
                    "layout_bbox": (
                        [layout_box.x1, layout_box.y1, layout_box.x2, layout_box.y2]
                        if layout_box is not None else None
                    ),
                })
            (visual_dir / "subtitle_events.json").write_text(
                json.dumps({
                    "subtitle_detection_count": class_counts.get("subtitle", 0),
                    "subtitle_track_count": len(subtitle_tracks),
                    "event_count": len(subtitle_events),
                    "status_counts": status_counts,
                    "events": audit_rows,
                }, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        # Initialize Class Processors
        logo_proc = self.logo_processor or LogoProcessor(action_name=request.logo_action)
        watermark_proc = self.watermark_processor or WatermarkProcessor(action_name=request.watermark_action)
        other_proc = self.other_text_processor or OtherTextProcessor(action_name=request.other_text_action)

        # 4. Video Frame Rendering
        t0 = time.monotonic()
        import cv2

        cap = cv2.VideoCapture(request.video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video for visual rendering: {request.video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        # VisualPipeline owns pixels only. It deliberately emits a video-only
        # artifact; final audio muxing belongs to ProcessVideoUseCase.
        Path(output_video_path).parent.mkdir(parents=True, exist_ok=True)
        visual_only_path = str(Path(output_video_path).resolve())
        writer = self._open_frame_writer(visual_only_path, fps, width, height)

        frame_idx = 0
        events_by_frame: Dict[int, List[SubtitleEvent]] = {}
        # Render spans are the REAL event times: pre/post-roll extends the
        # suppression map (hide the incoming caption early) but must not move
        # the rendered text — otherwise the caption jumps before it appears.
        # Near-duplicate twin events (the two rows of one caption) share one
        # suppression unit: union plate + union span for the kept event, so
        # neither row's source text outlives its own event.
        suppress_spans, twin_members = _unify_twin_events(subtitle_events, event_plate_boxes, fps)
        event_render_spans: Dict[str, tuple] = {
            ev.event_id: (int(round(ev.start_time * fps)), int(round(ev.end_time * fps)))
            for ev in subtitle_events
        }
        ready_events = sorted(
            [ev for ev in subtitle_events if ev.replacement_status == "READY"],
            key=lambda ev: ev.start_time,
        )
        # Fail-safe redaction: events whose translation/OCR failed must still
        # hide the original caption — leaving it readable was reported as the
        # "missing subtitle" regression. Non-READY events contribute their
        # layout box to the suppression map only (never rendered as text).
        # Ultra-short tracks (detection slivers, 1-4 frames) are skipped:
        # nothing is readable in that span and redacting them only produced
        # flickering dark boxes over scene content.
        suppress_only_by_frame = _build_suppress_only_map(
            subtitle_events, event_layout_boxes, event_suppress_boxes, fps,
            event_plate_boxes=event_plate_boxes,
        )
        for index, ev in enumerate(ready_events):
            coarse_start = int(round(ev.start_time * fps))
            coarse_end = int(round(ev.end_time * fps))
            expected_frames = max(1, coarse_end - coarse_start + 1)
            continuity = min(1.0, len(ev.bbox_trajectory) / float(expected_frames))
            if continuity >= 0.85:
                pre_roll, post_roll = 1, 2
            elif continuity >= 0.60:
                pre_roll, post_roll = 2, 3
            else:
                pre_roll, post_roll = 3, 4

            start_f = max(0, coarse_start - pre_roll)
            end_f = coarse_end + post_roll

            if index > 0:
                prev = ready_events[index - 1]
                prev_end = int(round(prev.end_time * fps))
                if prev.track_id == ev.track_id:
                    start_f = max(start_f, prev_end + 1)
            if index + 1 < len(ready_events):
                nxt = ready_events[index + 1]
                next_start = int(round(nxt.start_time * fps))
                if nxt.track_id == ev.track_id:
                    end_f = min(end_f, next_start - 1)

            for f_i in range(start_f, max(start_f, end_f) + 1):
                events_by_frame.setdefault(f_i, []).append(ev)

        sub_dets_by_frame: Dict[int, List[Any]] = {}
        # Real per-frame subtitle detections, used by the stable renderer to
        # decide whether pre/post-roll suppression has anything to hide: the
        # backward-filled map below would report detections in caption-free
        # gaps and cause empty-panel flashes.
        raw_sub_dets_by_frame: Dict[int, List[Any]] = {}
        for tr in subtitle_tracks:
            for det in tr.detections:
                sub_dets_by_frame.setdefault(det.frame_index, []).append(det)
                raw_sub_dets_by_frame.setdefault(det.frame_index, []).append(det)

            # Backward-fill safety: If track starts at frame N, fill preceding frames (~25 frames / 0.8s) if unmapped
            if tr.detections:
                first_det = tr.detections[0]
                start_idx = first_det.frame_index
                backward_fill_count = int(round(fps * 0.85))  # ~25 frames
                for b_idx in range(max(0, start_idx - backward_fill_count), start_idx):
                    if b_idx not in sub_dets_by_frame:
                        sub_dets_by_frame.setdefault(b_idx, []).append(first_det)

        # Word-level generic-model boxes leave glyph edges and inter-word gaps
        # outside their boxes; suppression/blur must cover the full caption
        # area, so collapse same-frame subtitle detections into one union box.
        sub_dets_by_frame = merge_same_frame_detections(sub_dets_by_frame)
        raw_sub_dets_by_frame = merge_same_frame_detections(raw_sub_dets_by_frame)

        try:
            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    break

                # A. Process non-subtitle visual processors
                if request.logo_enabled:
                    frame = logo_proc.process_frame_tracks(frame, frame_idx, logo_tracks)
                if request.watermark_enabled:
                    frame = watermark_proc.process_frame_tracks(frame, frame_idx, watermark_tracks)
                if request.other_text_enabled:
                    frame = other_proc.process_frame_tracks(frame, frame_idx, other_tracks)

                # B. Subtitle policy: preserve, blur-only, or fail-safe translation.
                frame = self._apply_subtitle_actions(
                    frame,
                    frame_idx,
                    request,
                    sub_dets_by_frame,
                    events_by_frame,
                    event_layout_boxes=event_layout_boxes,
                    event_render_spans=event_render_spans,
                    suppress_only_by_frame=suppress_only_by_frame,
                    raw_sub_dets_by_frame=raw_sub_dets_by_frame,
                    event_plate_boxes=event_plate_boxes,
                    suppress_spans=suppress_spans,
                    twin_members=twin_members,
                )

                writer.write(frame)
                frame_idx += 1
                if total_frames and frame_idx % 100 == 0:
                    render_pct = int(100 * frame_idx / total_frames)
                    logger.info(
                        "Visual: rendering %d/%d frames (%d%%)",
                        frame_idx, total_frames, render_pct,
                        extra={"mediaflow_pct": render_pct, "mediaflow_stage": "visual"},
                    )
        finally:
            # A mid-render crash must not leak the ffmpeg encoder process or
            # the capture handle (both hold file locks on Windows).
            cap.release()
            writer.release()

        timings["rendering"] = round(time.monotonic() - t0, 3)
        logger.info(
            "Visual: rendering done — %d frames in %.1fs",
            frame_idx, timings["rendering"],
        )
        timings["total"] = round(time.monotonic() - start_time, 3)

        visual_status = "SUCCESS"
        if request.subtitle_enabled and request.subtitle_mode == "translate_replace" and subtitle_tracks:
            ready_count = sum(1 for ev in subtitle_events if ev.replacement_status == "READY" and ev.overlay_applied)
            if ready_count == 0:
                visual_status = "FAILED"
                warnings.append(
                    f"Detected {class_counts.get('subtitle', 0)} subtitle boxes across "
                    f"{len(subtitle_tracks)} tracks, but 0 replacement events were READY; "
                    "source pixels were preserved. Inspect visual/subtitle_events.json."
                )
            elif ready_count < len(subtitle_events):
                warnings.append(
                    f"Subtitle replacement partial: {ready_count}/{len(subtitle_events)} events READY."
                )

        return VisualResult(
            enabled=True,
            status=visual_status,
            rendered_video_path=visual_only_path,
            detections_summary=class_counts,
            tracks_summary=track_summary,
            subtitle_events=subtitle_events,
            warnings=warnings,
            timings=timings,
        )
