"""
Subtitle Processor

Manages subtitle-specific processing with destructive replacement safety:
YOLO subtitle track -> SubtitleEventSegmenter -> SubtitleEvent[] -> OCR -> shared TranslationService -> Fail-Safe Check -> BlurAction & SubtitleRenderer.
"""

from __future__ import annotations

import logging
import re
from bisect import bisect_left
from typing import Any, List, Optional

from src.core.contracts import OCRPort
from src.core.models.bbox import BBox
from src.core.models.subtitle_event import SubtitleEvent
from src.core.models.track import Track
from src.features.translation.service import TranslationService
from src.features.visual.actions.blur import BlurAction
from src.features.visual.subtitle.boundary_refiner import SubtitleBoundaryRefiner
from src.features.visual.subtitle.event_segmenter import SubtitleEventSegmenter
from src.features.visual.subtitle.renderer import SubtitleRenderer

logger = logging.getLogger(__name__)


class SubtitleProcessor:
    def __init__(
        self,
        ocr_provider: Optional[OCRPort] = None,
        translation_service: Optional[TranslationService] = None,
        segmenter: Optional[SubtitleEventSegmenter] = None,
        boundary_refiner: Optional[SubtitleBoundaryRefiner] = None,
        renderer: Optional[SubtitleRenderer] = None,
        blur_action: Optional[BlurAction] = None,
    ):
        self.ocr_provider = ocr_provider
        self.translation_service = translation_service
        self.segmenter = segmenter or SubtitleEventSegmenter()
        self.boundary_refiner = boundary_refiner or SubtitleBoundaryRefiner()
        self.renderer = renderer or SubtitleRenderer()
        self.blur_action = blur_action or BlurAction(strength=25.0, padding_px=4)

    def process_tracks(
        self,
        tracks: List[Track],
        target_language: str = "vi",
        video_path: Optional[str] = None,
    ) -> List[SubtitleEvent]:
        all_events: List[SubtitleEvent] = []

        # 1. Segment spatial tracks into temporal SubtitleEvents
        for track in tracks:
            events = self.segmenter.segment_track(track, video_path=video_path)
            all_events.extend(events)

        if not all_events:
            return []

        # Refine coarse YOLO first/last timestamps using neighboring video frames.
        # This only extends an event while adjacent frames visually match the same
        # subtitle; it deliberately avoids fixed +N ms padding.
        if video_path and self.boundary_refiner is not None:
            all_events = self.boundary_refiner.refine(all_events, video_path)

        # Bilingual burned-in subtitles are commonly emitted by YOLO as two
        # vertically stacked tracks (e.g. Chinese + English) for the same semantic
        # caption. Treating them independently causes duplicate overlays and can
        # associate one row's translation with a neighboring event. Pair strongly
        # overlapping stacked events into one union crop before OCR/translation.
        all_events = self._merge_parallel_bilingual_events(all_events)

        # YOLO occasionally misses an entire subtitle lane for 1-3 seconds even
        # though the burned-in caption remains clearly visible. Recover only such
        # bounded gaps (between two known bottom-lane subtitle events) by sampling
        # the stable bottom band and requiring repeated OCR evidence before creating
        # a synthetic event. This avoids leaving raw source subtitles exposed while
        # still refusing to invent captions in genuine no-text gaps.
        if video_path and self.ocr_provider is not None:
            all_events = self._recover_bottom_lane_gap_events(all_events, video_path)

        # 2. OCR Stage (Multi-candidate crop selection & consensus)
        if self.ocr_provider is not None and video_path:
            import cv2

            cap = cv2.VideoCapture(video_path)
            all_crops = []
            crop_metadata = []

            for event in all_events:
                # Gap-recovery events already carry repeated OCR-confirmed source
                # text. Do not spend another OCR request re-reading the same band.
                if event.source_text:
                    continue
                dur = max(0.0, event.end_time - event.start_time)
                mid_ts = (event.start_time + event.end_time) / 2.0
                timestamps = [mid_ts]

                if dur >= 1.2:
                    t1 = max(event.start_time + 0.1, mid_ts - 0.3)
                    timestamps.append(t1)

                for c_idx, ts in enumerate(timestamps):
                    cap.set(cv2.CAP_PROP_POS_MSEC, ts * 1000.0)
                    ret, frame_img = cap.read()
                    if ret and frame_img is not None:
                        x1, y1, x2, y2 = event.anchor_bbox.xyxy_int()
                        crop = frame_img[y1:y2, x1:x2]
                        if crop.size > 0:
                            all_crops.append(crop)
                            crop_metadata.append((event, c_idx))

            cap.release()

            if all_crops:
                try:
                    texts = self.ocr_provider.read_crops(all_crops)
                    event_candidates = {}
                    for (ev, c_idx), text in zip(crop_metadata, texts):
                        event_candidates.setdefault(ev.event_id, []).append(text.strip() if text else "")

                    for event in all_events:
                        if event.source_text:
                            # Already confirmed by conservative gap-recovery OCR.
                            continue
                        cands = event_candidates.get(event.event_id, [])
                        final_text = self._resolve_ocr_consensus(cands)
                        final_text = self._select_bilingual_source_text(final_text)
                        final_text = self._sanitize_caption_text(final_text)
                        if final_text:
                            event.source_text = final_text
                        else:
                            event.replacement_status = "OCR_EMPTY"

                    all_events = self._merge_redundant_ocr_events(all_events)
                except Exception as exc:
                    logger.warning("OCR failed for subtitle events: %s", exc)
                    for event in all_events:
                        event.replacement_status = "OCR_FAILED"

        # 3. Translation Stage
        events_needing_translation = [e for e in all_events if e.source_text]

        if self.translation_service is not None and events_needing_translation:
            texts_to_translate = [e.source_text for e in events_needing_translation]
            try:
                translated = self.translation_service.translate_texts(
                    texts_to_translate, target_language=target_language
                )
                for e, trans in zip(events_needing_translation, translated):
                    if trans and trans.strip():
                        e.translated_text = trans.strip()
                    else:
                        e.replacement_status = "TRANSLATION_EMPTY"
            except Exception as exc:
                logger.warning("Translation failed for subtitle events: %s", exc)
                for e in events_needing_translation:
                    e.replacement_status = "TRANSLATION_FAILED"

        # 4. Fail-Safe Replacement Status Assignment
        for event in all_events:
            if event.source_text and event.translated_text:
                event.replacement_status = "READY"
                event.blur_applied = True
                event.overlay_applied = True
            else:
                if event.replacement_status == "SKIPPED":
                    if not event.source_text:
                        event.replacement_status = "OCR_EMPTY"
                    elif not event.translated_text:
                        event.replacement_status = "TRANSLATION_EMPTY"
                event.blur_applied = False
                event.overlay_applied = False
                logger.info(
                    "SubtitleEvent %s replacement status: %s (Pixels preserved)",
                    event.event_id,
                    event.replacement_status,
                )

        return all_events

    def _recover_bottom_lane_gap_events(
        self,
        events: List[SubtitleEvent],
        video_path: str,
        *,
        min_gap_seconds: float = 0.55,
        max_gap_seconds: float = 4.0,
        sample_step_seconds: float = 0.30,
    ) -> List[SubtitleEvent]:
        """Recover subtitle events that YOLO missed for a bounded bottom-lane gap.

        Recovery is intentionally conservative: both neighboring events must be in
        the bottom subtitle lane, the gap must be short, and the same OCR text must
        repeat on at least two consecutive samples. A single OCR guess never creates
        a synthetic caption.
        """
        if self.ocr_provider is None or len(events) < 2:
            return events

        import cv2

        ordered = sorted(events, key=lambda e: (e.start_time, e.end_time, e.event_id))
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return events

        fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        if fw <= 0 or fh <= 0:
            cap.release()
            return events

        candidate_gaps: list[tuple[int, float, float]] = []
        all_crops: list[Any] = []
        crop_meta: list[tuple[int, float]] = []

        for gap_index, (left, right) in enumerate(zip(ordered, ordered[1:])):
            left_cy = (left.anchor_bbox.y1 + left.anchor_bbox.y2) / 2.0
            right_cy = (right.anchor_bbox.y1 + right.anchor_bbox.y2) / 2.0
            if left_cy < fh * 0.72 or right_cy < fh * 0.72:
                continue

            gap_start = float(left.end_time)
            gap_end = float(right.start_time)
            gap_duration = gap_end - gap_start
            if gap_duration < min_gap_seconds or gap_duration > max_gap_seconds:
                continue

            # Stay away from the neighboring caption boundary by a small amount.
            # Ensure at least two samples even for a ~0.6s recoverable gap.
            inner_start = gap_start + min(0.15, gap_duration * 0.20)
            inner_end = gap_end - min(0.15, gap_duration * 0.20)
            if inner_end <= inner_start:
                continue

            times: list[float] = []
            t = inner_start
            while t <= inner_end + 1e-6:
                times.append(float(t))
                t += sample_step_seconds
            if len(times) < 2:
                times = [inner_start, inner_end]

            local_count = 0
            for ts in times:
                cap.set(cv2.CAP_PROP_POS_MSEC, ts * 1000.0)
                ok, frame = cap.read()
                if not ok or frame is None:
                    continue
                y1 = int(round(fh * 0.78))
                x1 = int(round(fw * 0.08))
                x2 = int(round(fw * 0.92))
                crop = frame[y1:fh, x1:x2]
                if crop.size == 0:
                    continue
                all_crops.append(crop)
                crop_meta.append((gap_index, ts))
                local_count += 1

            if local_count >= 2:
                candidate_gaps.append((gap_index, gap_start, gap_end))

        cap.release()
        if not all_crops or not candidate_gaps:
            return events

        try:
            texts = self.ocr_provider.read_crops(all_crops)
        except Exception as exc:
            logger.warning("Bottom-lane gap recovery OCR failed: %s", exc)
            return events

        by_gap: dict[int, list[tuple[float, str]]] = {}
        for (gap_index, ts), raw_text in zip(crop_meta, texts):
            value = self._sanitize_caption_text(
                self._select_bilingual_source_text((raw_text or "").strip())
            )
            by_gap.setdefault(gap_index, []).append((ts, value))

        def norm(text: str) -> str:
            return re.sub(r"[\s\W_]+", "", (text or "").lower(), flags=re.UNICODE)

        recovered: list[SubtitleEvent] = []
        for gap_index, gap_start, gap_end in candidate_gaps:
            samples = sorted(by_gap.get(gap_index, []), key=lambda pair: pair[0])
            if len(samples) < 2:
                continue

            # Build consecutive runs of the same OCR-normalized source text.
            runs: list[list[tuple[float, str]]] = []
            for sample in samples:
                key = norm(sample[1])
                if len(key) < 3:
                    continue
                if runs and norm(runs[-1][-1][1]) == key:
                    runs[-1].append(sample)
                else:
                    runs.append([sample])

            stable_runs = [run for run in runs if len(run) >= 2]
            if not stable_runs:
                continue

            # If there are multiple stable captions in one YOLO gap, place the
            # boundary halfway between adjacent OCR sample runs.
            for run_idx, run in enumerate(stable_runs):
                start = gap_start
                end = gap_end
                if run_idx > 0:
                    prev = stable_runs[run_idx - 1]
                    start = (prev[-1][0] + run[0][0]) / 2.0
                if run_idx + 1 < len(stable_runs):
                    nxt = stable_runs[run_idx + 1]
                    end = (run[-1][0] + nxt[0][0]) / 2.0
                if end - start < 0.20:
                    continue

                source = run[len(run) // 2][1]
                recovered.append(
                    SubtitleEvent(
                        event_id=f"GAP_RECOVERY_{gap_index:02d}_{run_idx:02d}",
                        track_id="RECOVERED_BOTTOM",
                        start_time=start,
                        end_time=end,
                        anchor_bbox=BBox(
                            x1=fw * 0.10,
                            y1=fh * 0.78,
                            x2=fw * 0.90,
                            y2=fh * 0.995,
                            frame_width=fw,
                            frame_height=fh,
                        ),
                        source_text=source,
                        split_reason="bottom_lane_gap_recovery",
                    )
                )

        if not recovered:
            return events

        logger.info(
            "Recovered %s bottom-lane subtitle event(s) across YOLO gaps.",
            len(recovered),
        )
        combined = [*events, *recovered]
        combined.sort(key=lambda e: (e.start_time, e.end_time, e.event_id))
        return combined

    @staticmethod
    def _merge_redundant_ocr_events(events: List[SubtitleEvent]) -> List[SubtitleEvent]:
        if len(events) < 2:
            return events

        def norm(text: str) -> str:
            return re.sub(r"[\s\W_]+", "", (text or "").lower(), flags=re.UNICODE)

        def union_box(a: BBox, b: BBox) -> BBox:
            return BBox(
                x1=min(a.x1, b.x1), y1=min(a.y1, b.y1),
                x2=max(a.x2, b.x2), y2=max(a.y2, b.y2),
                frame_width=a.frame_width, frame_height=a.frame_height,
            )

        ordered = sorted(events, key=lambda e: (e.start_time, e.end_time, e.event_id))
        merged: List[SubtitleEvent] = []
        for event in ordered:
            key = norm(event.source_text)
            if not key:
                merged.append(event)
                continue

            target_index = None
            for idx in range(len(merged) - 1, -1, -1):
                prev = merged[idx]
                if event.start_time - prev.end_time > 0.15:
                    break
                if norm(prev.source_text) != key:
                    continue

                overlap = min(prev.end_time, event.end_time) - max(prev.start_time, event.start_time)
                gap = max(0.0, event.start_time - prev.end_time)
                pcx = (prev.anchor_bbox.x1 + prev.anchor_bbox.x2) / 2.0
                pcy = (prev.anchor_bbox.y1 + prev.anchor_bbox.y2) / 2.0
                ecx = (event.anchor_bbox.x1 + event.anchor_bbox.x2) / 2.0
                ecy = (event.anchor_bbox.y1 + event.anchor_bbox.y2) / 2.0
                fw = max(1.0, float(prev.anchor_bbox.frame_width))
                fh = max(1.0, float(prev.anchor_bbox.frame_height))
                spatially_related = abs(pcx - ecx) / fw <= 0.15 and abs(pcy - ecy) / fh <= 0.16
                same_track_adjacent = prev.track_id == event.track_id and gap <= 0.12
                cross_track_overlap = prev.track_id != event.track_id and overlap > 0 and spatially_related
                if same_track_adjacent or cross_track_overlap:
                    target_index = idx
                    break

            if target_index is None:
                merged.append(event)
                continue

            prev = merged[target_index]
            same_track = prev.track_id == event.track_id
            trajectory = prev.bbox_trajectory + event.bbox_trajectory if same_track else []
            frame_indices = prev.bbox_frame_indices + event.bbox_frame_indices if same_track else []
            if same_track and frame_indices:
                dedup: dict[int, BBox] = {}
                for frame_index, box in sorted(zip(frame_indices, trajectory), key=lambda pair: pair[0]):
                    dedup[int(frame_index)] = box
                frame_indices = sorted(dedup)
                trajectory = [dedup[i] for i in frame_indices]

            merged[target_index] = SubtitleEvent(
                event_id=prev.event_id,
                track_id=prev.track_id if same_track else f"DEDUP:{prev.track_id}+{event.track_id}",
                start_time=min(prev.start_time, event.start_time),
                end_time=max(prev.end_time, event.end_time),
                anchor_bbox=union_box(prev.anchor_bbox, event.anchor_bbox),
                bbox_trajectory=trajectory,
                bbox_frame_indices=frame_indices,
                source_text=prev.source_text,
                split_reason="ocr_duplicate_merge",
            )

        merged.sort(key=lambda e: (e.start_time, e.end_time, e.event_id))
        return merged

    @staticmethod
    def _merge_parallel_bilingual_events(events: List[SubtitleEvent]) -> List[SubtitleEvent]:
        if len(events) < 2:
            return events

        ordered = sorted(events, key=lambda e: (e.start_time, e.end_time, e.event_id))
        used: set[int] = set()
        merged: List[SubtitleEvent] = []

        def _center(box: BBox) -> tuple[float, float]:
            return ((box.x1 + box.x2) / 2.0, (box.y1 + box.y2) / 2.0)

        def _union(a: BBox, b: BBox) -> BBox:
            return BBox(
                x1=min(a.x1, b.x1),
                y1=min(a.y1, b.y1),
                x2=max(a.x2, b.x2),
                y2=max(a.y2, b.y2),
                frame_width=a.frame_width,
                frame_height=a.frame_height,
            )

        for i, left in enumerate(ordered):
            if i in used:
                continue
            best_j = None
            best_score = -1.0
            left_dur = max(0.05, left.end_time - left.start_time)
            lx, ly = _center(left.anchor_bbox)
            lw = max(1.0, left.anchor_bbox.width)

            for j in range(i + 1, len(ordered)):
                if j in used:
                    continue
                right = ordered[j]
                if right.start_time > left.end_time + 0.15:
                    break
                if right.track_id == left.track_id:
                    continue

                overlap = min(left.end_time, right.end_time) - max(left.start_time, right.start_time)
                if overlap <= 0:
                    continue
                right_dur = max(0.05, right.end_time - right.start_time)
                overlap_ratio = overlap / min(left_dur, right_dur)
                overlap_ratio_long = overlap / max(left_dur, right_dur)
                if overlap_ratio < 0.60 or overlap_ratio_long < 0.60:
                    # Do not destructively merge a short bilingual row that only
                    # overlaps the tail of a much longer event. The old min-duration
                    # check allowed nested events to merge and then kept only the
                    # temporal intersection, silently dropping the longer event's
                    # leading/trailing subtitle coverage.
                    continue

                rx, ry = _center(right.anchor_bbox)
                h = float(max(1, left.anchor_bbox.frame_height))
                w = float(max(1, left.anchor_bbox.frame_width))
                y_sep = abs(ly - ry) / h
                x_sep = abs(lx - rx) / w
                if not (0.025 <= y_sep <= 0.16) or x_sep > 0.12:
                    continue

                x_overlap = max(
                    0.0,
                    min(left.anchor_bbox.x2, right.anchor_bbox.x2)
                    - max(left.anchor_bbox.x1, right.anchor_bbox.x1),
                )
                overlap_width_ratio = x_overlap / max(1.0, min(lw, right.anchor_bbox.width))
                if overlap_width_ratio < 0.25:
                    continue

                score = overlap_ratio + overlap_width_ratio - x_sep - y_sep
                if score > best_score:
                    best_score = score
                    best_j = j

            if best_j is None:
                merged.append(left)
                used.add(i)
                continue

            right = ordered[best_j]
            # Once two stacked rows have passed the strong temporal/spatial
            # bilingual-match checks above, preserve the UNION of their temporal
            # coverage. Using only the intersection creates visible source-text
            # leaks whenever one row is detected a few frames earlier/later than
            # the other (common for Chinese + English burned-in subtitles).
            start = min(left.start_time, right.start_time)
            end = max(left.end_time, right.end_time)
            if end <= start:
                merged.append(left)
                used.add(i)
                continue

            merged.append(
                SubtitleEvent(
                    event_id=f"{left.event_id}+{right.event_id}",
                    track_id=f"BILINGUAL:{left.track_id}+{right.track_id}",
                    start_time=start,
                    end_time=end,
                    anchor_bbox=_union(left.anchor_bbox, right.anchor_bbox),
                    bbox_trajectory=[],
                    bbox_frame_indices=[],
                    split_reason="parallel_bilingual_merge",
                )
            )
            used.add(i)
            used.add(best_j)

        merged.sort(key=lambda e: (e.start_time, e.end_time, e.event_id))
        return merged

    @staticmethod
    def _sanitize_caption_text(text: str) -> str:
        value = (text or "").strip()
        if not value:
            return ""
        # OCR often absorbs decorative bullets/dots placed near the subtitle row.
        # They are not semantic caption content and look like random punctuation
        # after translation.
        value = re.sub(r"^[\s•·●▪◦‣∙]+", "", value).strip()
        return value

    @staticmethod
    def _select_bilingual_source_text(text: str) -> str:
        value = (text or "").strip()
        if not value:
            return ""
        lines = [line.strip() for line in value.splitlines() if line.strip()]
        if len(lines) < 2:
            return value

        cjk_lines = [line for line in lines if len(re.findall(r"[\u3400-\u9fff]", line)) >= 2]
        latin_lines = [line for line in lines if len(re.findall(r"[A-Za-z]", line)) >= 3]
        if cjk_lines and latin_lines:
            # Prefer the native CJK row as semantic source. The lower English row
            # is a duplicate translation and should not become a second caption.
            return max(cjk_lines, key=lambda line: len(re.findall(r"[\u3400-\u9fff]", line)))
        return value

    @staticmethod
    def _resolve_ocr_consensus(candidates: List[str]) -> str:
        valid = [c.strip() for c in candidates if c and c.strip()]
        if not valid:
            return ""
        if len(valid) == 1:
            return valid[0]

        first = valid[0]
        # Check string duplication artifact: e.g. "拜拜" vs "拜拜拜拜" or "谢谢" vs "谢谢谢谢"
        for c in valid:
            if c != first and len(c) > 0 and len(first) > 0:
                if len(c) == 2 * len(first) and c == first + first:
                    return first
                if len(first) == 2 * len(c) and first == c + c:
                    return c

        # Majority vote
        from collections import Counter
        counts = Counter(valid)
        most_common, count = counts.most_common(1)[0]
        if count > 1:
            return most_common

        # Default to first (mid_ts crop)
        return valid[0]

    @staticmethod
    def select_render_events(events: List[SubtitleEvent]) -> List[SubtitleEvent]:
        """Collapse stacked bilingual captions to one translated overlay per lane.

        Chinese/native-script events are preferred over duplicate Latin/English
        rows. Distinct subtitle lanes remain independent.
        """
        ready = [e for e in events if e.replacement_status == "READY" and e.overlay_applied]
        if len(ready) <= 1:
            return ready

        groups: List[List[SubtitleEvent]] = []
        for event in sorted(ready, key=lambda e: (e.anchor_bbox.y1, e.anchor_bbox.x1)):
            cy = (event.anchor_bbox.y1 + event.anchor_bbox.y2) / 2.0
            h = max(1.0, float(event.anchor_bbox.frame_height))
            placed = False
            for group in groups:
                ref = group[0]
                rcy = (ref.anchor_bbox.y1 + ref.anchor_bbox.y2) / 2.0
                if abs(cy - rcy) / h <= 0.16:
                    group.append(event)
                    placed = True
                    break
            if not placed:
                groups.append([event])

        selected: List[SubtitleEvent] = []
        for group in groups:
            # Prefer native CJK semantic source over the duplicate English row;
            # otherwise prefer the event with more semantic content.
            def score(ev: SubtitleEvent) -> tuple[int, int, float]:
                cjk = len(re.findall(r"[\u3400-\u9fff]", ev.source_text or ""))
                return (1 if cjk >= 2 else 0, len(ev.source_text or ""), ev.end_time - ev.start_time)

            selected.append(max(group, key=score))
        return selected

    def render_region_events(self, frame: Any, events: List[SubtitleEvent], region: BBox) -> Any:
        """Draw a fully opaque local plate around an observed subtitle region."""
        from src.features.visual.actions.text_overlay import TextOverlayAction
        ready = [e for e in events if e.replacement_status == 'READY' and e.translated_text]
        if not ready:
            return frame
        # Source suppression and caption backing are drawn together, so text
        # wrapping cannot expose source pixels or exceed its own backing.
        return TextOverlayAction(text=ready[0].translated_text, anchor_to_bbox=True,
                                 bg_box=True, bg_color=(16, 12, 10, 255)).execute(frame, region)

    @staticmethod
    def suppress_source_subtitles(frame: Any, detections: List[Any]) -> Any:
        """Remove burned-in subtitle glyphs with a localized inpaint mask.

        Full-width blur/matte bands are visually heavy and still leave ghost glyphs.
        Instead, estimate high-contrast text strokes only inside YOLO subtitle boxes,
        dilate the mask to include outlines/shadows, and reconstruct the underlying
        pixels with Telea inpainting. A small local frosted patch is used only when
        the glyph mask is too weak to be trustworthy.
        """
        if not detections:
            return frame
        import cv2
        import numpy as np

        fh, fw = frame.shape[:2]
        boxes = []
        for detection in detections:
            bbox = detection.bbox if hasattr(detection, "bbox") else detection
            x1, y1, x2, y2 = bbox.xyxy_int()
            if x2 > x1 and y2 > y1:
                boxes.append((x1, y1, x2, y2))
        if not boxes:
            return frame

        # Bottom burned-in bilingual subtitles are the hardest case for inpainting:
        # the lower English row is often missed by YOLO on some frames and textured
        # footage can defeat glyph masks. Use one deliberate localized cinema plate
        # to hide *both* source rows completely, while keeping the rest of the frame
        # untouched. Non-bottom text still uses glyph-level inpainting below.
        bottom_boxes = [b for b in boxes if ((b[1] + b[3]) / 2.0) >= fh * 0.72]
        out = frame.copy()
        if bottom_boxes:
            min_x = min(b[0] for b in bottom_boxes)
            max_x = max(b[2] for b in bottom_boxes)
            heights = [max(1, b[3] - b[1]) for b in bottom_boxes]
            median_h = float(np.median(heights)) if heights else 42.0
            # The bottom caption uses a stable centered layout. A partial YOLO
            # detection must not pull the plate sideways and expose the other end
            # of a source line; detected extents only expand the minimum plate.
            center_x = float(fw) / 2.0
            panel_w = float(fw) * 0.72
            # The stable panel is a minimum, never a crop of detected text.
            # Long or off-center bilingual lines can exceed its fixed width.
            # Keep all source boxes inside the opaque interior, including the
            # rounded corners and feathered edge.
            edge_pad = 32
            px1 = max(0, min(int(round(center_x - panel_w / 2.0)), min_x - edge_pad))
            px2 = min(fw, max(int(round(center_x + panel_w / 2.0)), max_x + edge_pad))
            py1 = max(0, min(int(round(fh * 0.805)), min(b[1] for b in bottom_boxes) - edge_pad))
            py2 = min(fh, max(int(round(fh * 0.995)), max(b[3] for b in bottom_boxes) + edge_pad))

            alpha_mask = np.zeros((fh, fw), dtype=np.uint8)
            radius = max(10, min(22, int(round(median_h * 0.30))))
            cv2.rectangle(alpha_mask, (px1 + radius, py1), (px2 - radius, py2), 255, -1)
            cv2.rectangle(alpha_mask, (px1, py1 + radius), (px2, py2 - radius), 255, -1)
            for cx, cy in ((px1 + radius, py1 + radius), (px2 - radius, py1 + radius),
                           (px1 + radius, py2 - radius), (px2 - radius, py2 - radius)):
                cv2.circle(alpha_mask, (cx, cy), radius, 255, -1)
            alpha_mask = cv2.GaussianBlur(alpha_mask, (0, 0), 2.2).astype(np.float32) / 255.0
            alpha_mask *= 0.985
            matte = np.zeros_like(out, dtype=np.float32)
            matte[:, :, 0] = 10.0
            matte[:, :, 1] = 12.0
            matte[:, :, 2] = 16.0
            a = alpha_mask[:, :, None]
            out = np.clip(matte * a + out.astype(np.float32) * (1.0 - a), 0, 255).astype(np.uint8)

        mask = np.zeros((fh, fw), dtype=np.uint8)
        fallback_boxes: list[tuple[int, int, int, int]] = []

        for x1, y1, x2, y2 in boxes:
            if ((y1 + y2) / 2.0) >= fh * 0.72:
                continue
            h = max(1, y2 - y1)
            pad_x = max(8, int(h * 0.18))
            pad_y = max(5, int(h * 0.12))
            ax1 = max(0, x1 - pad_x)
            ay1 = max(0, y1 - pad_y)
            ax2 = min(fw, x2 + pad_x)
            ay2 = min(fh, y2 + pad_y)
            roi = frame[ay1:ay2, ax1:ax2]
            if roi.size == 0:
                continue

            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            smooth = cv2.GaussianBlur(gray, (0, 0), 2.2)
            contrast = cv2.absdiff(gray, smooth)

            # White/yellow subtitle fills are normally bright and locally sharp.
            # The second term catches antialiased/colored glyph edges that are not
            # quite bright enough for the first threshold.
            bright_text = (gray >= 142) & (contrast >= 8)
            sharp_text = contrast >= 24
            local = np.where(bright_text | sharp_text, 255, 0).astype(np.uint8)

            # Keep the mask text-like: close glyph strokes, then dilate far enough
            # to remove dark outlines/drop-shadows around the bright fill.
            close_k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            local = cv2.morphologyEx(local, cv2.MORPH_CLOSE, close_k, iterations=1)
            dilate_px = max(3, min(7, int(round(h * 0.10))))
            dilate_k = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (dilate_px * 2 + 1, dilate_px * 2 + 1)
            )
            local = cv2.dilate(local, dilate_k, iterations=1)

            coverage = float(np.count_nonzero(local)) / max(1.0, float(local.size))
            # Too little mask means the text color/contrast was not captured; too
            # much means scene texture dominated. Both are unsafe for inpainting.
            if coverage < 0.012 or coverage > 0.48:
                fallback_boxes.append((ax1, ay1, ax2, ay2))
                continue
            current = mask[ay1:ay2, ax1:ax2]
            mask[ay1:ay2, ax1:ax2] = cv2.max(current, local)

        if np.count_nonzero(mask):
            # Radius 4–5 px is enough for 720p subtitle strokes without melting
            # unrelated scene structure around the text.
            out = cv2.inpaint(out, mask, 4.5, cv2.INPAINT_TELEA)

        # Conservative fallback: local feathered blur only around the affected row,
        # never a full-width lower-third matte.
        for x1, y1, x2, y2 in fallback_boxes:
            roi = out[y1:y2, x1:x2]
            if roi.size == 0:
                continue
            blurred = cv2.GaussianBlur(roi, (0, 0), 8.0)
            alpha = 0.88
            out[y1:y2, x1:x2] = cv2.addWeighted(blurred, alpha, roi, 1.0 - alpha, 0)

        return out

    @staticmethod
    def get_source_suppression_bbox(bbox: BBox, pad_x: int = 30, pad_y: int = 20) -> BBox:
        x1, y1, x2, y2 = bbox.xyxy_int()
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(bbox.frame_width, x2 + pad_x)
        y2 = min(bbox.frame_height, y2 + pad_y)
        return BBox(x1=x1, y1=y1, x2=x2, y2=y2, frame_width=bbox.frame_width, frame_height=bbox.frame_height)

    @staticmethod
    def get_event_bbox(event: SubtitleEvent, frame_index: int) -> BBox:
        """Return a trajectory-aware bbox for the current frame.

        Missing frames are linearly interpolated between neighboring detections;
        padded/refined frames before/after the track use the nearest boundary bbox.
        """
        boxes = event.bbox_trajectory
        indices = event.bbox_frame_indices
        if not boxes or len(boxes) != len(indices):
            return event.anchor_bbox

        pos = bisect_left(indices, int(frame_index))
        if pos <= 0:
            return boxes[0]
        if pos >= len(indices):
            return boxes[-1]
        if indices[pos] == frame_index:
            return boxes[pos]

        left_i = indices[pos - 1]
        right_i = indices[pos]
        left = boxes[pos - 1]
        right = boxes[pos]
        span = max(1, right_i - left_i)
        ratio = (frame_index - left_i) / float(span)
        return BBox(
            x1=left.x1 + (right.x1 - left.x1) * ratio,
            y1=left.y1 + (right.y1 - left.y1) * ratio,
            x2=left.x2 + (right.x2 - left.x2) * ratio,
            y2=left.y2 + (right.y2 - left.y2) * ratio,
            frame_width=left.frame_width,
            frame_height=left.frame_height,
        )

    def render_frame_events(self, frame: Any, frame_index: int, events: List[SubtitleEvent]) -> Any:
        for event in events:
            # FAIL-SAFE: ONLY render Vietnamese overlay when replacement status is READY!
            if event.replacement_status == "READY" and event.overlay_applied:
                bbox = self.get_event_bbox(event, frame_index)
                frame = self.renderer.render_event(frame, event, bbox)
        return frame
