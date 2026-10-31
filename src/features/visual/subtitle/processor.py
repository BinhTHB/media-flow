"""
Subtitle Processor

Manages subtitle-specific processing with destructive replacement safety:
YOLO subtitle track -> SubtitleEventSegmenter -> SubtitleEvent[] -> OCR -> shared TranslationService -> Fail-Safe Check -> BlurAction & SubtitleRenderer.
"""

from __future__ import annotations

import logging
import re
import time
from bisect import bisect_left
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

from src.core.contracts import OCRPort
from src.core.models.bbox import BBox
from src.core.models.subtitle_event import SubtitleEvent
from src.core.models.track import Track
from src.features.translation.service import TranslationService
from src.features.visual.actions.blur import BlurAction
from src.features.visual.subtitle.boundary_refiner import SubtitleBoundaryRefiner
from src.features.visual.subtitle.event_segmenter import SubtitleEventSegmenter
from src.features.visual.subtitle.renderer import SubtitleRenderer
from src.features.translation.text_quality import (
    is_already_target_language,
    is_caption_worthy,
    is_effectively_identical,
)

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
        translation_chunk_size: int = 8,
        ocr_chunk_size: int = 16,
        ocr_retry_rounds: int = 1,
        ocr_verify_empty_rounds: int = 1,
        ocr_verify_scale: float = 2.0,
        ocr_max_workers: int = 4,
        translation_max_workers: int = 3,
    ):
        self.ocr_provider = ocr_provider
        self.translation_service = translation_service
        self.segmenter = segmenter or SubtitleEventSegmenter()
        self.boundary_refiner = boundary_refiner or SubtitleBoundaryRefiner()
        self.renderer = renderer or SubtitleRenderer()
        self.blur_action = blur_action or BlurAction(strength=25.0, padding_px=4)
        self.subtitle_layout: str = "in-place"
        self.translation_chunk_size = max(1, int(translation_chunk_size))
        self.ocr_chunk_size = max(1, int(ocr_chunk_size))
        self.ocr_retry_rounds = max(0, int(ocr_retry_rounds))
        self.ocr_verify_empty_rounds = max(0, int(ocr_verify_empty_rounds))
        self.ocr_verify_scale = max(1.0, float(ocr_verify_scale))
        self.ocr_max_workers = max(1, int(ocr_max_workers))
        self.translation_max_workers = max(1, int(translation_max_workers))

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
                self._ocr_event_crops(all_crops, crop_metadata, all_events)
                all_events = self._merge_redundant_ocr_events(all_events)

        # 2b. Evidence verification: a detector "subtitle" whose OCR came back
        # empty is usually a hallucinated region (scenery texture, watermark
        # edges). Before the pipeline concludes "no caption here", re-read each
        # empty event's crop upscaled — small or stylized burned-in text often
        # only becomes readable at 2x. Whatever is still empty after this pass
        # is treated as having no caption at all (never redacted downstream).
        if video_path and self.ocr_provider is not None and self.ocr_verify_empty_rounds > 0:
            self._verify_ocr_empty_events(all_events, video_path)

        # 2c. Caption-worthiness gate: YOLO labels on-screen counters and
        # watermark digits as "subtitle", and OCR faithfully reads them. A
        # caption contains letters; "122" does not. Dropping these here keeps
        # them out of translation and out of the rendered overlay.
        dropped_noise = 0
        for event in all_events:
            if event.source_text and not is_caption_worthy(event.source_text):
                logger.info(
                    "SubtitleEvent %s dropped as non-caption OCR noise: %r",
                    event.event_id, event.source_text,
                )
                event.source_text = ""
                event.replacement_status = "OCR_EMPTY"
                dropped_noise += 1
        if dropped_noise:
            logger.info("Subtitle: dropped %d non-caption event(s) after OCR.", dropped_noise)

        # 2d. A CJK row and its Latin twin often escape the pre-OCR geometry
        # merge (their tracks only overlap after OCR/dedup). Pair them here so
        # translation sees one caption and render draws one line at the band.
        all_events = self._merge_stacked_caption_rows(all_events)

        # 3. Translation Stage
        # Skip events whose source is already the target language: translating
        # Vietnamese into Vietnamese produced diacritic shuffles ("Kì lạ" ->
        # "Kỳ lạ") or invented text, and re-rendering it over the original
        # caption is strictly worse than leaving the pixels alone.
        already_target = [
            e for e in all_events
            if e.source_text and is_already_target_language(e.source_text, target_language)
        ]
        for event in already_target:
            event.translated_text = event.source_text
            event.replacement_status = "SOURCE_IS_TARGET"
            logger.debug(
                "SubtitleEvent %s already in target language %r; skipping translation.",
                event.event_id, target_language,
            )
        if already_target:
            logger.info(
                "Subtitle: %d/%d event(s) already in target language '%s'; translation skipped for them.",
                len(already_target), len([e for e in all_events if e.source_text]), target_language,
            )

        events_needing_translation = [
            e for e in all_events
            if e.source_text and e.replacement_status != "SOURCE_IS_TARGET"
        ]

        if self.translation_service is not None and events_needing_translation:
            self._translate_events(events_needing_translation, target_language)
            self._collapse_duplicate_translation_lines(events_needing_translation)
            all_events = self._merge_stacked_caption_rows(all_events, use_translation=True)

        # 4. Fail-Safe Replacement Status Assignment
        for event in all_events:
            if event.replacement_status == "SOURCE_IS_TARGET":
                # The caption is already in the target language. Rendering it
                # again would paint the same words over themselves (losing the
                # source styling) and translating produced diacritic shuffles.
                # Leave the original pixels untouched.
                event.translated_text = ""
                event.blur_applied = False
                event.overlay_applied = False
                logger.info(
                    "SubtitleEvent %s already in target language; original pixels preserved.",
                    event.event_id,
                )
                continue
            if event.source_text and event.translated_text:
                # A "translation" identical to its source means the provider
                # returned the input unchanged. Rendering it would paint the
                # same text over itself; leaving the original pixels is better.
                if is_effectively_identical(event.source_text, event.translated_text):
                    event.replacement_status = "TRANSLATION_UNCHANGED"
                    event.blur_applied = False
                    event.overlay_applied = False
                    logger.info(
                        "SubtitleEvent %s translation equals source; keeping original pixels.",
                        event.event_id,
                    )
                    continue
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

    def _ocr_event_crops(self, all_crops: List[Any], crop_metadata: List[Any], all_events: List[SubtitleEvent]) -> None:
        """OCR event crops in bounded batches, executed concurrently so
        per-request network latency overlaps instead of stacking. Failed
        batches get bounded retry rounds (also concurrent); persistent
        failures keep the fail-safe status for just their own events."""
        chunk_size = max(1, int(self.ocr_chunk_size))
        batches: List[tuple[List[Any], List[Any]]] = [
            (all_crops[start:start + chunk_size], crop_metadata[start:start + chunk_size])
            for start in range(0, len(all_crops), chunk_size)
        ]
        failed = self._run_ocr_batches(batches, all_events)
        for _round in range(max(0, int(self.ocr_retry_rounds))):
            if not failed:
                break
            time.sleep(5.0)
            failed = self._run_ocr_batches(failed, all_events)

    def _run_ocr_batches(
        self, batches: List[tuple[List[Any], List[Any]]], all_events: List[SubtitleEvent]
    ) -> List[tuple[List[Any], List[Any]]]:
        """Run OCR batches in parallel; each batch only touches its own
        events, so concurrent mutation of ``all_events`` stays disjoint.
        Returns the batches that failed and need a retry round."""
        if not batches:
            return []
        failed: List[tuple[List[Any], List[Any]]] = []
        workers = max(1, min(int(self.ocr_max_workers), len(batches)))
        if workers == 1:
            for crops, metadata in batches:
                if self._ocr_batch(crops, metadata, all_events):
                    failed.append((crops, metadata))
            return failed
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(self._ocr_batch, crops, metadata, all_events): (crops, metadata)
                for crops, metadata in batches
            }
            for future, batch in futures.items():
                if future.result():
                    failed.append(batch)
        return failed

    def _collapse_duplicate_translation_lines(self, events: List[SubtitleEvent]) -> None:
        """Collapse near-duplicate lines inside one translated caption.

        Merged bilingual events (Chinese row + its English twin) come back
        with one Vietnamese line per language row, and the two lines read as
        the same caption twice. A human placing subtitles writes ONE line, so
        keep the first line and drop any line whose tokens overlap an
        already-kept line by >= 0.6 Jaccard. Genuinely different lines (a
        real two-sentence caption) survive.
        """
        import unicodedata

        def _tokens(line: str) -> frozenset:
            value = line.lower().replace("đ", "d")
            value = unicodedata.normalize("NFD", value)
            value = "".join(ch for ch in value if unicodedata.category(ch) != "Mn")
            return frozenset(re.findall(r"[a-z0-9]{2,}", value))

        for event in events:
            text = event.translated_text or ""
            if "\n" not in text:
                continue
            kept: List[str] = []
            kept_tokens: List[frozenset] = []
            for line in text.split("\n"):
                line = line.strip()
                if not line:
                    continue
                tk = _tokens(line)
                if tk and any(
                    len(k & tk) / max(1, len(k | tk)) >= 0.6 for k in kept_tokens
                ):
                    continue
                kept.append(line)
                kept_tokens.append(tk)
            event.translated_text = "\n".join(kept)

    def _verify_ocr_empty_events(self, events: List[SubtitleEvent], video_path: str) -> None:
        """Re-OCR empty/failed events once from an upscaled crop.

        Small or low-contrast burned-in text often survives the first pass at
        native crop resolution but reads fine at 2x. Events that recover here
        rejoin the normal translate/READY flow; the rest keep their empty
        status and are deliberately NOT redacted downstream (the evidence
        rule: never paint without text evidence).
        """
        empty = [
            ev for ev in events
            if ev.replacement_status in {"OCR_EMPTY", "OCR_FAILED"} and not ev.source_text
        ]
        if not empty:
            return
        import cv2

        cap = cv2.VideoCapture(video_path)
        crops: List[Any] = []
        metadata: List[Any] = []
        try:
            for ev in empty:
                mid_ts = (ev.start_time + ev.end_time) / 2.0
                cap.set(cv2.CAP_PROP_POS_MSEC, mid_ts * 1000.0)
                ret, frame_img = cap.read()
                if not ret or frame_img is None:
                    continue
                x1, y1, x2, y2 = ev.anchor_bbox.xyxy_int()
                crop = frame_img[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                if self.ocr_verify_scale > 1.0:
                    crop = cv2.resize(
                        crop, None, fx=self.ocr_verify_scale, fy=self.ocr_verify_scale,
                        interpolation=cv2.INTER_CUBIC,
                    )
                crops.append(crop)
                metadata.append((ev, 0))
        finally:
            cap.release()
        chunk_size = max(1, int(self.ocr_chunk_size))
        self._run_ocr_batches(
            [(crops[start:start + chunk_size], metadata[start:start + chunk_size])
             for start in range(0, len(crops), chunk_size)],
            events,
        )

    def _ocr_batch(self, crops: List[Any], metadata: List[Any], all_events: List[SubtitleEvent]) -> bool:
        """OCR one batch. Returns True when the batch failed and keeps its
        fail-safe status."""
        try:
            texts = self.ocr_provider.read_crops(crops)
        except Exception as exc:
            logger.warning("OCR failed for %d crops: %s", len(crops), exc)
            for ev, _c_idx in metadata:
                if not ev.source_text:
                    ev.replacement_status = "OCR_FAILED"
            return True
        event_candidates: Dict[str, List[str]] = {}
        for (ev, _c_idx), text in zip(metadata, texts):
            event_candidates.setdefault(ev.event_id, []).append(text.strip() if text else "")

        touched = {ev.event_id for ev, _c_idx in metadata}
        for event in all_events:
            if event.event_id not in touched or event.source_text:
                # Untouched events keep their status; gap-recovery-confirmed
                # source text is never overwritten.
                continue
            cands = event_candidates.get(event.event_id, [])
            final_text = self._resolve_ocr_consensus(cands)
            final_text = self._select_bilingual_source_text(final_text)
            final_text = self._sanitize_caption_text(final_text)
            if final_text:
                event.source_text = final_text
                # Clear a stale failure status from a previous round; the
                # fail-safe pass promotes translated events to READY.
                event.replacement_status = "SKIPPED"
            else:
                event.replacement_status = "OCR_EMPTY"
        return False

    def _translate_events(self, events: List[SubtitleEvent], target_language: str) -> None:
        """Translate events in bounded chunks, executed concurrently so
        per-request network latency overlaps instead of stacking. One failing
        provider call cannot discard the whole batch; failed chunks keep the
        fail-safe status and their source pixels are preserved. Failed chunks
        get one bounded second round because transient outages often recover
        within the same job but outlast the per-call retry window."""
        chunk_size = max(1, int(self.translation_chunk_size))
        chunks: List[List[SubtitleEvent]] = [
            events[start:start + chunk_size]
            for start in range(0, len(events), chunk_size)
        ]
        failed_chunks = self._run_translation_chunks(chunks, target_language)
        if failed_chunks:
            time.sleep(5.0)
            self._run_translation_chunks(failed_chunks, target_language)

    def _run_translation_chunks(
        self, chunks: List[List[SubtitleEvent]], target_language: str
    ) -> List[List[SubtitleEvent]]:
        """Translate chunks in parallel; each chunk only mutates its own
        events. Returns the chunks that failed and need the retry round."""
        if not chunks:
            return []
        failed: List[List[SubtitleEvent]] = []
        workers = max(1, min(int(self.translation_max_workers), len(chunks)))
        if workers == 1:
            for chunk in chunks:
                if self._translate_chunk(chunk, target_language):
                    failed.append(chunk)
            return failed
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(self._translate_chunk, chunk, target_language): chunk
                for chunk in chunks
            }
            for future, chunk in futures.items():
                if future.result():
                    failed.append(chunk)
        return failed

    def _translate_chunk(self, chunk: List[SubtitleEvent], target_language: str) -> bool:
        """Translate one chunk. Returns True when the chunk failed and keeps
        its fail-safe status."""
        try:
            translated = self.translation_service.translate_texts(
                [e.source_text for e in chunk], target_language=target_language
            )
        except Exception as exc:
            logger.warning("Translation failed for %d subtitle events: %s", len(chunk), exc)
            for event in chunk:
                event.replacement_status = "TRANSLATION_FAILED"
            return True
        if len(translated) != len(chunk):
            logger.warning(
                "Translation returned %d results for %d subtitle events; unpaired events keep their fail-safe status.",
                len(translated), len(chunk),
            )
        for event, trans in zip(chunk, translated):
            if trans and trans.strip():
                event.translated_text = trans.strip()
                # Clear a stale failure status from a previous round; the
                # fail-safe pass promotes translated events to READY.
                event.replacement_status = "SKIPPED"
            else:
                event.replacement_status = "TRANSLATION_EMPTY"
        return False

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

        candidate_gaps: list[tuple[int, float, float, SubtitleEvent, SubtitleEvent]] = []
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
                candidate_gaps.append((gap_index, gap_start, gap_end, left, right))

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
        for gap_index, gap_start, gap_end, left, right in candidate_gaps:
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
                        anchor_bbox=self._recovered_anchor_bbox(left, right, fw, fh),
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
    def _recovered_anchor_bbox(
        left: SubtitleEvent, right: SubtitleEvent, frame_width: int, frame_height: int
    ) -> BBox:
        """Size a recovered caption from its neighbors, not the wide OCR crop.

        The OCR window is 80% of the frame so a missed line can still be read.
        That window is not a render box: using it painted a cinema-wide black
        bar over the scene (seen at 4s on the 30s proof clip).
        """
        a, b = left.anchor_bbox, right.anchor_bbox
        x1 = min(a.x1, b.x1)
        x2 = max(a.x2, b.x2)
        y1 = min(a.y1, b.y1)
        y2 = max(a.y2, b.y2)
        height = max(1.0, y2 - y1)
        # A single-row neighbor pair still hides a two-row bilingual caption.
        if height < frame_height * 0.10:
            y1 = min(y1, y2 - frame_height * 0.12)
        # Neighbors are often a clipped English chip. The missed caption is the
        # full bilingual plate, so grow toward a centered band and cap at 68%.
        union_w = max(1.0, x2 - x1)
        target_w = min(frame_width * 0.68, max(union_w * 1.28, frame_width * 0.52))
        cx = (x1 + x2) / 2.0
        pad_y = max(10.0, height * 0.10)
        return BBox.clamped(
            cx - target_w / 2.0,
            y1 - pad_y,
            cx + target_w / 2.0,
            y2 + pad_y,
            frame_width=frame_width,
            frame_height=frame_height,
        )

    def _merge_stacked_caption_rows(
        self,
        events: List[SubtitleEvent],
        *,
        use_translation: bool = False,
    ) -> List[SubtitleEvent]:
        """Merge a CJK row with its Latin twin after OCR (or after translation).

        Pre-OCR geometry merge misses pairs whose tracks only overlap once
        fragments are deduped. Left unmerged, they render two Vietnamese
        captions: one on the Chinese row (mid-frame) and one on the English
        row, with the source line still visible (seen at 12s).
        """
        if len(events) < 2:
            return events

        def _tokens(text: str) -> frozenset[str]:
            import unicodedata

            value = (text or "").lower().replace("đ", "d")
            value = unicodedata.normalize("NFD", value)
            value = "".join(ch for ch in value if unicodedata.category(ch) != "Mn")
            return frozenset(re.findall(r"[a-z0-9]{2,}", value))

        def _cjk(text: str) -> bool:
            return len(re.findall(r"[\u3400-\u9fff]", text or "")) >= 2

        def _latin(text: str) -> bool:
            return len(re.findall(r"[A-Za-z]", text or "")) >= 3 and not _cjk(text)

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

        def _same_meaning(left: SubtitleEvent, right: SubtitleEvent) -> bool:
            if _cjk(left.source_text) and _latin(right.source_text):
                return True
            if _latin(left.source_text) and _cjk(right.source_text):
                return True
            if not use_translation:
                return False
            a, b = _tokens(left.translated_text), _tokens(right.translated_text)
            if not a or not b:
                return False
            return len(a & b) / max(1, len(a | b)) >= 0.55

        ordered = sorted(events, key=lambda e: (e.start_time, e.end_time, e.event_id))
        used: set[int] = set()
        merged: List[SubtitleEvent] = []

        for i, left in enumerate(ordered):
            if i in used:
                continue
            best_j = None
            best_score = -1.0
            left_dur = max(0.05, left.end_time - left.start_time)
            lx, ly = _center(left.anchor_bbox)
            lw = max(1.0, left.anchor_bbox.width)
            fh = float(max(1, left.anchor_bbox.frame_height))
            fw = float(max(1, left.anchor_bbox.frame_width))

            for j in range(i + 1, len(ordered)):
                if j in used:
                    continue
                right = ordered[j]
                if right.start_time > left.end_time + 0.12:
                    break
                overlap = min(left.end_time, right.end_time) - max(left.start_time, right.start_time)
                if overlap <= 0:
                    continue
                right_dur = max(0.05, right.end_time - right.start_time)
                overlap_min = overlap / min(left_dur, right_dur)
                overlap_long = overlap / max(left_dur, right_dur)
                # Complementary scripts may share only the shorter row's span.
                if overlap_min < 0.55 or overlap_long < 0.40:
                    continue
                if not _same_meaning(left, right):
                    continue

                rx, ry = _center(right.anchor_bbox)
                y_sep = abs(ly - ry) / fh
                x_sep = abs(lx - rx) / fw
                if not (0.015 <= y_sep <= 0.18) or x_sep > 0.14:
                    continue
                x_overlap = max(
                    0.0,
                    min(left.anchor_bbox.x2, right.anchor_bbox.x2)
                    - max(left.anchor_bbox.x1, right.anchor_bbox.x1),
                )
                width_ratio = x_overlap / max(1.0, min(lw, right.anchor_bbox.width))
                if width_ratio < 0.25:
                    continue
                score = overlap_min + width_ratio - x_sep - y_sep
                if score > best_score:
                    best_score = score
                    best_j = j

            if best_j is None:
                merged.append(left)
                used.add(i)
                continue

            right = ordered[best_j]
            cjk_event, other = (left, right) if _cjk(left.source_text) else (right, left)
            source = self._select_bilingual_source_text(
                f"{left.source_text}\n{right.source_text}"
            ) or cjk_event.source_text
            translated = (cjk_event.translated_text or other.translated_text or "").strip()
            merged.append(
                SubtitleEvent(
                    event_id=f"{left.event_id}+{right.event_id}",
                    track_id=f"STACKED:{left.track_id}+{right.track_id}",
                    start_time=min(left.start_time, right.start_time),
                    end_time=max(left.end_time, right.end_time),
                    anchor_bbox=_union(left.anchor_bbox, right.anchor_bbox),
                    bbox_trajectory=[],
                    bbox_frame_indices=[],
                    source_text=source,
                    translated_text=translated,
                    replacement_status=cjk_event.replacement_status or left.replacement_status,
                    split_reason="stacked_caption_merge",
                )
            )
            used.add(i)
            used.add(best_j)

        merged.sort(key=lambda e: (e.start_time, e.end_time, e.event_id))
        return merged

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
                # A gap of about one line or more is a second caption, not the
                # English row of the same sentence.
                row_gap = max(
                    0.0,
                    max(left.anchor_bbox.y1, right.anchor_bbox.y1)
                    - min(left.anchor_bbox.y2, right.anchor_bbox.y2),
                )
                shorter = min(left.anchor_bbox.height, right.anchor_bbox.height)
                if row_gap > 0.6 * max(1.0, shorter):
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
        """Collapse duplicate caption tracks to one translated overlay per
        visual line.

        Events whose anchor boxes vertically overlap are the same on-screen
        line seen through multiple tracks (chip + rescue + full line): the
        longest, CJK-richest event wins. Distinct stacked rows (a title row
        above a content row) have disjoint y-ranges and stay independent —
        their translations are different meanings, not duplicates.
        """
        ready = [e for e in events if e.replacement_status == "READY" and e.overlay_applied]
        if len(ready) <= 1:
            return ready

        groups: List[List[SubtitleEvent]] = []
        for event in sorted(ready, key=lambda e: (e.anchor_bbox.y1, e.anchor_bbox.x1)):
            placed = False
            for group in groups:
                ref = group[0]
                if event.anchor_bbox.y1 < ref.anchor_bbox.y2 - 2 and ref.anchor_bbox.y1 < event.anchor_bbox.y2 - 2:
                    group.append(event)
                    placed = True
                    break
            if not placed:
                groups.append([event])

        selected: List[SubtitleEvent] = []
        for group in groups:
            # Pick deterministically so overlapping pre/post-roll spans cannot
            # swap the rendered caption mid-way: the longest event wins (a
            # short duplicate fragment appearing mid-span no longer takes
            # over); CJK semantic source and content length only break ties
            # between equal-length full lines.
            def score(ev: SubtitleEvent) -> tuple[float, int, int, str]:
                cjk = len(re.findall(r"[\u3400-\u9fff]", ev.source_text or ""))
                return (
                    ev.end_time - ev.start_time,
                    1 if cjk >= 2 else 0,
                    len(ev.source_text or ""),
                    ev.event_id,
                )

            selected.append(max(group, key=score))
        return selected

    def render_region_events(self, frame: Any, events: List[SubtitleEvent], region: BBox, bg_box: bool = True) -> Any:
        """Draw a local plate around an observed subtitle region."""
        from src.features.visual.actions.text_overlay import TextOverlayAction
        ready = [e for e in events if e.replacement_status == 'READY' and e.translated_text]
        if not ready:
            return frame
        # Source suppression and caption backing are drawn together, so text
        # wrapping cannot expose source pixels or exceed its own backing.
        # Text sits in the middle of this line's box. Each row keeps its own
        # box, so a tall row and a short row do not share one baseline.
        return TextOverlayAction(
            text=ready[0].translated_text,
            anchor_to_bbox=True,
            vertical_anchor="center",
            bg_box=bg_box,
            bg_color=(16, 12, 10, 255),
        ).execute(frame, region)

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

        # One plate per box. A second caption row a line below the name must
        # not stretch the name's plate down across the costume between them.
        # The caller already pads the trajectory box; a few more pixels cover
        # the stroke. A 32px union pad painted an empty black margin.
        bottom_boxes = [b for b in boxes if ((b[1] + b[3]) / 2.0) >= fh * 0.72]
        out = frame.copy()
        edge_pad = 4
        for x1, y1, x2, y2 in bottom_boxes:
            box_h = max(1, y2 - y1)
            px1 = max(0, x1 - edge_pad)
            px2 = min(fw, x2 + edge_pad)
            py1 = max(0, y1 - edge_pad)
            py2 = min(fh, y2 + edge_pad)
            if px2 - px1 < 4 or py2 - py1 < 4:
                continue

            alpha_mask = np.zeros((fh, fw), dtype=np.uint8)
            radius = max(4, min(14, int(round(box_h * 0.18))))
            radius = min(radius, (px2 - px1) // 2, (py2 - py1) // 2)
            cv2.rectangle(alpha_mask, (px1 + radius, py1), (px2 - radius, py2), 255, -1)
            cv2.rectangle(alpha_mask, (px1, py1 + radius), (px2, py2 - radius), 255, -1)
            for cx, cy in ((px1 + radius, py1 + radius), (px2 - radius, py1 + radius),
                           (px1 + radius, py2 - radius), (px2 - radius, py2 - radius)):
                cv2.circle(alpha_mask, (cx, cy), radius, 255, -1)
            alpha_mask = cv2.GaussianBlur(alpha_mask, (0, 0), 1.2).astype(np.float32) / 255.0
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

    def render_frame_events(
        self,
        frame: Any,
        frame_index: int,
        events: List[SubtitleEvent],
        anchor_to_bbox: Optional[bool] = None,
    ) -> Any:
        if anchor_to_bbox is None:
            anchor_to_bbox = getattr(self, "subtitle_layout", "in-place") in {"in-place", "temporal"}
        for event in events:
            # FAIL-SAFE: ONLY render Vietnamese overlay when replacement status is READY!
            if event.replacement_status == "READY" and event.overlay_applied:
                bbox = self.get_event_bbox(event, frame_index)
                frame = self.renderer.render_event(
                    frame, event, bbox, anchor_to_bbox=anchor_to_bbox
                )
        return frame
