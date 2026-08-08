"""
Subtitle Processor

Manages subtitle-specific processing with destructive replacement safety:
YOLO subtitle track -> SubtitleEventSegmenter -> SubtitleEvent[] -> OCR -> shared TranslationService -> Fail-Safe Check -> BlurAction & SubtitleRenderer.
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional

from src.core.contracts import OCRPort
from src.core.models.bbox import BBox
from src.core.models.subtitle_event import SubtitleEvent
from src.core.models.track import Track
from src.features.translation.service import TranslationService
from src.features.visual.actions.blur import BlurAction
from src.features.visual.subtitle.event_segmenter import SubtitleEventSegmenter
from src.features.visual.subtitle.renderer import SubtitleRenderer

logger = logging.getLogger(__name__)


class SubtitleProcessor:
    def __init__(
        self,
        ocr_provider: Optional[OCRPort] = None,
        translation_service: Optional[TranslationService] = None,
        segmenter: Optional[SubtitleEventSegmenter] = None,
        renderer: Optional[SubtitleRenderer] = None,
        blur_action: Optional[BlurAction] = None,
    ):
        self.ocr_provider = ocr_provider
        self.translation_service = translation_service
        self.segmenter = segmenter or SubtitleEventSegmenter()
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
            events = self.segmenter.segment_track(track)
            all_events.extend(events)

        if not all_events:
            return []

        # 2. OCR Stage (Multi-candidate crop selection & consensus)
        if self.ocr_provider is not None and video_path:
            import cv2

            cap = cv2.VideoCapture(video_path)
            all_crops = []
            crop_metadata = []

            for event in all_events:
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
                        cands = event_candidates.get(event.event_id, [])
                        final_text = self._resolve_ocr_consensus(cands)
                        if final_text:
                            event.source_text = final_text
                        else:
                            event.replacement_status = "OCR_EMPTY"
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
    def get_source_suppression_bbox(bbox: BBox, pad_x: int = 18, pad_y: int = 14) -> BBox:
        x1, y1, x2, y2 = bbox.xyxy_int()
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(bbox.frame_width, x2 + pad_x)
        y2 = min(bbox.frame_height, y2 + pad_y)
        return BBox(x1=x1, y1=y1, x2=x2, y2=y2, frame_width=bbox.frame_width, frame_height=bbox.frame_height)

    def render_frame_events(self, frame: Any, frame_index: int, events: List[SubtitleEvent]) -> Any:
        for event in events:
            # FAIL-SAFE: ONLY blur and render overlay when replacement status is READY!
            if event.replacement_status == "READY" and event.blur_applied and event.overlay_applied:
                # Calculate expanded source suppression region from Chinese source bbox
                suppression_bbox = self.get_source_suppression_bbox(event.anchor_bbox, pad_x=18, pad_y=14)
                # 1. Apply Gaussian Blur over entire source suppression region
                frame = self.blur_action.execute(frame, suppression_bbox, padding_px=0)
                # 2. Render dark backdrop box + Vietnamese text overlay over source suppression region
                frame = self.renderer.render_event(frame, event, suppression_bbox)
            else:
                # PRESERVE original source pixels!
                pass
        return frame
