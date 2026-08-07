"""
Subtitle Processor

Manages subtitle-specific processing with destructive replacement safety:
YOLO subtitle track -> SubtitleEventSegmenter -> SubtitleEvent[] -> OCR -> shared TranslationService -> Fail-Safe Check -> BlurAction & SubtitleRenderer.
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional

from src.core.contracts import OCRPort
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

        # 2. OCR Stage (Representative Crop selection from stable middle of event)
        if self.ocr_provider is not None and video_path:
            import cv2

            cap = cv2.VideoCapture(video_path)
            for event in all_events:
                # Pick stable middle frame timestamp
                mid_ts = (event.start_time + event.end_time) / 2.0
                cap.set(cv2.CAP_PROP_POS_MSEC, mid_ts * 1000.0)
                ret, frame_img = cap.read()

                if ret and frame_img is not None:
                    x1, y1, x2, y2 = event.anchor_bbox.xyxy_int()
                    crop = frame_img[y1:y2, x1:x2]
                    if crop.size > 0:
                        try:
                            texts = self.ocr_provider.read_crops([crop])
                            if texts and texts[0].strip():
                                event.source_text = texts[0].strip()
                            else:
                                event.replacement_status = "OCR_EMPTY"
                        except Exception as exc:
                            logger.warning("OCR failed for event %s: %s", event.event_id, exc)
                            event.replacement_status = "OCR_FAILED"
                    else:
                        event.replacement_status = "OCR_EMPTY"
                else:
                    event.replacement_status = "OCR_FAILED"

            cap.release()

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

    def render_frame_events(self, frame: Any, frame_index: int, events: List[SubtitleEvent]) -> Any:
        for event in events:
            # FAIL-SAFE: ONLY blur and render overlay when replacement status is READY!
            if event.replacement_status == "READY" and event.blur_applied and event.overlay_applied:
                # Apply blur to original subtitle region
                frame = self.blur_action.execute(frame, event.anchor_bbox)
                # Render translated text overlay at source position
                frame = self.renderer.render_event(frame, event, event.anchor_bbox)
            else:
                # PRESERVE original source pixels!
                pass
        return frame
