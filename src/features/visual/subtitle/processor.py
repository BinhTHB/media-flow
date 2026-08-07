"""
Subtitle Processor

Manages subtitle-specific processing:
YOLO subtitle track -> SubtitleEventSegmenter -> SubtitleEvent[] -> OCR -> shared TranslationService -> BlurAction -> SubtitleRenderer.
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

        # 2. Run OCR if OCR provider is present
        if self.ocr_provider is not None:
            # Crop representative evidence frames and OCR
            for event in all_events:
                if video_path and event.anchor_bbox:
                    import cv2
                    cap = cv2.VideoCapture(video_path)
                    cap.set(cv2.CAP_PROP_POS_MSEC, event.start_time * 1000.0)
                    ret, frame_img = cap.read()
                    cap.release()

                    if ret and frame_img is not None:
                        x1, y1, x2, y2 = event.anchor_bbox.xyxy_int()
                        crop = frame_img[y1:y2, x1:x2]
                        if crop.size > 0:
                            texts = self.ocr_provider.read_crops([crop])
                            if texts:
                                event.source_text = texts[0]

        # 3. Run Shared Translation
        if self.translation_service is not None:
            texts_to_translate = [e.source_text for e in all_events if e.source_text]
            if texts_to_translate:
                translated = self.translation_service.translate_texts(
                    texts_to_translate, target_language=target_language
                )
                trans_idx = 0
                for e in all_events:
                    if e.source_text:
                        e.translated_text = translated[trans_idx]
                        trans_idx += 1

        return all_events

    def render_frame_events(self, frame: Any, frame_index: int, events: List[SubtitleEvent]) -> Any:
        for event in events:
            # Apply blur to original subtitle region
            frame = self.blur_action.execute(frame, event.anchor_bbox)
            # Render translated text overlay at source position
            frame = self.renderer.render_event(frame, event, event.anchor_bbox)
        return frame
