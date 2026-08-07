"""
Lazy Enrichment Service

Manages lazy execution of expensive downstream providers (Gemini OCR, Gemini Translation,
Whisper ASR, Image Asset resolution).

Ensures expensive dependencies are NEVER loaded or invoked unless an Action Plan explicitly
requires them (e.g., action_type == replace_text or replace_image).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.domain.entities.action_plan import RenderingPlan, TrackActionPlan
from src.domain.entities.canonical_detection import ClassAwareTrack
from src.domain.entities.processing_policy import ActionType

logger = logging.getLogger(__name__)


class LazyEnrichmentService:
    def __init__(
        self,
        translation_provider_name: str = "gemini",
        ocr_provider_name: str = "hybrid_gemini",
        speech_provider_name: str = "faster-whisper",
        target_language: str = "vi",
    ):
        self.translation_provider_name = translation_provider_name
        self.ocr_provider_name = ocr_provider_name
        self.speech_provider_name = speech_provider_name
        self.target_language = target_language

    def enrich_plan(
        self,
        rendering_plan: RenderingPlan,
        tracks: List[ClassAwareTrack],
        input_video_path: str,
        device: str = "cuda",
    ) -> RenderingPlan:
        # 1. Identify which actions need replacement text or replacement image
        text_tracks = [
            act for act in rendering_plan.actions
            if act.action_type == ActionType.REPLACE_TEXT.value
        ]
        image_tracks = [
            act for act in rendering_plan.actions
            if act.action_type == ActionType.REPLACE_IMAGE.value
        ]

        if not text_tracks and not image_tracks:
            logger.info("LazyEnrichment: No tracks require text/image enrichment. Skipping OCR, Translation, and STT.")
            return rendering_plan

        # 2. Enrich text replacement tracks if present
        if text_tracks:
            logger.info("LazyEnrichment: Enriching %d tracks with OCR & Translation.", len(text_tracks))
            self._enrich_text_replacement(text_tracks, tracks, input_video_path, device, rendering_plan)

        # 3. Enrich image replacement tracks if present
        if image_tracks:
            logger.info("LazyEnrichment: Validating %d image replacement tracks.", len(image_tracks))
            self._enrich_image_replacement(image_tracks)

        return rendering_plan

    def _enrich_text_replacement(
        self,
        text_action_plans: List[TrackActionPlan],
        tracks: List[ClassAwareTrack],
        input_video_path: str,
        device: str,
        rendering_plan: RenderingPlan,
    ) -> None:
        # Check API key if Gemini is required for text replacement
        if self.translation_provider_name == "gemini" or self.ocr_provider_name in ("hybrid_gemini", "gemini_vision"):
            if not os.getenv("GEMINI_API_KEY"):
                raise RuntimeError("BLOCKED BY EXTERNAL CREDENTIAL: GEMINI_API_KEY is required for replace_text action.")

        from src.interfaces.config.config_loader import ConfigLoader
        from src.infrastructure.factory.provider_factory import ProviderFactory
        from src.infrastructure.registry.provider_registry import ProviderRegistry

        loader = ConfigLoader()
        app_config = loader.load()
        registry = ProviderRegistry()
        factory = ProviderFactory(
            config={"defaults": app_config.defaults, "providers": app_config.providers},
            registry=registry,
            gemini_config=app_config.gemini,
            gemini_api_key=os.getenv("GEMINI_API_KEY"),
        )

        ocr_provider = factory.create_ocr_provider(name=self.ocr_provider_name)
        translation_provider = factory.create_translation_provider(name=self.translation_provider_name)

        # Create track mapping
        track_map = {t.track_id: t for t in tracks}

        # Perform OCR & Translation for each text replacement action
        for act in text_action_plans:
            track = track_map.get(act.track_id)
            if not track or not track.groups:
                continue

            # Read text via OCR
            text_source = act.params.get("text_source", "ocr")
            raw_text = ""

            if text_source in ("ocr", "hybrid"):
                mid_group = track.groups[len(track.groups) // 2]
                from src.domain.entities.subtitle_region import SubtitleRegion
                from src.domain.entities.frame import Frame

                region = SubtitleRegion(
                    start_time=track.start_time,
                    end_time=track.end_time,
                    x=mid_group.union_bbox_normalized[0],
                    y=mid_group.union_bbox_normalized[1],
                    width=mid_group.union_bbox_normalized[2] - mid_group.union_bbox_normalized[0],
                    height=mid_group.union_bbox_normalized[3] - mid_group.union_bbox_normalized[1],
                    confidence=track.mean_confidence,
                    track_id=track.track_id,
                )

                if hasattr(ocr_provider, "read_confirmed_tracks"):
                    import cv2
                    cap = cv2.VideoCapture(input_video_path)
                    frame_pos_ms = mid_group.timestamp_sec * 1000.0
                    cap.set(cv2.CAP_PROP_POS_MSEC, frame_pos_ms)
                    ret, frame_img = cap.read()
                    cap.release()

                    if ret and frame_img is not None:
                        frames_dir = Path("output") / rendering_plan.job_id / "frames"
                        frames_dir.mkdir(parents=True, exist_ok=True)
                        f_path = str(frames_dir / f"frame_{mid_group.frame_index:06d}.jpg")
                        cv2.imwrite(f_path, frame_img)
                        frames = [Frame(timestamp=mid_group.timestamp_sec, image_path=f_path)]
                        res = ocr_provider.read_confirmed_tracks(
                            [region], frames, rendering_plan.video_width, rendering_plan.video_height
                        )
                        if res and len(res) > 0:
                            raw_text = res[0].text

            if raw_text and hasattr(translation_provider, "translate"):
                from src.domain.entities.segment import Segment
                seg = Segment(id=f"seg_{act.track_id}", start=act.start_time, end=act.end_time, text=raw_text)
                translated = translation_provider.translate([seg], self.target_language)
                if translated and len(translated) > 0:
                    act.replacement_text = translated[0].translated_text or raw_text
                else:
                    act.replacement_text = raw_text
            elif raw_text:
                act.replacement_text = raw_text

    def _enrich_image_replacement(self, image_action_plans: List[TrackActionPlan]) -> None:
        for act in image_action_plans:
            asset_path = act.replacement_image_path
            if not asset_path or not os.path.exists(asset_path):
                raise FileNotFoundError(f"Image asset for replace_image action not found at: {asset_path}")
