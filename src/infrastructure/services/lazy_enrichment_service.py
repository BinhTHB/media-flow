"""
Lazy Enrichment Service

Manages lazy execution of expensive downstream providers (Gemini OCR, Gemini Translation,
Whisper ASR, Image Asset resolution).

Ensures expensive dependencies are NEVER loaded or invoked unless an Action Plan explicitly
requires them (e.g. overlay action == replace_text or replace_image).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

from src.domain.entities.action_plan import RenderingPlan, TrackActionPlan
from src.domain.entities.canonical_detection import ClassAwareTrack

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
        # Collect active overlay targets
        text_overlays: List[Tuple[TrackActionPlan, Dict[str, Any]]] = []
        image_overlays: List[Tuple[TrackActionPlan, Dict[str, Any]]] = []

        for act in rendering_plan.actions:
            if act.status != "active":
                continue
            for o in act.overlays:
                act_type = o.get("action")
                if act_type == "replace_text":
                    text_overlays.append((act, o))
                elif act_type == "replace_image":
                    image_overlays.append((act, o))

        if not text_overlays and not image_overlays:
            logger.info("LazyEnrichment: No active overlays require enrichment. Skipping OCR, Translation, and STT.")
            return rendering_plan

        if text_overlays:
            logger.info("LazyEnrichment: Enriching %d text overlays.", len(text_overlays))
            self._enrich_text_overlays(text_overlays, tracks, input_video_path, device, rendering_plan)

        if image_overlays:
            logger.info("LazyEnrichment: Validating %d image overlays.", len(image_overlays))
            self._enrich_image_overlays(image_overlays)

        return rendering_plan

    def _enrich_text_overlays(
        self,
        text_overlays: List[Tuple[TrackActionPlan, Dict[str, Any]]],
        tracks: List[ClassAwareTrack],
        input_video_path: str,
        device: str,
        rendering_plan: RenderingPlan,
    ) -> None:
        needs_ocr = any(o.get("text_source", "ocr") in ("ocr", "hybrid") for _, o in text_overlays)
        needs_stt = any(o.get("text_source") in ("stt", "hybrid") for _, o in text_overlays)

        if (needs_ocr or needs_stt) and self.translation_provider_name == "gemini":
            if not os.getenv("GEMINI_API_KEY"):
                raise RuntimeError("BLOCKED BY EXTERNAL CREDENTIAL: GEMINI_API_KEY is required for replace_text overlay action.")

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

        translation_provider = factory.create_translation_provider(name=self.translation_provider_name)
        ocr_provider = factory.create_ocr_provider(name=self.ocr_provider_name) if needs_ocr else None
        speech_provider = factory.create_speech_provider(name=self.speech_provider_name) if needs_stt else None

        speech_segments = []
        if needs_stt and speech_provider is not None:
            video_provider = factory.create_video_provider("ffmpeg")
            audio_path = video_provider.extract_audio(input_video_path)
            speech_segments = speech_provider.transcribe(audio_path, language="auto")

        track_map = {t.track_id: t for t in tracks}

        for act, overlay in text_overlays:
            track = track_map.get(act.track_id)
            if not track or not track.groups:
                continue

            text_source = overlay.get("text_source", "ocr")
            raw_text = ""

            if text_source in ("ocr", "hybrid") and ocr_provider is not None:
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
                    cap.set(cv2.CAP_PROP_POS_MSEC, mid_group.timestamp_sec * 1000.0)
                    ret, frame_img = cap.read()
                    cap.release()

                    if ret and frame_img is not None:
                        job_frames_dir = Path("output") / "jobs" / rendering_plan.job_id / "frames"
                        job_frames_dir.mkdir(parents=True, exist_ok=True)
                        f_path = str(job_frames_dir / f"frame_{mid_group.frame_index:06d}.jpg")
                        cv2.imwrite(f_path, frame_img)
                        frames = [Frame(timestamp=mid_group.timestamp_sec, image_path=f_path)]
                        res = ocr_provider.read_confirmed_tracks(
                            [region], frames, rendering_plan.video_width, rendering_plan.video_height
                        )
                        if res and len(res) > 0:
                            raw_text = res[0].text

            if (text_source == "stt" or (text_source == "hybrid" and not raw_text)) and speech_segments:
                overlapping = [
                    s for s in speech_segments
                    if max(s.start, act.start_time) < min(s.end, act.end_time)
                ]
                if overlapping:
                    raw_text = " ".join(s.text for s in overlapping)

            target_lang = overlay.get("target_language", self.target_language)
            if raw_text and hasattr(translation_provider, "translate"):
                from src.domain.entities.segment import Segment
                seg = Segment(id=f"seg_{act.track_id}", start=act.start_time, end=act.end_time, text=raw_text)
                translated = translation_provider.translate([seg], target_lang)
                if translated and len(translated) > 0:
                    overlay["replacement_text"] = translated[0].translated_text or raw_text
                else:
                    overlay["replacement_text"] = raw_text
            elif raw_text:
                overlay["replacement_text"] = raw_text

    def _enrich_image_overlays(self, image_overlays: List[Tuple[TrackActionPlan, Dict[str, Any]]]) -> None:
        for _, overlay in image_overlays:
            asset_path = overlay.get("image_asset_path")
            if not asset_path or not os.path.exists(asset_path):
                raise FileNotFoundError(f"Image asset for replace_image overlay not found at: {asset_path}")
