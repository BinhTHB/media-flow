"""
Independent Audio Pipeline

Executes audio extraction, ASR transcription, shared Translation, TTS synthesis, and AudioComposer.
Independent from VisualPipeline: zero visual/YOLO/OCR imports.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Optional

from src.features.audio.composer import AudioComposer
from src.features.audio.models import AudioRequest, AudioResult, TTSClip
from src.features.audio.ports import ASRPort, TTSPort
from src.features.translation.service import TranslationService

logger = logging.getLogger(__name__)


class AudioPipeline:
    def __init__(
        self,
        asr_provider: Optional[ASRPort] = None,
        translation_service: Optional[TranslationService] = None,
        tts_provider: Optional[TTSPort] = None,
        composer: Optional[AudioComposer] = None,
    ):
        self.asr_provider = asr_provider
        self.translation_service = translation_service
        self.tts_provider = tts_provider
        self.composer = composer or AudioComposer()

    def process(self, request: AudioRequest, job_workspace_dir: str) -> AudioResult:
        start_time = time.monotonic()
        timings = {}
        warnings = []

        if not request.enabled:
            logger.info("AudioPipeline is disabled in request. Skipping execution.")
            return AudioResult(
                enabled=False,
                status="SKIPPED",
                warnings=["Audio processing disabled in request."],
            )

        ws_dir = Path(job_workspace_dir)
        audio_dir = ws_dir / "audio"
        audio_dir.mkdir(parents=True, exist_ok=True)
        extracted_audio_path = str(audio_dir / "audio.wav")

        # 1. Audio Extraction
        t0 = time.monotonic()
        input_path = request.audio_path
        if input_path.endswith((".mp4", ".mkv", ".avi", ".mov", ".webm")):
            from src.infrastructure.video.ffmpeg import extract_audio
            extract_audio(input_path, extracted_audio_path)
        else:
            extracted_audio_path = input_path
        timings["audio_extraction"] = round(time.monotonic() - t0, 3)

        # 2. ASR Transcription
        if self.asr_provider is None:
            raise RuntimeError("ASR provider is not configured for enabled AudioPipeline.")

        t0 = time.monotonic()
        speech_segments = self.asr_provider.transcribe(
            audio_path=extracted_audio_path, language=request.source_language
        )
        timings["asr"] = round(time.monotonic() - t0, 3)

        # 3. Shared Translation
        if self.translation_service is not None and speech_segments:
            t0 = time.monotonic()
            raw_texts = [s.source_text for s in speech_segments]
            translated_texts = self.translation_service.translate_texts(
                raw_texts, target_language=request.target_language
            )
            for seg, trans in zip(speech_segments, translated_texts):
                seg.translated_text = trans
            timings["translation"] = round(time.monotonic() - t0, 3)

        # 4. Dubbing / TTS
        dubbed_audio_path = None
        tts_clips = []
        status = "SUCCESS"

        if request.dubbing_enabled:
            if self.tts_provider is None:
                warnings.append("Dubbing requested but TTS provider is not available. Reporting PARTIAL status.")
                status = "PARTIAL"
            else:
                t0 = time.monotonic()
                tts_dir = audio_dir / "tts"
                tts_dir.mkdir(parents=True, exist_ok=True)

                for seg in speech_segments:
                    text_to_speak = seg.translated_text or seg.source_text
                    if not text_to_speak.strip():
                        continue
                    clip_path = str(tts_dir / f"tts_{seg.segment_id}.mp3")
                    try:
                        out_clip = self.tts_provider.synthesize(
                            text=text_to_speak, output_path=clip_path, voice=request.voice
                        )
                        duration = seg.end_time - seg.start_time
                        tts_clips.append(
                            TTSClip(
                                segment_id=seg.segment_id,
                                start_time=seg.start_time,
                                end_time=seg.end_time,
                                audio_clip_path=out_clip,
                                duration=duration,
                            )
                        )
                    except Exception as exc:
                        warnings.append(f"TTS synthesis failed for segment {seg.segment_id}: {exc}")

                timings["tts"] = round(time.monotonic() - t0, 3)

                if tts_clips:
                    t0 = time.monotonic()
                    dubbed_audio_path = str(audio_dir / "dubbed_timeline.wav")
                    dubbed_audio_path = self.composer.compose_dubbed_timeline(
                        tts_clips=tts_clips,
                        total_duration=speech_segments[-1].end_time if speech_segments else 0.0,
                        output_audio_path=dubbed_audio_path,
                        original_audio_path=extracted_audio_path,
                    )
                    timings["audio_composition"] = round(time.monotonic() - t0, 3)

        timings["total"] = round(time.monotonic() - start_time, 3)

        return AudioResult(
            enabled=True,
            status=status,
            extracted_audio_path=extracted_audio_path,
            dubbed_audio_path=dubbed_audio_path,
            segments=speech_segments,
            tts_clips=tts_clips,
            warnings=warnings,
            timings=timings,
        )
