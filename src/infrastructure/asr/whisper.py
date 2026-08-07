"""
Faster-Whisper ASR Infrastructure Adapter

Implements ASRPort using faster_whisper.
Converts transcription segments into core SpeechSegment objects.
"""

from __future__ import annotations

import logging
from typing import List

from src.core.contracts import ASRPort
from src.core.models.speech_segment import SpeechSegment

logger = logging.getLogger(__name__)


class WhisperASRAdapter(ASRPort):
    def __init__(self, model_size: str = "base", device: str = "cuda"):
        from src.infrastructure.providers.speech.faster_whisper_speech_provider import FasterWhisperSpeechProvider

        self.provider = FasterWhisperSpeechProvider(model_size=model_size, device=device)

    def transcribe(self, audio_path: str, language: str = "auto") -> List[SpeechSegment]:
        raw_segments = self.provider.transcribe(audio_path, language=language)
        results: List[SpeechSegment] = []

        for idx, seg in enumerate(raw_segments):
            results.append(
                SpeechSegment(
                    segment_id=getattr(seg, "id", f"speech_{idx}"),
                    start_time=float(seg.start),
                    end_time=float(seg.end),
                    source_text=str(seg.text).strip(),
                    source_language=language,
                )
            )

        return results
