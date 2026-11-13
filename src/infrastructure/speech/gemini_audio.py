"""Gemini Audio ASR infrastructure adapter.

Implements ASRPort with the existing Gemini REST runtime. Audio is compressed to
mono MP3 and long inputs are chunked so the adapter remains lightweight locally.
"""

from __future__ import annotations

import base64
import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, List

from src.core.contracts import ASRPort
from src.core.models.speech_segment import SpeechSegment
from src.infrastructure.gemini.client import GeminiClient
from src.infrastructure.gemini.contracts import GeminiRequest
from src.infrastructure.gemini.request_scheduler import RequestScheduler
from src.infrastructure.gemini.structured_output import parse_json_recovering

logger = logging.getLogger(__name__)


class GeminiAudioASRAdapter(ASRPort):
    """Remote ASR using Gemini multimodal audio understanding.

    The adapter intentionally owns only provider/media mechanics. It returns
    canonical SpeechSegment values and does not translate or synthesize audio.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gemini-3.5-flash-lite",
        chunk_seconds: int = 60,
    ) -> None:
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model = model
        self.chunk_seconds = max(60, int(chunk_seconds))

    def transcribe(self, audio_path: str, language: str = "auto") -> List[SpeechSegment]:
        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY is required for Gemini Audio ASR.")

        source = Path(audio_path).resolve()
        if not source.is_file():
            raise FileNotFoundError(f"Audio file not found: {source}")

        with tempfile.TemporaryDirectory(prefix="mediaflow_gemini_asr_") as temp_dir:
            temp_root = Path(temp_dir)
            prepared = temp_root / "audio.mp3"
            self._prepare_audio(source, prepared)
            duration = self._probe_duration(prepared)

            if duration <= self.chunk_seconds:
                return self._transcribe_chunk(prepared, language=language, offset=0.0, index_base=0)

            chunks = self._split_audio(prepared, temp_root)
            segments: List[SpeechSegment] = []
            next_index = 0
            for chunk_index, chunk in enumerate(chunks):
                offset = float(chunk_index * self.chunk_seconds)
                try:
                    chunk_segments = self._transcribe_chunk(
                        chunk,
                        language=language,
                        offset=offset,
                        index_base=next_index,
                    )
                except Exception as exc:
                    logger.warning(
                        "Gemini ASR chunk %d failed; keeping the other chunks: %s",
                        chunk_index,
                        exc,
                    )
                    continue
                segments.extend(chunk_segments)
                next_index += len(chunk_segments)

            return sorted(segments, key=lambda item: (item.start_time, item.end_time))

    @staticmethod
    def _prepare_audio(source: Path, output: Path) -> None:
        cmd = [
            "ffmpeg", "-y", "-v", "error", "-i", str(source),
            "-vn", "-ac", "1", "-ar", "16000", "-b:a", "48k", str(output),
        ]
        subprocess.run(cmd, capture_output=True, text=True, check=True)

    def _split_audio(self, source: Path, temp_root: Path) -> List[Path]:
        pattern = temp_root / "chunk_%04d.mp3"
        cmd = [
            "ffmpeg", "-y", "-v", "error", "-i", str(source),
            "-f", "segment", "-segment_time", str(self.chunk_seconds),
            "-reset_timestamps", "1", "-c", "copy", str(pattern),
        ]
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        chunks = sorted(temp_root.glob("chunk_*.mp3"))
        if not chunks:
            raise RuntimeError("Gemini ASR audio chunking produced no chunks.")
        return chunks

    @staticmethod
    def _probe_duration(path: Path) -> float:
        cmd = [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", str(path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return max(0.0, float(result.stdout.strip() or 0.0))

    def _transcribe_chunk(
        self,
        path: Path,
        *,
        language: str,
        offset: float,
        index_base: int,
    ) -> List[SpeechSegment]:
        audio_b64 = base64.b64encode(path.read_bytes()).decode("ascii")
        language_rule = (
            "Detect the spoken language automatically."
            if language == "auto"
            else f"The expected spoken language is {language}."
        )
        prompt = (
            "Transcribe this audio into speech segments with accurate timestamps.\n"
            f"{language_rule}\n"
            "Rules:\n"
            "1. Return spoken words only; do not translate.\n"
            "2. Split at natural phrase/sentence boundaries.\n"
            "3. Timestamps are seconds relative to the supplied audio chunk.\n"
            "4. Keep start/end monotonic and non-overlapping where practical.\n"
            "5. If speech is absent, return an empty segments array.\n"
            "6. Return JSON only in this exact shape:\n"
            '{"detected_language":"xx","segments":[{"start":0.0,"end":1.2,"text":"...","speaker_id":null}]}'
        )
        client = GeminiClient(
            api_key=self.api_key,
            model=self.model,
            timeout=45.0,
            scheduler=RequestScheduler(max_retries=2, backoff_max_seconds=8.0),
        )
        response = client.generate_content(
            GeminiRequest(
                contents=[{
                    "parts": [
                        {"text": prompt},
                        {"inline_data": {"mime_type": "audio/mpeg", "data": audio_b64}},
                    ]
                }],
                model=self.model,
                generation_config={"temperature": 0.0},
                request_id=f"audio_asr_{index_base}",
                source_kind="audio_asr",
                api_surface="generate_content",
            ),
            use_cache=False,
        )
        payload = parse_json_recovering(response.text)
        if isinstance(payload, list):
            raw_segments: Any = payload
            detected_language = language
        elif isinstance(payload, dict):
            raw_segments = payload.get("segments", [])
            detected_language = str(payload.get("detected_language") or language or "auto")
        else:
            raise RuntimeError("Gemini ASR returned an unsupported JSON shape.")

        if not isinstance(raw_segments, list):
            raise RuntimeError("Gemini ASR response field 'segments' is not a list.")

        result: List[SpeechSegment] = []
        last_start = 0.0
        for item in raw_segments:
            if not isinstance(item, dict):
                continue
            text = str(item.get("text") or item.get("source_text") or "").strip()
            if not text:
                continue
            try:
                start = max(0.0, float(item.get("start", item.get("start_time", last_start))))
                end = float(item.get("end", item.get("end_time", start + 0.1)))
            except (TypeError, ValueError):
                continue
            end = max(start + 0.05, end)
            absolute_start = offset + start
            absolute_end = offset + end
            seg_index = index_base + len(result)
            result.append(
                SpeechSegment.normalized_timing(
                    segment_id=f"S{seg_index:05d}",
                    start_time=absolute_start,
                    end_time=absolute_end,
                    source_text=text,
                    source_language=detected_language,
                    speaker_id=item.get("speaker_id"),
                    minimum_duration=0.05,
                )
            )
            last_start = start

        return result
