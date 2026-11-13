"""Groq Whisper ASR infrastructure adapter.

Uses Groq's OpenAI-compatible transcription endpoint through the Python standard
library, so no Groq SDK dependency is required.
"""

from __future__ import annotations

import json
import mimetypes
import os
import uuid
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, List

from src.core.contracts import ASRPort
from src.core.models.speech_segment import SpeechSegment


class GroqWhisperASRAdapter(ASRPort):
    endpoint = "https://api.groq.com/openai/v1/audio/transcriptions"

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "whisper-large-v3",
        timeout_seconds: float = 120.0,
        max_upload_bytes: int = 25 * 1024 * 1024,
    ) -> None:
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.max_upload_bytes = max_upload_bytes

    def transcribe(self, audio_path: str, language: str = "auto") -> List[SpeechSegment]:
        if not self.api_key:
            raise RuntimeError("GROQ_API_KEY is required for Groq Whisper ASR.")
        path = Path(audio_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Audio file not found: {path}")
        if path.stat().st_size > self.max_upload_bytes:
            raise RuntimeError(
                f"Groq ASR upload is {path.stat().st_size} bytes, above configured "
                f"limit {self.max_upload_bytes}; use Gemini ASR or chunk the input."
            )

        fields = {
            "model": self.model,
            "response_format": "verbose_json",
            "temperature": "0",
        }
        if language and language != "auto":
            fields["language"] = language

        body, content_type = self._multipart_body(fields, path)
        request = urllib.request.Request(
            self.endpoint,
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": content_type,
                "Accept": "application/json",
                "User-Agent": "MediaFlow/1.0 (+https://github.com/)",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Groq ASR HTTP {exc.code}: {detail[:500]}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Groq ASR network error: {exc}") from exc

        raw_segments: Any = payload.get("segments", []) if isinstance(payload, dict) else []
        detected_language = str(payload.get("language") or language or "auto") if isinstance(payload, dict) else language
        result: List[SpeechSegment] = []
        if isinstance(raw_segments, list):
            for item in raw_segments:
                if not isinstance(item, dict):
                    continue
                text = str(item.get("text") or "").strip()
                if not text:
                    continue
                try:
                    start = max(0.0, float(item.get("start", 0.0)))
                    end = max(start + 0.05, float(item.get("end", start + 0.1)))
                except (TypeError, ValueError):
                    continue
                logprob = item.get("avg_logprob")
                result.append(
                    SpeechSegment.normalized_timing(
                        segment_id=f"S{len(result):05d}",
                        start_time=start,
                        end_time=end,
                        source_text=text,
                        source_language=detected_language,
                        minimum_duration=0.05,
                        asr_logprob=float(logprob) if logprob is not None else None,
                    )
                )

        if not result and isinstance(payload, dict):
            text = str(payload.get("text") or "").strip()
            duration = payload.get("duration")
            if text and duration:
                result.append(
                    SpeechSegment.normalized_timing(
                        segment_id="S00000",
                        start_time=0.0,
                        end_time=float(duration),
                        source_text=text,
                        source_language=detected_language,
                    )
                )
        return result

    @staticmethod
    def _multipart_body(fields: dict[str, str], path: Path) -> tuple[bytes, str]:
        boundary = f"----MediaFlow{uuid.uuid4().hex}"
        chunks: list[bytes] = []
        for name, value in fields.items():
            chunks.extend([
                f"--{boundary}\r\n".encode(),
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode(),
                str(value).encode("utf-8"),
                b"\r\n",
            ])
        mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        chunks.extend([
            f"--{boundary}\r\n".encode(),
            f'Content-Disposition: form-data; name="file"; filename="{path.name}"\r\n'.encode(),
            f"Content-Type: {mime}\r\n\r\n".encode(),
            path.read_bytes(),
            b"\r\n",
            f"--{boundary}--\r\n".encode(),
        ])
        return b"".join(chunks), f"multipart/form-data; boundary={boundary}"
