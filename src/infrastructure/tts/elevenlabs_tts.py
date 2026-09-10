"""ElevenLabs REST TTS adapter for optional free-tier development use.

Uses the synchronous Text-to-Speech REST endpoint directly through urllib, so
MediaFlow does not require the ElevenLabs SDK. The default model is `eleven_v3`
with `language_code=vi`, selected after short-segment Vietnamese testing. Native
Voice Library voices may require a paid ElevenLabs tier for API access; free-tier
users can still use premade voices available to their account.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from src.core.contracts import TTSPort
from src.infrastructure.elevenlabs_dubbing import configured_elevenlabs_keys


# Sarah is a premade/free-API voice. With Eleven v3 + language_code=vi it was
# the most stable of the account's premade voices on MediaFlow's short-segment
# Vietnamese benchmark. A native Vietnamese professional/library voice is still
# preferable when the account tier permits API access.
DEFAULT_ELEVENLABS_VOICE_ID = "EXAVITQu4vr4xnSDxMaL"


def _env_float(name: str, default: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    raw = os.getenv(name, "").strip()
    try:
        value = float(raw) if raw else float(default)
    except ValueError:
        value = float(default)
    return max(minimum, min(maximum, value))


class ElevenLabsTTSAdapter(TTSPort):
    endpoint_base = "https://api.elevenlabs.io/v1/text-to-speech"

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        timeout_seconds: float = 120.0,
    ) -> None:
        self.api_keys = [api_key.strip()] if api_key and api_key.strip() else configured_elevenlabs_keys()
        self.api_key = self.api_keys[0] if self.api_keys else ""
        self.model = (model or os.getenv("ELEVENLABS_TTS_MODEL") or "eleven_v3").strip()
        self.timeout_seconds = max(5.0, float(timeout_seconds))

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def synthesize(self, text: str, output_path: str, voice: str = "") -> str:
        text = (text or "").strip()
        if not text:
            raise ValueError("TTS text must not be empty.")
        if not self.api_key:
            raise RuntimeError("ELEVENLABS_API_KEY is required for ElevenLabs TTS.")

        voice_id = self._normalize_voice(voice)
        target = Path(output_path).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        # The pipeline normally requests MP3 and later normalizes with FFmpeg.
        output_format = os.getenv("ELEVENLABS_OUTPUT_FORMAT", "mp3_44100_128").strip() or "mp3_44100_128"
        endpoint = (
            f"{self.endpoint_base}/{urllib.parse.quote(voice_id, safe='')}"
            f"?output_format={urllib.parse.quote(output_format, safe='')}"
        )

        payload = {
            "text": text,
            "model_id": self.model,
            "voice_settings": {
                # Lower stability generally allows more emotional variation while
                # keeping similarity high enough to avoid excessive voice drift.
                "stability": _env_float("ELEVENLABS_STABILITY", 0.40),
                "similarity_boost": _env_float("ELEVENLABS_SIMILARITY_BOOST", 0.78),
                "style": _env_float("ELEVENLABS_STYLE", 0.15),
                "use_speaker_boost": os.getenv("ELEVENLABS_SPEAKER_BOOST", "true").strip().lower()
                not in {"0", "false", "off", "no"},
                "speed": _env_float("ELEVENLABS_SPEED", 1.0, 0.7, 1.2),
            },
        }
        # multilingual_v2 auto-detects language and does not accept language_code;
        # v3/Flash can be forced to Vietnamese, which is important for very short
        # dubbing segments that would otherwise be ambiguous.
        language_code = os.getenv("ELEVENLABS_LANGUAGE_CODE", "vi").strip()
        if language_code and self.model != "eleven_multilingual_v2":
            payload["language_code"] = language_code
        audio = b""
        errors: list[str] = []
        for index, key in enumerate(self.api_keys):
            request = urllib.request.Request(
                endpoint,
                data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                method="POST",
                headers={
                    "xi-api-key": key,
                    "Content-Type": "application/json",
                    "Accept": "audio/mpeg, application/octet-stream",
                    "User-Agent": "MediaFlow/1.0",
                },
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                    audio = response.read()
                self.api_key = key
                break
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")
                if exc.code in {401, 402, 403, 429}:
                    errors.append(f"key#{index + 1}: HTTP {exc.code}: {detail[:180]}")
                    continue
                raise RuntimeError(f"ElevenLabs TTS HTTP {exc.code}: {detail[:400]}") from exc
            except urllib.error.URLError as exc:
                raise RuntimeError(f"ElevenLabs TTS network error: {exc.reason}") from exc
        if not audio:
            raise RuntimeError("ElevenLabs TTS key pool exhausted. " + " | ".join(errors))

        if len(audio) < 128:
            raise RuntimeError("ElevenLabs TTS returned an empty/invalid audio body.")
        target.write_bytes(audio)
        return str(target)

    @staticmethod
    def _normalize_voice(voice: str) -> str:
        value = (voice or "").strip()
        if value.startswith("elevenlabs:"):
            selected = value.split(":", 1)[1].strip()
            if selected and selected.lower() not in {"env", "default"}:
                return selected
        configured = os.getenv("ELEVENLABS_VOICE_ID", "").strip()
        return configured or DEFAULT_ELEVENLABS_VOICE_ID
