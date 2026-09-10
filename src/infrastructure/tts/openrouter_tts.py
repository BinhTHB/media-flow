"""OpenRouter TTS adapter using the shared multi-account OpenRouter key pool.

Primary model: Fish Audio S2.1 Pro Free. The adapter speaks through OpenRouter's
OpenAI-compatible /api/v1/audio/speech endpoint, so no Fish-specific API key is
required. Edge TTS remains the outer fallback at the composition root.
"""

from __future__ import annotations

import json
import os
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional, Sequence

from src.core.contracts import TTSPort
from src.infrastructure.translation.openrouter_translation import configured_openrouter_keys


class OpenRouterTTSAdapter(TTSPort):
    endpoint = "https://openrouter.ai/api/v1/audio/speech"

    def __init__(
        self,
        api_keys: Optional[Sequence[str]] = None,
        model: str | None = None,
        timeout_seconds: float = 120.0,
    ) -> None:
        self.api_keys = [k.strip() for k in (api_keys or configured_openrouter_keys()) if k and k.strip()]
        self.model = model or os.getenv("OPENROUTER_TTS_MODEL", "fish-audio/s2.1-pro-free:free")
        self.timeout_seconds = timeout_seconds
        self._cursor = 0
        self._disabled: set[int] = set()
        self._cooldown_until: dict[int, float] = {}
        self._lock = threading.Lock()

    @property
    def available(self) -> bool:
        return bool(self.api_keys)

    def synthesize(self, text: str, output_path: str, voice: str = "vi-VN-HoaiMyNeural") -> str:
        if not text or not text.strip():
            raise ValueError("TTS text is empty.")
        if not self.api_keys:
            raise RuntimeError("OpenRouter key pool is required for OpenRouter TTS.")

        target = Path(output_path).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        response_format = "mp3" if target.suffix.lower() == ".mp3" else "wav"
        fish_voice = self._normalize_voice(voice)
        styled_text = self._apply_style(text.strip())
        payload = {
            "model": self.model,
            "input": styled_text,
            "voice": fish_voice,
            "response_format": response_format,
        }

        with self._lock:
            start = self._cursor % len(self.api_keys)

        errors: list[str] = []
        for offset in range(len(self.api_keys)):
            idx = (start + offset) % len(self.api_keys)
            with self._lock:
                if idx in self._disabled:
                    continue
                cooldown = self._cooldown_until.get(idx, 0.0)
            if cooldown > time.monotonic():
                continue

            req = urllib.request.Request(
                self.endpoint,
                data=json.dumps(payload).encode("utf-8"),
                method="POST",
                headers={
                    "Authorization": f"Bearer {self.api_keys[idx]}",
                    "Content-Type": "application/json",
                    "Accept": "audio/mpeg, audio/wav, application/octet-stream",
                    "HTTP-Referer": os.getenv("OPENROUTER_HTTP_REFERER", ""),
                    "X-Title": os.getenv("OPENROUTER_APP_TITLE", "MediaFlow"),
                },
            )
            try:
                with urllib.request.urlopen(req, timeout=self.timeout_seconds) as response:
                    audio = response.read()
                if len(audio) < 128:
                    raise RuntimeError("OpenRouter TTS returned an empty/invalid audio body.")
                target.write_bytes(audio)
                with self._lock:
                    self._cursor = (idx + 1) % len(self.api_keys)
                return str(target)
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")
                errors.append(f"HTTP {exc.code}: {detail[:220]}")
                if exc.code in {401, 403}:
                    with self._lock:
                        self._disabled.add(idx)
                elif exc.code in {402, 429}:
                    with self._lock:
                        self._cooldown_until[idx] = time.monotonic() + 900.0
            except urllib.error.URLError as exc:
                errors.append(f"network error: {exc.reason}")
            except Exception as exc:
                errors.append(str(exc)[:220])

        raise RuntimeError("OpenRouter Fish TTS exhausted configured key pool. " + " | ".join(errors[-4:]))

    @staticmethod
    def _apply_style(text: str) -> str:
        mode = os.getenv("OPENROUTER_FISH_STYLE_MODE", "auto").strip().lower()
        if mode in {"", "off", "none", "false", "0"} or text.lstrip().startswith("["):
            return text
        if mode not in {"auto", "natural"}:
            return f"[{mode}] {text}"

        lowered = text.lower().strip()
        if lowered.startswith(("xin lỗi", "tôi xin lỗi", "thành thật xin lỗi")):
            tag = "sincere apologetic conversational tone"
        elif lowered.startswith(("cảm ơn", "cám ơn")):
            tag = "warm friendly conversational tone"
        elif lowered.startswith(("tạm biệt", "chào nhé", "bye")):
            tag = "warm casual farewell"
        elif text.rstrip().endswith("?"):
            tag = "natural curious conversational tone"
        elif text.rstrip().endswith("!"):
            tag = "lively natural conversational tone"
        else:
            tag = "natural conversational tone"
        return f"[{tag}] {text}"

    @staticmethod
    def _normalize_voice(voice: str) -> str:
        value = (voice or "").strip()
        if value.startswith("fish:") and value[5:].strip():
            return value[5:].strip()

        female_default = os.getenv(
            "OPENROUTER_FISH_FEMALE_VOICE",
            "c5660c5f4f7b4b74be5f248a75d7fee6",
        )
        male_default = os.getenv(
            "OPENROUTER_FISH_MALE_VOICE",
            "e692694d683743509dcbe81fab88b147",
        )
        if value == "vi-VN-NamMinhNeural":
            return male_default
        if value.startswith("vi-") or value.endswith("Neural") or not value:
            return female_default
        return value
