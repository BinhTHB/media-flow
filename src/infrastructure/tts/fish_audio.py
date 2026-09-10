"""Fish Audio S2.1 Pro Free TTS adapter.

Uses the public REST endpoint directly, avoiding an SDK dependency. If ``voice``
is in the form ``fish:<reference_id>``, that reference is used for voice cloning;
otherwise ``FISH_REFERENCE_ID`` can provide a default clone/reference voice.
"""

from __future__ import annotations

import json
import os
import re
import threading
import urllib.error
import urllib.request
from pathlib import Path
from typing import List, Optional, Sequence

from src.core.contracts import TTSPort


def _split_values(raw: str | None) -> List[str]:
    if not raw:
        return []
    return [item.strip() for item in re.split(r"[,;\n]+", raw) if item.strip()]


def configured_fish_keys() -> List[str]:
    values: List[str] = []
    values.extend(_split_values(os.getenv("FISH_API_KEYS")))
    for env_name in ("FISH_API_KEY", "FISH_AUDIO_API_KEY"):
        value = os.getenv(env_name, "").strip()
        if value:
            values.append(value)
    return list(dict.fromkeys(values))


class FishAudioTTSAdapter(TTSPort):
    endpoint = "https://api.fish.audio/v1/tts"

    def __init__(
        self,
        api_keys: Optional[Sequence[str]] = None,
        *,
        model: str = "s2.1-pro-free",
        reference_id: Optional[str] = None,
        timeout_seconds: float = 120.0,
    ) -> None:
        self.api_keys = [key.strip() for key in (api_keys or configured_fish_keys()) if key and key.strip()]
        self.model = model
        self.reference_id = reference_id or os.getenv("FISH_REFERENCE_ID", "").strip() or None
        self.timeout_seconds = timeout_seconds
        self._cursor = 0
        self._lock = threading.Lock()

    @property
    def available(self) -> bool:
        return bool(self.api_keys)

    def synthesize(
        self,
        text: str,
        output_path: str,
        voice: str = "vi-VN-HoaiMyNeural",
    ) -> str:
        if not text.strip():
            raise ValueError("TTS text cannot be empty.")
        if not self.api_keys:
            raise RuntimeError("Fish Audio TTS requires FISH_API_KEY(S).")

        reference_id = self.reference_id
        if voice and voice.startswith("fish:"):
            candidate = voice.split(":", 1)[1].strip()
            if candidate:
                reference_id = candidate

        payload = {"text": text, "format": "mp3"}
        if reference_id:
            payload["reference_id"] = reference_id

        with self._lock:
            start_cursor = self._cursor % len(self.api_keys)
        errors: List[str] = []
        for offset in range(len(self.api_keys)):
            key_index = (start_cursor + offset) % len(self.api_keys)
            key = self.api_keys[key_index]
            try:
                audio_bytes = self._post(payload, key)
                if len(audio_bytes) < 256:
                    raise RuntimeError("Fish Audio returned an unexpectedly small audio body")
                destination = Path(output_path).resolve()
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(audio_bytes)
                if not destination.is_file() or destination.stat().st_size == 0:
                    raise RuntimeError("Fish Audio output file is empty")
                with self._lock:
                    self._cursor = (key_index + 1) % len(self.api_keys)
                return str(destination)
            except Exception as exc:
                errors.append(f"key#{key_index + 1}: {str(exc)[:240]}")

        raise RuntimeError("Fish Audio TTS exhausted configured key pool. " + " | ".join(errors[-4:]))

    def _post(self, payload: dict, api_key: str) -> bytes:
        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            method="POST",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "Accept": "audio/mpeg,application/octet-stream",
                "model": self.model,
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {exc.code}: {detail[:300]}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"network error: {exc.reason}") from exc
