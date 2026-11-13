"""Microsoft Edge online TTS adapter.

Uses the lightweight edge-tts client: no local neural model and no API key.
"""

from __future__ import annotations

import asyncio
import subprocess
import tempfile
import threading
import time
from pathlib import Path

from src.core.contracts import TTSPort

_EDGE_LOCK = threading.Lock()


class EdgeTTSAdapter(TTSPort):
    def __init__(self, rate: str = "+0%", volume: str = "+0%", pitch: str = "+0Hz") -> None:
        self.rate = rate
        self.volume = volume
        self.pitch = pitch

    def synthesize(self, text: str, output_path: str, voice: str = "vi-VN-HoaiMyNeural") -> str:
        text = (text or "").strip()
        if not text:
            raise ValueError("TTS text must not be empty.")

        target = Path(output_path).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)

        if target.suffix.lower() == ".mp3":
            self._run_edge_tts(text, voice, target)
            return str(target)

        with tempfile.TemporaryDirectory(prefix="mediaflow_edge_tts_") as temp_dir:
            temp_mp3 = Path(temp_dir) / "speech.mp3"
            self._run_edge_tts(text, voice, temp_mp3)
            cmd = [
                "ffmpeg", "-y", "-v", "error", "-i", str(temp_mp3),
                "-ar", "48000", "-ac", "2", str(target),
            ]
            subprocess.run(cmd, capture_output=True, text=True, check=True)
        return str(target)

    def _run_edge_tts(self, text: str, voice: str, output: Path) -> None:
        try:
            import edge_tts
        except ImportError as exc:
            raise RuntimeError(
                "edge-tts is not installed. Install it in the project venv with "
                "'.venv\\Scripts\\python.exe -m pip install edge-tts'."
            ) from exc

        async def _save() -> None:
            communicate = edge_tts.Communicate(
                text=text,
                voice=voice,
                rate=self.rate,
                volume=self.volume,
                pitch=self.pitch,
                connect_timeout=8,
                receive_timeout=15,
            )
            await communicate.save(str(output))

        last_error: Exception | None = None
        for attempt in range(3):
            if output.exists():
                output.unlink()
            try:
                with _EDGE_LOCK:
                    asyncio.run(_save())
                    # The public service drops a burst of sockets and then answers
                    # with an empty stream. One voice at a time, with a gap.
                time.sleep(0.8)
                if output.is_file() and output.stat().st_size > 0:
                    return
                last_error = RuntimeError(
                    "Edge TTS returned without producing a non-empty audio file."
                )
            except Exception as exc:
                last_error = exc
            if attempt < 2 and _edge_retryable(last_error):
                time.sleep(2.0 * (attempt + 1))
                continue
            break
        if last_error is None:
            raise RuntimeError("Edge TTS returned without producing a non-empty audio file.")
        raise last_error


def _edge_retryable(exc: Exception | None) -> bool:
    """The Edge service sometimes accepts the socket and then sends no audio."""
    if exc is None:
        return False
    if isinstance(exc, TimeoutError):
        return True
    message = str(exc)
    return (
        "No audio was received" in message
        or "non-empty audio file" in message
        or "timed out" in message.lower()
    )
