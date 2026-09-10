"""Microsoft Edge online TTS adapter.

Uses the lightweight edge-tts client: no local neural model and no API key.
"""

from __future__ import annotations

import asyncio
import subprocess
import tempfile
from pathlib import Path

from src.core.contracts import TTSPort


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
            )
            await communicate.save(str(output))

        asyncio.run(_save())
        if not output.is_file() or output.stat().st_size == 0:
            raise RuntimeError("Edge TTS returned without producing a non-empty audio file.")
