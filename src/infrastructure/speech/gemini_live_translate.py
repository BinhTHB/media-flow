"""Experimental Gemini 3.5 Live Translate audio-to-audio adapter.

This is intentionally separate from the normal ASR -> text translation -> TTS
pipeline. It streams 16 kHz mono PCM to Gemini Live Translation and writes the
translated 24 kHz mono PCM response as WAV, plus input/output transcripts.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path


@dataclass
class LiveTranslateResult:
    audio_path: str
    input_transcript: str
    output_transcript: str
    audio_bytes: int


class GeminiLiveTranslateAdapter:
    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gemini-3.5-live-translate-preview",
        response_timeout_seconds: float = 45.0,
    ) -> None:
        raw_key = api_key or os.getenv("GEMINI_API_KEY")
        # Match the existing Gemini REST client: project configs may contain a
        # comma-separated pool, while one Live WebSocket session uses one key.
        self.api_key = raw_key.split(",", 1)[0].strip() if raw_key else None
        self.model = model
        self.response_timeout_seconds = max(5.0, float(response_timeout_seconds))

    def translate_audio(
        self,
        input_path: str,
        output_wav_path: str,
        target_language: str = "vi",
    ) -> LiveTranslateResult:
        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY is required for Gemini Live Translate.")
        source = Path(input_path).resolve()
        if not source.is_file():
            raise FileNotFoundError(f"Input media not found: {source}")
        output = Path(output_wav_path).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory(prefix="mediaflow_gemini_live_translate_") as temp_dir:
            pcm_path = Path(temp_dir) / "input_16k_mono_s16le.pcm"
            self._prepare_pcm(source, pcm_path)
            result = asyncio.run(
                self._translate_pcm(
                    pcm_path.read_bytes(),
                    target_language=target_language,
                )
            )

        audio_bytes, input_text, output_text = result
        if len(audio_bytes) < 2:
            raise RuntimeError("Gemini Live Translate returned no translated audio.")
        self._write_wav(output, audio_bytes, sample_rate=24000)
        return LiveTranslateResult(
            audio_path=str(output),
            input_transcript=input_text.strip(),
            output_transcript=output_text.strip(),
            audio_bytes=len(audio_bytes),
        )

    async def _translate_pcm(self, pcm: bytes, target_language: str) -> tuple[bytes, str, str]:
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise RuntimeError(
                "google-genai is required for Gemini Live Translate. Install google-genai>=2.22."
            ) from exc

        client = genai.Client(api_key=self.api_key)
        config = types.LiveConnectConfig(
            response_modalities=["AUDIO"],
            input_audio_transcription=types.AudioTranscriptionConfig(),
            output_audio_transcription=types.AudioTranscriptionConfig(),
            translation_config=types.TranslationConfig(
                target_language_code=target_language,
                echo_target_language=True,
            ),
        )
        audio_parts: list[bytes] = []
        input_parts: list[str] = []
        output_parts: list[str] = []
        completed = asyncio.Event()

        async with client.aio.live.connect(model=self.model, config=config) as session:
            async def receive_loop() -> None:
                async for response in session.receive():
                    content = response.server_content
                    if content is None:
                        continue
                    if content.input_transcription and content.input_transcription.text:
                        input_parts.append(content.input_transcription.text)
                    if content.output_transcription and content.output_transcription.text:
                        output_parts.append(content.output_transcription.text)
                    if content.model_turn:
                        for part in content.model_turn.parts or []:
                            if part.inline_data and part.inline_data.data:
                                audio_parts.append(part.inline_data.data)
                    if content.turn_complete or content.waiting_for_input:
                        completed.set()
                        return

            receiver = asyncio.create_task(receive_loop())
            # 100 ms at 16 kHz, mono, 16-bit PCM = 3200 bytes.
            chunk_bytes = 3200
            for start in range(0, len(pcm), chunk_bytes):
                chunk = pcm[start : start + chunk_bytes]
                if chunk:
                    await session.send_realtime_input(
                        audio=types.Blob(data=chunk, mime_type="audio/pcm;rate=16000")
                    )
                    # Live Translation is a continuous real-time interpreter. Pacing
                    # 100 ms chunks close to wall clock avoids overwhelming VAD/language
                    # detection with an entire clip delivered in a burst.
                    await asyncio.sleep(0.10)
            await session.send_realtime_input(audio_stream_end=True)

            try:
                await asyncio.wait_for(completed.wait(), timeout=self.response_timeout_seconds)
            except asyncio.TimeoutError:
                # Some preview sessions don't emit waiting_for_input promptly. Keep any
                # translated audio already received, then cancel the receiver cleanly.
                pass
            finally:
                if not receiver.done():
                    receiver.cancel()
                    try:
                        await receiver
                    except asyncio.CancelledError:
                        pass

        return b"".join(audio_parts), " ".join(input_parts), " ".join(output_parts)

    @staticmethod
    def _prepare_pcm(source: Path, output: Path) -> None:
        cmd = [
            "ffmpeg", "-y", "-v", "error", "-i", str(source),
            "-vn", "-ac", "1", "-ar", "16000", "-f", "s16le", "-acodec", "pcm_s16le",
            str(output),
        ]
        subprocess.run(cmd, capture_output=True, text=True, check=True)

    @staticmethod
    def _write_wav(path: Path, pcm: bytes, sample_rate: int) -> None:
        with wave.open(str(path), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(sample_rate)
            handle.writeframes(pcm)
