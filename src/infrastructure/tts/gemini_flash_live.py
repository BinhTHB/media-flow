"""Gemini 3.1 Flash Live native-audio adapter used as an experimental TTS engine.

Unlike Gemini Live Translate, this adapter receives already translated Vietnamese
text and asks the general Live model to speak it verbatim. This avoids language
recognition/translation errors while using the higher-availability Live quota.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import tempfile
import wave
from pathlib import Path

from src.core.contracts import TTSPort


class GeminiFlashLiveTTSAdapter(TTSPort):
    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gemini-3.1-flash-live-preview",
        voice_name: str | None = None,
        timeout_seconds: float = 30.0,
    ) -> None:
        raw_key = api_key or os.getenv("GEMINI_API_KEY")
        self.api_key = raw_key.split(",", 1)[0].strip() if raw_key else None
        self.model = model
        self.voice_name = voice_name or os.getenv("GEMINI_LIVE_TTS_VOICE", "").strip() or None
        self.timeout_seconds = max(5.0, float(timeout_seconds))
        self.last_output_transcript = ""
        self.last_audio_bytes = 0

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def synthesize(self, text: str, output_path: str, voice: str = "") -> str:
        text = (text or "").strip()
        if not text:
            raise ValueError("TTS text must not be empty.")
        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY is required for Gemini Flash Live TTS.")

        requested_voice = self._voice_from_request(voice) or self.voice_name
        pcm, transcript = asyncio.run(self._synthesize_live(text, requested_voice))
        if len(pcm) < 2:
            raise RuntimeError("Gemini Flash Live returned no audio.")
        self.last_output_transcript = transcript.strip()
        self.last_audio_bytes = len(pcm)

        target = Path(output_path).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.suffix.lower() == ".wav":
            self._write_wav(target, pcm, 24000)
            return str(target)

        with tempfile.TemporaryDirectory(prefix="mediaflow_gemini_live_tts_") as temp_dir:
            wav_path = Path(temp_dir) / "speech.wav"
            self._write_wav(wav_path, pcm, 24000)
            cmd = [
                "ffmpeg", "-y", "-v", "error", "-i", str(wav_path),
                "-ar", "48000", "-ac", "2", str(target),
            ]
            subprocess.run(cmd, capture_output=True, text=True, check=True)
        if not target.is_file() or target.stat().st_size == 0:
            raise RuntimeError("Gemini Flash Live did not produce a non-empty audio file.")
        return str(target)

    async def _synthesize_live(self, text: str, voice_name: str | None) -> tuple[bytes, str]:
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise RuntimeError("google-genai>=2.22 is required for Gemini Flash Live TTS.") from exc

        client = genai.Client(api_key=self.api_key)
        speech_config = types.SpeechConfig(language_code="vi-VN")
        if voice_name:
            speech_config.voice_config = types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice_name)
            )

        config = types.LiveConnectConfig(
            response_modalities=["AUDIO"],
            speech_config=speech_config,
            output_audio_transcription=types.AudioTranscriptionConfig(),
            thinking_config=types.ThinkingConfig(thinking_level="minimal"),
            system_instruction=(
                "Bạn là bộ máy lồng tiếng tiếng Việt. Khi nhận văn bản, chỉ phát âm nguyên văn nội dung "
                "sau nhãn TEXT. Không giải thích, không chào hỏi, không thêm hoặc bỏ ý. Đọc tự nhiên như hội thoại "
                "của người thật, giữ đúng dấu câu và sắc thái, không đọc từ TEXT."
            ),
        )
        audio_parts: list[bytes] = []
        transcript_parts: list[str] = []
        completed = asyncio.Event()

        async with client.aio.live.connect(model=self.model, config=config) as session:
            async def receive_loop() -> None:
                async for response in session.receive():
                    content = response.server_content
                    if content is None:
                        continue
                    if content.output_transcription and content.output_transcription.text:
                        transcript_parts.append(content.output_transcription.text)
                    if content.model_turn:
                        for part in content.model_turn.parts or []:
                            if part.inline_data and part.inline_data.data:
                                audio_parts.append(part.inline_data.data)
                    if content.turn_complete or content.waiting_for_input:
                        completed.set()
                        return

            receiver = asyncio.create_task(receive_loop())
            await session.send_realtime_input(text=f"TEXT: {text}")
            try:
                await asyncio.wait_for(completed.wait(), timeout=self.timeout_seconds)
            except asyncio.TimeoutError:
                pass
            finally:
                if not receiver.done():
                    receiver.cancel()
                    try:
                        await receiver
                    except asyncio.CancelledError:
                        pass

        return b"".join(audio_parts), " ".join(transcript_parts)

    @staticmethod
    def _voice_from_request(voice: str) -> str | None:
        value = (voice or "").strip()
        if value.startswith("gemini-live:") and value[12:].strip():
            return value[12:].strip()
        return None

    @staticmethod
    def _write_wav(path: Path, pcm: bytes, sample_rate: int) -> None:
        with wave.open(str(path), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(sample_rate)
            handle.writeframes(pcm)
