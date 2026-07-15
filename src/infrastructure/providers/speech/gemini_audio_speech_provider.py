from __future__ import annotations

import base64
import hashlib
import json
import shutil
import wave
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.application.ports.speech_provider import SpeechProvider
from src.domain.entities.segment import Segment
from src.infrastructure.gemini.contracts import (
    MODE_ROLE_ESCALATION,
    MODE_ROLE_PRIMARY,
    GeminiRequest,
)
from src.infrastructure.gemini.exceptions import GeminiError, GeminiValidationError
from src.infrastructure.gemini.runtime import GeminiRuntime
from src.infrastructure.gemini.structured_output import parse_structured
from src.shared.exceptions import ProviderError


class GeminiAudioSpeechProvider(SpeechProvider):
    """Gemini inline-audio ASR provider, intended for qualification shadow runs."""

    _PROMPT = (
        "Transcribe audible speech verbatim. Do not translate. Do not summarize. "
        "Do not correct grammar, names, pronunciation, idioms or homophones "
        "(e.g., do not correct homophones like 英国/因果). Do not use visual subtitles "
        "or OCR text. Mark unclear speech instead of guessing. Return only JSON matching "
        'this schema: {"source_language":"string","audio_duration_seconds":number,'
        '"segments":[{"start":number,"end":number,"speaker":"string",'
        '"text_verbatim":"string","speech_status":"clear|unclear|partially_audible"}]}'
    )
    _STRICT_PROMPT = _PROMPT + " Output valid JSON only. Preserve the supplied timestamps and speech."
    _STATUSES = {"clear", "unclear", "partially_audible"}

    def __init__(
        self,
        runtime: Optional[GeminiRuntime] = None,
        model: str = "gemini-3.1-flash-lite",
        escalation_model: str = "gemini-3.5-flash",
        api_key: Optional[str] = None,
        prompt_version: str = "gemini_asr_v1",
        schema_version: str = "gemini_asr_segments_v1",
        api_surface: str = "generate_content",
        full_audio_max_seconds: float = 300.0,
        chunk_fallback_enabled: bool = True,
        chunk_duration_seconds: float = 60.0,
        chunk_overlap_seconds: float = 2.0,
        escalation_enabled: bool = False,
        max_escalation_requests: int = 2,
        inline_payload_safety_limit_bytes: int = 19_000_000,
    ) -> None:
        if full_audio_max_seconds <= 0 or chunk_duration_seconds <= 0:
            raise ValueError("Audio duration limits must be positive.")
        if not 0 <= chunk_overlap_seconds < chunk_duration_seconds:
            raise ValueError("chunk_overlap_seconds must be non-negative and less than chunk_duration_seconds.")
        if max_escalation_requests < 0 or inline_payload_safety_limit_bytes <= 0:
            raise ValueError("Request limits must be non-negative and payload limit must be positive.")
        self.runtime = runtime or GeminiRuntime.create()
        self.model = model
        self.escalation_model = escalation_model
        self.api_key = api_key
        self.prompt_version = prompt_version
        self.schema_version = schema_version
        self.api_surface = api_surface
        self.full_audio_max_seconds = full_audio_max_seconds
        self.chunk_fallback_enabled = chunk_fallback_enabled
        self.chunk_duration_seconds = chunk_duration_seconds
        self.chunk_overlap_seconds = chunk_overlap_seconds
        self.escalation_enabled = escalation_enabled
        self.max_escalation_requests = max_escalation_requests
        self.inline_payload_safety_limit_bytes = inline_payload_safety_limit_bytes
        self.provenance: List[Dict[str, Any]] = []
        self.validation: List[Dict[str, Any]] = []
        self._escalation_requests = 0

    @classmethod
    def build_prompt(cls) -> str:
        return cls._PROMPT

    def transcribe(self, audio_path: str, source_language: str) -> List[Segment]:
        self.provenance, self.validation, self._escalation_requests = [], [], 0
        metadata = self._inspect_wav(audio_path)
        audio = self._read_bytes(audio_path)
        if metadata["duration"] > self.full_audio_max_seconds or self._exceeds_inline_limit(audio):
            return self._chunk_or_raise(audio_path, source_language, metadata)

        try:
            return self._request_and_validate(
                audio, source_language, metadata["duration"], 0.0, metadata["duration"]
            )
        except (GeminiError, ProviderError) as exc:
            self.validation.append({"scope": "full_audio", "error": str(exc)})
            if not self.chunk_fallback_enabled:
                raise ProviderError(f"Gemini ASR failed: {exc}") from exc
            return self._chunk_or_raise(audio_path, source_language, metadata, exc)

    def _chunk_or_raise(
        self,
        audio_path: str,
        source_language: str,
        metadata: Dict[str, Any],
        cause: Optional[Exception] = None,
    ) -> List[Segment]:
        if not self.chunk_fallback_enabled:
            if self._exceeds_inline_limit(self._read_bytes(audio_path)):
                raise ProviderError("WAV payload size exceeds inline limit")
            raise ProviderError("Gemini ASR chunk fallback is disabled") from cause
        chunk_dir = Path(audio_path).parent / "qualification" / "gemini_asr" / "tmp_chunks"
        try:
            chunks = self._write_chunks(audio_path, metadata, chunk_dir)
            segments: List[Segment] = []
            for path, start, end in chunks:
                blob = self._read_bytes(path)
                if self._exceeds_inline_limit(blob):
                    raise ProviderError("WAV payload size exceeds inline limit")
                segments.extend(
                    self._request_and_validate(blob, source_language, end - start, start, end)
                )
            return self._reconcile_overlaps(segments)
        except (GeminiError, ProviderError) as exc:
            if (
                self.escalation_enabled
                and "structured response invalid" in str(exc)
                and self._escalation_requests < self.max_escalation_requests
            ):
                self._escalation_requests += 1
                try:
                    return self._escalate_full_audio(audio_path, source_language, metadata)
                except (GeminiError, ProviderError) as escalation_exc:
                    raise ProviderError(f"Gemini ASR chunk fallback failed: {escalation_exc}") from escalation_exc
            raise ProviderError(f"Gemini ASR chunk fallback failed: {exc}") from exc
        finally:
            shutil.rmtree(chunk_dir, ignore_errors=True)

    def _request_and_validate(
        self, audio: bytes, source_language: str, duration: float, chunk_start: float, chunk_end: float
    ) -> List[Segment]:
        last_error: Optional[Exception] = None
        for attempt, prompt in enumerate((self._PROMPT, self._STRICT_PROMPT)):
            try:
                response = self._send(
                    audio, prompt, self.model, MODE_ROLE_PRIMARY, source_language, chunk_start, chunk_end
                )
                return self._parse_segments(response.text, source_language, duration, chunk_start)
            except (GeminiError, ProviderError) as exc:
                last_error = exc
                self.validation.append(
                    {"chunk_start": chunk_start, "chunk_end": chunk_end, "attempt": attempt + 1, "error": str(exc)}
                )
        raise ProviderError(f"Gemini ASR structured response invalid after retry: {last_error}")

    def _escalate_full_audio(
        self, audio_path: str, source_language: str, metadata: Dict[str, Any]
    ) -> List[Segment]:
        audio = self._read_bytes(audio_path)
        if self._exceeds_inline_limit(audio):
            raise ProviderError("WAV payload size exceeds inline limit")
        response = self._send(
            audio, self._STRICT_PROMPT, self.escalation_model, MODE_ROLE_ESCALATION, source_language,
            0.0, metadata["duration"],
        )
        return self._parse_segments(response.text, source_language, metadata["duration"], 0.0)

    def _send(
        self,
        audio: bytes,
        prompt: str,
        model: str,
        role: str,
        source_language: str,
        chunk_start: float,
        chunk_end: float,
    ):
        encoded = base64.b64encode(audio).decode("ascii")
        contents = [{
            "parts": [
                {"text": prompt},
                {"text": f"source_language={source_language}; chunk_start={chunk_start}; chunk_end={chunk_end}"},
                {"inline_data": {"mime_type": "audio/wav", "data": encoded}},
            ]
        }]
        generation_config = {"responseMimeType": "application/json", "temperature": 0.0}
        body = {"contents": contents, "generationConfig": generation_config, "systemInstruction": {"parts": [{"text": prompt}]}}
        if len(json.dumps(body, ensure_ascii=False).encode("utf-8")) > self.inline_payload_safety_limit_bytes:
            raise ProviderError("WAV payload size exceeds inline limit")
        request = GeminiRequest(
            contents=contents,
            model=model,
            generation_config=generation_config,
            system_instruction=prompt,
            prompt_version=self.prompt_version,
            schema_version=self.schema_version,
            source_kind="audio_wav",
            media_sha256=hashlib.sha256(audio).hexdigest(),
            api_surface=self.api_surface,
            role=role,
        )
        try:
            response = self.runtime.get_client(model=model, role=role, api_key=self.api_key).generate_content(request)
        except GeminiError:
            raise
        except Exception as exc:
            raise ProviderError(f"Gemini ASR request failed: {exc}") from exc
        self.provenance.append({
            "model": response.model or model, "role": role, "model_stage": response.model_stage,
            "api_surface": self.api_surface, "prompt_version": self.prompt_version,
            "schema_version": self.schema_version, "chunk_start": chunk_start, "chunk_end": chunk_end,
            "request_id": response.request_id, "cache_hit": response.cache_hit,
        })
        return response

    def _parse_segments(
        self, response_text: str, source_language: str, duration: float, offset: float
    ) -> List[Segment]:
        try:
            payload = parse_structured(response_text)
        except GeminiError as exc:
            raise GeminiValidationError(f"Gemini ASR did not return structured JSON: {exc}") from exc
        if not isinstance(payload, dict) or not isinstance(payload.get("segments"), list):
            raise GeminiValidationError("Gemini ASR response requires a segments list.")
        response_language = payload.get("source_language")
        if (
            source_language != "auto"
            and response_language != source_language
        ):
            raise GeminiValidationError("Gemini ASR response source_language does not match request.")
        result: List[Segment] = []
        seen: set[tuple[float, float, str]] = set()
        previous_end = -1.0
        for index, item in enumerate(payload["segments"]):
            if not isinstance(item, dict):
                raise GeminiValidationError("Gemini ASR segment must be an object.")
            start, end, text = item.get("start"), item.get("end"), item.get("text_verbatim")
            if (not isinstance(start, (int, float)) or not isinstance(end, (int, float))
                    or start < 0 or end <= start or end > duration + 1e-6 or start < previous_end - 1e-6):
                raise GeminiValidationError("Gemini ASR segment timestamps are invalid or not monotonic.")
            if not isinstance(text, str) or not text.strip() or item.get("speech_status") not in self._STATUSES:
                raise GeminiValidationError("Gemini ASR segment text or speech_status is invalid.")
            normalized = self._normalized(text)
            if "transcribe audible speech" in normalized or "return only json" in normalized:
                raise GeminiValidationError("Gemini ASR transcript contains prompt instructions.")
            key = (float(start), float(end), normalized)
            if key in seen:
                raise GeminiValidationError("Gemini ASR response contains duplicate segments.")
            seen.add(key)
            previous_end = float(end)
            result.append(Segment(
                id=f"speech_{int(offset * 1000)}_{index}", start=offset + float(start),
                end=offset + float(end), text=text.strip(), source="speech", confidence=0.0,
            ))
        return result

    def _write_chunks(
        self, audio_path: str, metadata: Dict[str, Any], chunk_dir: Path
    ) -> List[Tuple[str, float, float]]:
        chunk_dir.mkdir(parents=True, exist_ok=True)
        frame_rate, total_frames = metadata["frame_rate"], metadata["frame_count"]
        chunk_frames = max(1, round(self.chunk_duration_seconds * frame_rate))
        step_frames = max(1, chunk_frames - round(self.chunk_overlap_seconds * frame_rate))
        chunks: List[Tuple[str, float, float]] = []
        try:
            with wave.open(audio_path, "rb") as source:
                for index, start_frame in enumerate(range(0, total_frames, step_frames)):
                    end_frame = min(total_frames, start_frame + chunk_frames)
                    source.setpos(start_frame)
                    path = chunk_dir / f"chunk_{index:04d}.wav"
                    with wave.open(str(path), "wb") as target:
                        target.setparams(source.getparams())
                        target.writeframes(source.readframes(end_frame - start_frame))
                    chunks.append((str(path), start_frame / frame_rate, end_frame / frame_rate))
                    if end_frame == total_frames:
                        break
        except (wave.Error, OSError) as exc:
            raise ProviderError(f"Could not write WAV chunks: {exc}") from exc
        return chunks

    def _reconcile_overlaps(self, segments: Sequence[Segment]) -> List[Segment]:
        reconciled: List[Segment] = []
        for segment in sorted(segments, key=lambda item: (item.start, item.end)):
            duplicate = next((
                existing for existing in reconciled
                if abs(existing.start - segment.start) <= 1.0
                and abs(existing.end - segment.end) <= 1.0
                and SequenceMatcher(None, self._normalized(existing.text), self._normalized(segment.text)).ratio() >= 0.85
            ), None)
            if duplicate is None:
                reconciled.append(segment)
        return reconciled

    @staticmethod
    def _normalized(text: str) -> str:
        return " ".join(text.casefold().split())

    def _exceeds_inline_limit(self, audio: bytes) -> bool:
        encoded_size = 4 * ((len(audio) + 2) // 3)
        return encoded_size > self.inline_payload_safety_limit_bytes

    @staticmethod
    def _read_bytes(path: str) -> bytes:
        try:
            return Path(path).read_bytes()
        except OSError as exc:
            raise ProviderError(f"Could not read WAV file '{path}': {exc}") from exc

    @staticmethod
    def _inspect_wav(path: str) -> Dict[str, Any]:
        try:
            with wave.open(str(path), "rb") as wav:
                frame_count, frame_rate = wav.getnframes(), wav.getframerate()
                channels, sample_width, compression_type = wav.getnchannels(), wav.getsampwidth(), wav.getcomptype()
        except (wave.Error, EOFError, OSError) as exc:
            raise GeminiValidationError(f"Invalid or corrupt WAV file: {exc}") from exc
        if (frame_rate <= 0 or frame_count <= 0 or channels <= 0 or sample_width <= 0
                or compression_type != "NONE"):
            raise GeminiValidationError("WAV must be non-empty, uncompressed PCM audio with valid format fields.")
        return {
            "frame_count": frame_count, "frame_rate": frame_rate, "channels": channels,
            "sample_width": sample_width, "compression_type": compression_type,
            "duration": frame_count / frame_rate,
        }

    def write_qualification_artifacts(
        self, job_id: str, segments: List[Segment], runtime_usage: Any = None, output_dir: str = "output"
    ) -> None:
        path = Path(output_dir) / job_id / "qualification" / "gemini_asr"
        path.mkdir(parents=True, exist_ok=True)
        usage = runtime_usage or self.runtime.usage_recorder.totals()
        payloads = {
            "transcript.json": [segment.to_dict() for segment in segments],
            "provenance.json": self.provenance,
            "usage.json": usage,
            "validation.json": self.validation,
            "comparison.json": {"reference": None, "segment_count": len(segments), "notes": "shadow only; not merged into production"},
        }
        for name, payload in payloads.items():
            with (path / name).open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2, default=str)
