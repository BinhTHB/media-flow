from __future__ import annotations

import hashlib
import json
import mimetypes
import time
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from src.application.ports.ocr_provider import VideoSubtitleOCRProvider
from src.domain.entities.segment import Segment
from src.infrastructure.gemini.contracts import MODE_ROLE_PRIMARY, GeminiRequest
from src.infrastructure.gemini.exceptions import GeminiError
from src.infrastructure.gemini.runtime import GeminiRuntime
from src.infrastructure.gemini.structured_output import parse_structured
from src.shared.exceptions import ProviderError


_SUBTITLE_CLASSES = {"dialogue_subtitle", "narration_subtitle"}
_ALL_CLASSES = _SUBTITLE_CLASSES | {
    "title_card", "scene_text", "watermark", "logo", "user_interface",
    "warning_overlay", "account_or_channel_label", "other", "review_required",
}


class GeminiDirectVideoOCRProvider(VideoSubtitleOCRProvider):
    """G.5 direct-video OCR adapter. It never receives or creates image inputs."""

    def __init__(
        self,
        runtime: Optional[GeminiRuntime] = None,
        model: str = "gemini-3.1-flash-lite",
        api_key: Optional[str] = None,
        discovery_fps: float = 6,
        classification_fps: float = 6,
        segmentation_fps: float = 12,
        geometry_fps: float = 12,
        transcription_fps: float = 12,
        media_resolution: str = "high",
        clip_padding_seconds: float = 0.40,
        max_retry_per_stage: int = 1,
        prompt_version: str = "gemini_direct_video_g5_v1",
        schema_version: str = "gemini_direct_video_g5_v2",
        target_language: str = "vi",
    ) -> None:
        if model != "gemini-3.1-flash-lite":
            raise ValueError("G.5 direct-video OCR requires gemini-3.1-flash-lite.")
        if clip_padding_seconds < 0:
            raise ValueError("clip_padding_seconds must be non-negative.")
        self.runtime = runtime or GeminiRuntime.create()
        self.model = model
        self.api_key = api_key
        self.fps = {
            "discovery": discovery_fps, "classification": classification_fps,
            "segmentation": segmentation_fps, "geometry": geometry_fps,
            "transcription": transcription_fps,
        }
        self.media_resolution = media_resolution
        self.clip_padding_seconds = clip_padding_seconds
        self.max_retry_per_stage = max_retry_per_stage
        self.prompt_version = prompt_version
        self.schema_version = schema_version
        self.target_language = target_language
        self.provenance: List[Dict[str, Any]] = []
        self.artifacts: Dict[str, Any] = {}
        self._render_eligibility: List[Dict[str, Any]] = []

    # ── public entry point ────────────────────────────────────────────────

    def extract_subtitles(
        self, video_path: Path, source_language: str | None
    ) -> List[Segment]:
        if not video_path.is_file():
            raise ProviderError(f"Direct-video OCR input does not exist: '{video_path}'.")
        self._reset()
        mime_type = mimetypes.guess_type(str(video_path))[0] or "video/mp4"
        if mime_type != "video/mp4":
            raise ProviderError("G.5 direct-video OCR currently requires video/mp4.")

        # --- read video metadata with FFprobe (allowed pre-inference) ---
        duration = self._probe_duration(str(video_path))
        if duration <= 0:
            raise ProviderError(f"Video duration must be positive, got {duration}.")
        self.artifacts["video_metadata"] = {
            "duration_seconds": duration,
            "source_fps": self._probe_fps(str(video_path)),
            "width": self._probe_width(str(video_path)),
            "height": self._probe_height(str(video_path)),
        }

        self.runtime.get_client(self.model, api_key=self.api_key)
        # ── Files API upload (once) ──
        try:
            file_handle = self.runtime.file_manager.upload_file(
                str(video_path), mime_type=mime_type
            )
        except GeminiError as exc:
            raise ProviderError(f"Gemini video upload failed: {exc}") from exc
        if not file_handle.uri:
            raise ProviderError("Gemini Files API upload returned no usable file URI.")
        # store upload_count / reuse_count snapshot at this point
        upload_count_snapshot = self.runtime.file_manager.upload_count
        reuse_count_snapshot = self.runtime.file_manager.reuse_count
        self.artifacts["video_upload"] = {
            "upload_count": upload_count_snapshot,
            "reuse_count": reuse_count_snapshot,
            "final_remote_state": file_handle.metadata.get("state"),
        }

        # ── FPS smoke test (gate) ──
        self.artifacts["fps_smoke_test"] = self._fps_smoke(
            file_handle.uri, mime_type, duration
        )
        if not self.artifacts["fps_smoke_test"]["effective"]:
            raise ProviderError("VIDEO_METADATA_FPS_NOT_EFFECTIVE")

        # ── V1: Discovery ──
        discovery = self._request(
            "v1_discovery", self._discovery_prompt(), self._discovery_schema(),
            file_handle.uri, mime_type, self.fps["discovery"],
            duration_seconds=duration,
        )
        candidates = self._validate_candidates(discovery, duration)
        self.artifacts["candidates"] = {"candidates": candidates}
        if not candidates:
            self._empty_artifacts()
            self._final_artifact(candidates)
            return []

        # ── V2: Classification ──
        classifications = self._request(
            "v2_classification", self._classification_prompt(candidates),
            self._classification_schema(), file_handle.uri, mime_type,
            self.fps["classification"], duration_seconds=duration,
        )
        accepted = self._validate_classifications(classifications, candidates)
        self.artifacts["classifications"] = {
            "classifications": classifications.get("classifications", [])
        }

        # ── V3: Temporal segmentation per candidate ──
        events: List[Dict[str, Any]] = []
        for candidate in accepted:
            clip_start = max(0.0, candidate["start_seconds"] - self.clip_padding_seconds)
            clip_end = min(duration, candidate["end_seconds"] + self.clip_padding_seconds)
            if clip_start >= clip_end:
                continue
            payload = self._request(
                "v3_segmentation", self._segmentation_prompt(candidate),
                self._segmentation_schema(), file_handle.uri, mime_type,
                self.fps["segmentation"], start=clip_start, end=clip_end,
                duration_seconds=duration,
            )
            events.extend(self._validate_events(payload, candidate, clip_start))
        self.artifacts["temporal_events"] = {"events": events}

        # ── V4: Geometry refinement per event ──
        geometry: Dict[str, Dict[str, Any]] = {}
        for event in events:
            clip_start = max(0.0, event["start_seconds_in_source"] - 0.25)
            clip_end = min(duration, event["end_seconds_in_source"] + 0.25)
            payload = self._request(
                "v4_geometry", self._geometry_prompt(event),
                self._geometry_schema(), file_handle.uri, mime_type,
                self.fps["geometry"], start=clip_start, end=clip_end,
                duration_seconds=duration,
            )
            geometry[event["event_id"]] = self._validate_geometry(payload, event)
        self.artifacts["geometry"] = {"events": list(geometry.values())}

        # ── V5: Transcription per event ──
        transcriptions: List[Dict[str, Any]] = []
        for event in events:
            item = geometry.get(event["event_id"])
            if item is None:
                continue
            clip_start = max(0.0, event["start_seconds_in_source"] - 0.25)
            clip_end = min(duration, event["end_seconds_in_source"] + 0.25)
            payload = self._request(
                "v5_transcription", self._transcription_prompt(event, item),
                self._transcription_schema(), file_handle.uri, mime_type,
                self.fps["transcription"], start=clip_start, end=clip_end,
                duration_seconds=duration,
            )
            transcription = self._validate_transcription(payload, event)
            if transcription is not None:
                transcriptions.append({**event, **item, **transcription})
        self.artifacts["transcriptions"] = {"events": transcriptions}

        # ── Translation ──
        translations: List[Dict[str, Any]] = []
        for item in transcriptions:
            payload = self._text_request(
                "translation",
                self._translation_prompt(
                    item["event_id"], item["text_verbatim"], source_language
                ),
                self._translation_schema(),
            )
            translated = self._validate_translation(
                payload, item["event_id"], item["text_verbatim"]
            )
            translations.append({
                **item,
                "translated_text": translated or "",
                "translation_status": "ok" if translated else "REVIEW_REQUIRED",
            })
        self.artifacts["translations"] = {"events": translations}

        # ── Build render eligibility records and final segment list ──
        self._build_render_eligibility(candidates, accepted, events, geometry,
                                        transcriptions, translations, duration)
        self._final_artifact(candidates)

        return [
            Segment(
                id=item["event_id"], start=item["start_seconds_in_source"],
                end=item["end_seconds_in_source"], text=item["text_verbatim"],
                translated_text=item["translated_text"],
                source="ocr", confidence=1.0, confidence_available=False,
                bbox=self._pixel_free_bbox(item["representative_box_2d"]),
                metadata={"provider": "gemini_video_direct", "model": self.model,
                          "event_id": item["event_id"],
                          "source_language": source_language,
                          "geometry_source": "v4_direct_video"},
            )
            for item in translations
            if item["translated_text"]
        ]

    def debug_artifacts(self) -> Dict[str, Any]:
        return dict(self.artifacts)

    # ── reset ────────────────────────────────────────────────────────────

    def _reset(self) -> None:
        self.provenance = []
        self.artifacts = {}
        self._render_eligibility = []

    # ── request helpers ──────────────────────────────────────────────────

    def _request(
        self, stage: str, prompt: str, schema: Dict[str, Any], uri: str,
        mime_type: str, fps: float, start: Optional[float] = None,
        end: Optional[float] = None, duration_seconds: Optional[float] = None,
    ) -> Dict[str, Any]:
        clamp_start = start
        clamp_end = end
        if duration_seconds is not None:
            if clamp_start is not None:
                clamp_start = max(0.0, clamp_start)
            if clamp_end is not None:
                clamp_end = min(duration_seconds, clamp_end)
            else:
                clamp_end = duration_seconds

        part: Dict[str, Any] = {
            "fileData": {"fileUri": uri, "mimeType": mime_type},
            "videoMetadata": {"fps": fps},
        }
        video_metadata = {"fps": fps}
        if clamp_start is not None:
            video_metadata["startOffset"] = f"{clamp_start:.3f}s"
        if clamp_end is not None:
            video_metadata["endOffset"] = f"{clamp_end:.3f}s"
        part["videoMetadata"] = video_metadata

        request = GeminiRequest(
            contents=[{"parts": [{"text": prompt}, part]}], model=self.model,
            generation_config={"responseMimeType": "application/json",
                               "responseSchema": schema,
                               "mediaResolution": self.media_resolution},
            system_instruction=prompt, prompt_version=self.prompt_version,
            schema_version=self.schema_version, source_kind=stage,
            media_sha256=hashlib.sha256(uri.encode("utf-8")).hexdigest(),
            api_surface="generate_content",
        )
        client = self.runtime.get_client(self.model, role=MODE_ROLE_PRIMARY,
                                          api_key=self.api_key)

        last_error: Optional[str] = None
        response_payload: Optional[Dict[str, Any]] = None
        max_attempts = 1 + self.max_retry_per_stage

        for attempt in range(max_attempts):
            try:
                # canonical serialized payload hash for countTokens + generate
                serialized_body = self._canonical_body(request)
                serialized_sha = hashlib.sha256(
                    json.dumps(serialized_body, sort_keys=True).encode("utf-8")
                ).hexdigest()
                request.predicted_tokens = client.count_tokens(request)
                counted_tokens = request.predicted_tokens
                response = client.generate_content(request)
                response_payload = parse_structured(response.text)
            except GeminiError as exc:
                last_error = f"Gemini direct-video {stage} attempt {attempt+1} failed: {exc}"
                if attempt < max_attempts - 1:
                    continue
                raise ProviderError(last_error) from exc

            if response_payload is None or not isinstance(response_payload, dict):
                last_error = f"Gemini direct-video {stage} response was not an object on attempt {attempt+1}"
                if attempt < max_attempts - 1:
                    response_payload = None
                    continue
                raise ProviderError(last_error)

            # success
            self.provenance.append({
                "stage": stage, "input_type": "video", "model": self.model,
                "mime_type": mime_type, "file_data_present": True,
                "video_metadata_present": True, "requested_fps": fps,
                "clip_start": clamp_start, "clip_end": clamp_end,
                "media_resolution": self.media_resolution, "image_part_count": 0,
                "local_ocr_calls": 0,
                "serialized_payload_sha256": serialized_sha,
                "count_tokens": counted_tokens,
                "actual_usage": response.usage.to_dict() if response.usage else {},
                "cache_hit": getattr(response, "cache_hit", False),
                "retry_attempt": attempt,
            })
            return response_payload

        raise ProviderError(last_error or f"Gemini direct-video {stage} exhausted retries.")

    def _fps_smoke(self, uri: str, mime_type: str,
                    duration: float) -> Dict[str, Any]:
        records = []
        for fps in (1, 6):
            result = self._request(
                "fps_smoke",
                "Return only {\"ok\":true}. Do not transcribe, classify, or translate.",
                {"type": "object", "properties": {"ok": {"type": "boolean"}},
                 "required": ["ok"]},
                uri, mime_type, fps, 0.0, min(6.0, duration),
                duration_seconds=duration,
            )
            records.append({"fps": fps, "response": result,
                            **self.provenance[-1]})
        effective = (
            all(item["video_metadata_present"] for item in records)
            and (
                records[0].get("count_tokens") != records[1].get("count_tokens")
                or records[0].get("actual_usage") != records[1].get("actual_usage")
            )
        )
        return {"requests": records, "effective": effective}

    def _text_request(
        self, stage: str, prompt: str, schema: Dict[str, Any]
    ) -> Dict[str, Any]:
        request = GeminiRequest(
            contents=[{"parts": [{"text": prompt}]}],
            model=self.model,
            generation_config={"responseMimeType": "application/json",
                               "responseSchema": schema},
            system_instruction=prompt,
            prompt_version=self.prompt_version,
            schema_version=self.schema_version,
            source_kind=stage,
            api_surface="generate_content",
        )
        client = self.runtime.get_client(self.model, role=MODE_ROLE_PRIMARY,
                                          api_key=self.api_key)
        last_error: Optional[str] = None
        response_payload: Optional[Dict[str, Any]] = None
        max_attempts = 1 + self.max_retry_per_stage

        for attempt in range(max_attempts):
            try:
                serialized_body = self._canonical_body(request)
                serialized_sha = hashlib.sha256(
                    json.dumps(serialized_body, sort_keys=True).encode("utf-8")
                ).hexdigest()
                request.predicted_tokens = client.count_tokens(request)
                counted_tokens = request.predicted_tokens
                response = client.generate_content(request)
                response_payload = parse_structured(response.text)
            except GeminiError as exc:
                last_error = f"Gemini direct-video {stage} text attempt {attempt+1} failed: {exc}"
                if attempt < max_attempts - 1:
                    continue
                raise ProviderError(last_error) from exc

            if response_payload is None or not isinstance(response_payload, dict):
                last_error = f"Gemini direct-video {stage} text response invalid on attempt {attempt+1}"
                if attempt < max_attempts - 1:
                    response_payload = None
                    continue
                raise ProviderError(last_error)

            self.provenance.append({
                "stage": stage, "input_type": "text", "model": self.model,
                "image_part_count": 0, "local_ocr_calls": 0,
                "serialized_payload_sha256": serialized_sha,
                "count_tokens": counted_tokens,
                "actual_usage": response.usage.to_dict() if response.usage else {},
                "cache_hit": getattr(response, "cache_hit", False),
                "retry_attempt": attempt,
            })
            return response_payload

        raise ProviderError(last_error or f"Gemini direct-video {stage} text exhausted retries.")

    @staticmethod
    def _canonical_body(request: GeminiRequest) -> Dict[str, Any]:
        body: Dict[str, Any] = {"contents": request.contents}
        if request.generation_config is not None:
            body["generationConfig"] = request.generation_config
        if request.system_instruction is not None:
            body["systemInstruction"] = {"parts": [{"text": request.system_instruction}]}
        if request.safety_settings is not None:
            body["safetySettings"] = request.safety_settings
        return body

    # ── bbox / geometry validation ──────────────────────────────────────

    @staticmethod
    def _validate_box(box: Any) -> List[float]:
        if not isinstance(box, list) or len(box) != 4:
            raise ProviderError("Direct-video bbox must have four values.")
        values = [float(value) for value in box]
        ymin, xmin, ymax, xmax = values
        if min(values) < 0 or max(values) > 1000 or ymin >= ymax or xmin >= xmax:
            raise ProviderError("Direct-video bbox is invalid.")
        if (ymax - ymin) * (xmax - xmin) > 450000:
            raise ProviderError("Direct-video bbox exceeds cover safety policy.")
        return values

    @staticmethod
    def _box_area(box: Sequence[float]) -> float:
        ymin, xmin, ymax, xmax = box
        return (ymax - ymin) * (xmax - xmin)

    # ── FFprobe helpers (pre-inference only) ────────────────────────────

    @staticmethod
    def _probe_duration(path: str) -> float:
        import subprocess
        try:
            result = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "default=noprint_wrappers=1:nokey=1", path],
                capture_output=True, text=True, timeout=30,
            )
            return float(result.stdout.strip())
        except Exception as exc:
            raise ProviderError(f"FFprobe duration failed: {exc}") from exc

    @staticmethod
    def _probe_fps(path: str) -> float:
        import subprocess
        try:
            result = subprocess.run(
                ["ffprobe", "-v", "error", "-select_streams", "v:0",
                 "-show_entries", "stream=r_frame_rate",
                 "-of", "default=noprint_wrappers=1:nokey=1", path],
                capture_output=True, text=True, timeout=30,
            )
            parts = result.stdout.strip().split("/")
            if len(parts) == 2:
                return float(parts[0]) / float(parts[1])
            return float(parts[0])
        except Exception:
            return 0.0

    @staticmethod
    def _probe_width(path: str) -> int:
        import subprocess
        try:
            result = subprocess.run(
                ["ffprobe", "-v", "error", "-select_streams", "v:0",
                 "-show_entries", "stream=width",
                 "-of", "default=noprint_wrappers=1:nokey=1", path],
                capture_output=True, text=True, timeout=30,
            )
            return int(result.stdout.strip())
        except Exception:
            return 0

    @staticmethod
    def _probe_height(path: str) -> int:
        import subprocess
        try:
            result = subprocess.run(
                ["ffprobe", "-v", "error", "-select_streams", "v:0",
                 "-show_entries", "stream=height",
                 "-of", "default=noprint_wrappers=1:nokey=1", path],
                capture_output=True, text=True, timeout=30,
            )
            return int(result.stdout.strip())
        except Exception:
            return 0

    # ── V1 validation ───────────────────────────────────────────────────

    def _validate_candidates(
        self, payload: Dict[str, Any], duration: float
    ) -> List[Dict[str, Any]]:
        items = payload.get("candidates")
        if not isinstance(items, list):
            raise ProviderError("V1 response requires candidates.")
        result, ids = [], set()
        for item in items:
            if not isinstance(item, dict):
                raise ProviderError("V1 candidate must be a dict.")
            cid = item.get("candidate_id")
            if not isinstance(cid, str):
                raise ProviderError("V1 candidate ID must be a string.")
            if cid in ids:
                raise ProviderError("V1 candidate IDs must be unique.")
            start, end = item.get("start_seconds"), item.get("end_seconds")
            if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
                raise ProviderError("V1 candidate timing must be numbers.")
            start_f, end_f = float(start), float(end)
            if start_f < 0 or end_f <= start_f or end_f > duration:
                raise ProviderError(
                    f"V1 candidate {cid} timing {start_f}-{end_f} outside "
                    f"valid range 0..{duration}."
                )
            ids.add(cid)
            box = self._validate_box(item.get("representative_box_2d"))
            result.append({
                **item,
                "candidate_id": cid,
                "start_seconds": start_f,
                "end_seconds": end_f,
                "representative_box_2d": box,
            })
        return result

    # ── V2 validation ───────────────────────────────────────────────────

    def _validate_classifications(
        self, payload: Dict[str, Any], candidates: Sequence[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        known = {item["candidate_id"]: item for item in candidates}
        accepted = []
        entries = payload.get("classifications")
        if not isinstance(entries, list):
            raise ProviderError("V2 response requires classifications list.")

        seen_ids: set[str] = set()
        for item in entries:
            if not isinstance(item, dict):
                raise ProviderError("V2 classification entry must be a dict.")
            cid = item.get("candidate_id")
            classification = item.get("classification")
            decision = item.get("decision")
            if cid not in known or classification not in _ALL_CLASSES or decision not in {"accepted", "rejected"}:
                raise ProviderError(f"V2 classification for {cid} is invalid.")
            if cid in seen_ids:
                raise ProviderError(f"V2 duplicate classification for candidate {cid}.")
            seen_ids.add(cid)
            if decision == "accepted" and classification in _SUBTITLE_CLASSES:
                accepted.append(known[cid])

        # completeness: every V1 candidate must have exactly one V2 classification
        missing = set(known) - seen_ids
        if missing:
            raise ProviderError(
                f"V2 classification completeness: missing classifications for "
                f"candidates: {', '.join(sorted(missing))}"
            )
        return accepted

    # ── V3 validation ──────────────────────────────────────────────────

    @staticmethod
    def _validate_events(
        payload: Dict[str, Any], candidate: Dict[str, Any], clip_start: float
    ) -> List[Dict[str, Any]]:
        if payload.get("candidate_id") != candidate["candidate_id"]:
            raise ProviderError("V3 response candidate_id mismatch.")
        entries = payload.get("events")
        if not isinstance(entries, list):
            raise ProviderError("V3 response must contain events list.")
        clip_duration = float(candidate["end_seconds"]) - float(candidate["start_seconds"])
        result: List[Dict[str, Any]] = []
        seen: set[str] = set()
        for item in entries:
            if not isinstance(item, dict):
                raise ProviderError("V3 event must be a dict.")
            event_id = item.get("event_id")
            rel_start = item.get("start_seconds_in_source")
            rel_end = item.get("end_seconds_in_source")
            if not isinstance(event_id, str) or not event_id or event_id in seen:
                raise ProviderError(f"V3 event id is invalid or duplicated: {event_id!r}")
            seen.add(event_id)
            if not isinstance(rel_start, (int, float)) or not isinstance(rel_end, (int, float)):
                raise ProviderError("V3 event timing is invalid.")
            rs, re = float(rel_start), float(rel_end)
            # If model returned values clearly within the candidate interval,
            # treat them as source-absolute. Otherwise assume clip-relative
            # and add clip_start to derive absolute source timeline.
            is_source_absolute = (
                candidate["start_seconds"] - 0.001 <= rs
                and re <= candidate["end_seconds"] + 0.001
            )
            if is_source_absolute:
                absolute_start = rs
                absolute_end = re
                clip_relative_start = rs - clip_start
                clip_relative_end = re - clip_start
            else:
                clip_relative_start = max(0.0, rs)
                clip_relative_end = min(clip_duration, re)
                absolute_start = clip_start + clip_relative_start
                absolute_end = clip_start + clip_relative_end
            if absolute_end <= absolute_start:
                raise ProviderError(f"V3 event {event_id} has zero or negative duration.")
            if absolute_start < 0 or absolute_end > candidate["end_seconds"] + 0.001:
                raise ProviderError(f"V3 event {event_id} timing escapes clip range.")
            result.append({
                **item,
                "candidate_id": candidate["candidate_id"],
                "clip_source_start": clip_start,
                "clip_relative_start": clip_relative_start,
                "clip_relative_end": clip_relative_end,
                "source_absolute_start": absolute_start,
                "source_absolute_end": absolute_end,
                "start_seconds_in_source": absolute_start,
                "end_seconds_in_source": absolute_end,
            })
        return result

    # ── V4 validation ───────────────────────────────────────────────────

    def _validate_geometry(
        self, payload: Dict[str, Any], event: Dict[str, Any]
    ) -> Dict[str, Any]:
        if payload.get("event_id") != event["event_id"]:
            raise ProviderError("V4 event ID mismatch.")
        box = self._validate_box(payload.get("representative_box_2d"))
        # safety: no oversized cover
        area = self._box_area(box)
        if area > 450000:
            raise ProviderError("V4 geometry exceeds cover safety policy.")
        return {
            "event_id": event["event_id"],
            "representative_box_2d": box,
            "box_keyframes": payload.get("box_keyframes", []),
        }

    # ── V5 validation ───────────────────────────────────────────────────

    @staticmethod
    def _validate_transcription(
        payload: Dict[str, Any], event: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        if payload.get("event_id") != event["event_id"]:
            raise ProviderError("V5 event ID mismatch.")
        text = payload.get("text_verbatim")
        status = payload.get("status")
        if status != "readable":
            return None
        if not isinstance(text, str) or not text.strip():
            return None
        cleaned = text.strip()
        if "\ufffd" in cleaned or any(unicodedata.category(c).startswith("C")
                                       for c in cleaned):
            return None
        # mixed-script check: only one substantive script (+ common)
        scripts = set()
        for char in cleaned:
            if not char.isalpha():
                continue
            name = unicodedata.name(char, "")
            if any(tag in name for tag in ("CJK UNIFIED", "HIRAGANA", "KATAKANA", "HANGUL")):
                scripts.add("cjk")
            elif "LATIN" in name:
                scripts.add("latin")
        if len(scripts - {"common"}) > 1:
            return None
        return {
            "text_verbatim": cleaned,
            "language": payload.get("language"),
            "script": payload.get("script"),
            "status": "readable",
        }

    # ── translation ──────────────────────────────────────────────────────

    @staticmethod
    def _validate_translation(
        payload: Dict[str, Any], event_id: str, source_text: str
    ) -> Optional[str]:
        if payload.get("event_id") != event_id:
            raise ProviderError("Translation event ID mismatch.")
        text = payload.get("translated_text")
        if not isinstance(text, str) or not text.strip():
            return None
        text = text.strip()
        source_copy = bool(payload.get("is_source_copy"))
        reason = payload.get("reason_for_source_copy")
        if source_copy and text != source_text:
            return None
        if text == source_text and (not source_copy or not isinstance(reason, str) or not reason.strip()):
            return None
        return text

    # ── render eligibility ──────────────────────────────────────────────

    def _build_render_eligibility(
        self,
        candidates: List[Dict[str, Any]],
        accepted_candidates: List[Dict[str, Any]],
        events: List[Dict[str, Any]],
        geometry: Dict[str, Dict[str, Any]],
        transcriptions: List[Dict[str, Any]],
        translations: List[Dict[str, Any]],
        duration: float,
    ) -> None:
        accepted_ids = {c["candidate_id"] for c in accepted_candidates}
        event_map = {e["event_id"]: e for e in events}
        geom_map = dict(geometry)
        tr_map = {t["event_id"]: t for t in transcriptions}
        tl_map = {t["event_id"]: t for t in translations}

        for c in candidates:
            cid = c["candidate_id"]
            c_valid = (
                "start_seconds" in c and "end_seconds" in c
                and 0 <= c["start_seconds"] < c["end_seconds"] <= duration
            )
            c_accepted = cid in accepted_ids
            # each event under this candidate
            for e in events:
                if e.get("candidate_id") != cid:
                    continue
                eid = e["event_id"]
                g = geom_map.get(eid)
                tr = tr_map.get(eid)
                tl = tl_map.get(eid)
                record = {
                    "event_id": eid,
                    "candidate_valid": c_valid,
                    "classification_valid": c_accepted,
                    "timing_valid": e["end_seconds_in_source"] > e["start_seconds_in_source"],
                    "geometry_valid": g is not None and "representative_box_2d" in g,
                    "transcription_valid": tr is not None and tr.get("status") == "readable",
                    "translation_valid": tl is not None and bool(tl.get("translated_text")),
                    "cover_safety_valid": g is not None
                        and self._box_area(g.get("representative_box_2d", [0, 0, 1, 1])) <= 450000,
                    "person_overlap_valid": False,
                    "render_eligible": False,
                    "reason_codes": [],
                }
                reasons = []
                if not c_valid:
                    reasons.append("INVALID_CANDIDATE")
                if not c_accepted:
                    reasons.append("CLASSIFICATION_REJECTED")
                if not record["timing_valid"]:
                    reasons.append("INVALID_TIMING")
                if not record["geometry_valid"]:
                    reasons.append("GEOMETRY_FAILED")
                if not record["transcription_valid"]:
                    reasons.append("TRANSCRIPTION_FAILED")
                if not record["translation_valid"]:
                    reasons.append("TRANSLATION_FAILED")
                if not record["cover_safety_valid"]:
                    reasons.append("COVER_SAFETY_FAILED")
                if not reasons:
                    record["render_eligible"] = True
                else:
                    record["reason_codes"] = reasons
                self._render_eligibility.append(record)

        self.artifacts["render_eligibility"] = {"events": self._render_eligibility}

    # ── final artifact ──────────────────────────────────────────────────

    def _final_artifact(self, candidates: List[Dict[str, Any]]) -> None:
        paths = self.provenance
        eligible = [e for e in self._render_eligibility if e.get("render_eligible")]
        self.artifacts["final"] = {
            "total_requests": len(paths),
            "no_frame_input": all(
                p.get("image_part_count", 0) == 0 and p.get("input_type") == "video"
                for p in paths
                if p.get("input_type") == "video"
            ),
            "no_local_ocr": all(p.get("local_ocr_calls", 0) == 0 for p in paths),
            "total_accepted_events": len(eligible),
            "v1_candidate_count": len(candidates),
        }

    # ── prompts ─────────────────────────────────────────────────────────

    @staticmethod
    def _discovery_prompt() -> str:
        return (
            "Analyze the supplied video in chronological order. "
            "Locate every temporally changing text-overlay region that may contain "
            "viewer-facing subtitles. Return candidate text tracks with: candidate ID; "
            "approximate start and end time; representative bounding box. "
            "Do not classify the text as subtitle, title, watermark, or scene text. "
            "Do not transcribe the text. Do not translate. "
            "Do not combine spatially separate regions into one candidate."
        )

    @staticmethod
    def _classification_prompt(candidates: Sequence[Dict[str, Any]]) -> str:
        return (
            "Classify each supplied candidate text track using the original video. "
            "Choose exactly one classification: dialogue_subtitle, narration_subtitle, "
            "title_card, scene_text, watermark, logo, user_interface, warning_overlay, "
            "account_or_channel_label, other, review_required. "
            "Do not transcribe. Do not translate. "
            "Do not modify candidate timing or bounding boxes.\n"
            + json.dumps(candidates, ensure_ascii=False)
        )

    @staticmethod
    def _segmentation_prompt(candidate: Dict[str, Any]) -> str:
        return (
            "This video clip contains one accepted subtitle track. "
            "Identify each temporal interval during which the visible subtitle text "
            "remains unchanged. Return intervals only with source-absolute timestamps. "
            "Do not transcribe. Do not classify. Do not translate.\n"
            + json.dumps(candidate)
        )

    @staticmethod
    def _geometry_prompt(event: Dict[str, Any]) -> str:
        return (
            "Refine the bounding box of the supplied accepted subtitle event. "
            "Return a representative box and keyframes only when geometry materially "
            "changes. Do not transcribe. Do not classify. Do not translate.\n"
            + json.dumps(event)
        )

    @staticmethod
    def _transcription_prompt(
        event: Dict[str, Any], geometry: Dict[str, Any]
    ) -> str:
        return (
            "Transcribe only the subtitle text inside the supplied bounding box "
            "during this single stable subtitle event. Preserve original characters, "
            "punctuation, reading order, line breaks. "
            "Do not translate, classify, change timestamps, return a new bbox, "
            "or read text outside the supplied region. "
            "Return REVIEW_REQUIRED if text is not stable or readable.\n"
            + json.dumps({"event": event, "bbox": geometry["representative_box_2d"]})
        )

    def _translation_prompt(
        self, event_id: str, text: str, source_language: Optional[str]
    ) -> str:
        lang = source_language or "unknown"
        return (
            f"Translate event {event_id} from {lang} to {self.target_language}. "
            "Translate naturally without summarizing or omitting content. "
            "Return source-copy only for intentional proper-name copy and explain why.\n"
            f"Source text: {text}"
        )

    # ── schemas ─────────────────────────────────────────────────────────

    @staticmethod
    def _discovery_schema() -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "video_id": {"type": "string"},
                "candidates": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "candidate_id": {"type": "string"},
                            "start_seconds": {"type": "number"},
                            "end_seconds": {"type": "number"},
                            "representative_box_2d": {
                                "type": "array", "items": {"type": "number"},
                            },
                        },
                        "required": ["candidate_id", "start_seconds",
                                      "end_seconds", "representative_box_2d"],
                    },
                },
            },
            "required": ["candidates"],
        }

    @staticmethod
    def _classification_schema() -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "classifications": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "candidate_id": {"type": "string"},
                            "classification": {"type": "string"},
                            "decision": {"type": "string"},
                        },
                        "required": ["candidate_id", "classification", "decision"],
                    },
                },
            },
            "required": ["classifications"],
        }

    @staticmethod
    def _segmentation_schema() -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "candidate_id": {"type": "string"},
                "events": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "event_id": {"type": "string"},
                            "start_seconds_in_source": {"type": "number"},
                            "end_seconds_in_source": {"type": "number"},
                        },
                        "required": ["event_id", "start_seconds_in_source",
                                      "end_seconds_in_source"],
                    },
                },
            },
            "required": ["candidate_id", "events"],
        }

    @staticmethod
    def _geometry_schema() -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "event_id": {"type": "string"},
                "representative_box_2d": {
                    "type": "array", "items": {"type": "number"},
                },
                "box_keyframes": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "timestamp_seconds": {"type": "number"},
                            "box_2d": {"type": "array", "items": {"type": "number"}},
                        },
                    },
                },
            },
            "required": ["event_id", "representative_box_2d"],
        }

    @staticmethod
    def _transcription_schema() -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "event_id": {"type": "string"},
                "text_verbatim": {"type": "string"},
                "language": {"type": "string"},
                "script": {"type": "string"},
                "status": {"type": "string"},
            },
            "required": ["event_id", "text_verbatim", "language", "script", "status"],
        }

    @staticmethod
    def _translation_schema() -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "event_id": {"type": "string"},
                "translated_text": {"type": "string"},
                "is_source_copy": {"type": "boolean"},
                "reason_for_source_copy": {"type": "string"},
            },
            "required": ["event_id", "translated_text"],
        }

    # ── helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _pixel_free_bbox(box: Sequence[float]) -> Dict[str, float]:
        ymin, xmin, ymax, xmax = box
        return {"left": xmin, "top": ymin, "right": xmax, "bottom": ymax}

    def _empty_artifacts(self) -> None:
        for name in ("classifications", "temporal_events", "geometry",
                      "transcriptions", "translations", "render_eligibility"):
            self.artifacts.setdefault(name, {"events": []})
