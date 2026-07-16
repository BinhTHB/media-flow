from __future__ import annotations

import base64
import hashlib
import io
import json
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

from PIL import Image, ImageEnhance

from src.application.ports.ocr_provider import OCRProvider
from src.domain.entities.frame import Frame
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

_ALLOWED_TEXT_TYPES = {"subtitle", "caption", "title", "scene_text", "unknown"}
_READABLE = "readable"
_FLOAT_EPSILON = 1e-6


class GeminiVisionOCRProvider(OCRProvider):
    """Gemini OCR adapter intended for qualification-only shadow runs."""

    def __init__(
        self,
        runtime: Optional[GeminiRuntime] = None,
        model: str = "gemini-3.1-flash-lite",
        escalation_model: str = "gemini-3.5-flash",
        api_key: Optional[str] = None,
        batch_size: int = 4,
        prompt_version: str = "gemini_ocr_v1",
        schema_version: str = "ocr_detection_v1",
        api_surface: str = "generate_content",
        max_escalation_requests: int = 2,
        escalation_enabled: bool = False,
        crop_retry_enabled: bool = True,
        crop_padding_ratio: float = 0.05,
        crop_upscale_factor: float = 2.0,
        enable_contrast_norm: bool = False,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1.")
        if not 0.0 <= crop_padding_ratio < 0.5:
            raise ValueError("crop_padding_ratio must be in [0.0, 0.5).")
        if crop_upscale_factor < 1.0:
            raise ValueError("crop_upscale_factor must be at least 1.0.")
        if max_escalation_requests < 0:
            raise ValueError("max_escalation_requests must be non-negative.")

        self.runtime = runtime or GeminiRuntime.create()
        self.model = model
        self.escalation_model = escalation_model
        self.api_key = api_key
        self.batch_size = batch_size
        self.prompt_version = prompt_version
        self.schema_version = schema_version
        self.api_surface = api_surface
        self.max_escalation_requests = max_escalation_requests
        self.escalation_enabled = escalation_enabled
        self.crop_retry_enabled = crop_retry_enabled
        self.crop_padding_ratio = crop_padding_ratio
        self.crop_upscale_factor = crop_upscale_factor
        self.enable_contrast_norm = enable_contrast_norm
        self.observations: List[Dict[str, Any]] = []
        self.provenance: List[Dict[str, Any]] = []
        self._escalation_requests = 0

    @staticmethod
    def build_prompt() -> str:
        return (
            "Transcribe visible subtitle characters exactly.\n"
            "Do not translate.\n"
            "Do not correct grammar or spelling.\n"
            "Do not infer missing or obscured characters.\n"
            "Do not copy watermarks, logos, UI labels or general scene text as subtitles.\n"
            "Return an empty detections list when no subtitle is visible.\n"
            "Return only JSON matching this schema:\n"
            '{"frames":[{"frame_id":"string","timestamp":number,"detections":'
            '[{"text_verbatim":"string","language":"string","text_type":'
            '"subtitle|caption|title|scene_text|unknown","box_2d":'
            "[ymin,xmin,ymax,xmax],"
            '"transcription_status":"readable|ambiguous"}]}]}. '
            "box_2d coordinates are [ymin,xmin,ymax,xmax] on the inclusive 0..1000 scale."
        )

    def detect_text(self, frames: List[Frame]) -> List[Segment]:
        self.observations = []
        self.provenance = []
        self._escalation_requests = 0
        segments: List[Segment] = []
        for offset in range(0, len(frames), self.batch_size):
            batch = frames[offset : offset + self.batch_size]
            response = self._request_batch(batch, offset)
            try:
                payload = parse_structured(response.text)
                segments.extend(self._process_batch(payload, batch, offset))
            except GeminiError as exc:
                raise ProviderError(f"Gemini OCR response failed validation: {exc}") from exc
        return segments

    def _request_batch(self, frames: Sequence[Frame], offset: int):
        parts: List[Dict[str, Any]] = [{"text": self.build_prompt()}]
        blobs: List[bytes] = []
        for index, frame in enumerate(frames, start=offset):
            blob = self._read_bytes(frame.image_path)
            blobs.append(blob)
            parts.extend(
                [
                    {"text": f"frame_id=frame_{index}; timestamp={frame.timestamp}"},
                    {
                        "inline_data": {
                            "mime_type": self._mime_type(frame.image_path),
                            "data": base64.b64encode(blob).decode("ascii"),
                        }
                    },
                ]
            )
        return self._send(
            model=self.model,
            role=MODE_ROLE_PRIMARY,
            contents=[{"parts": parts}],
            media=blobs,
            source_kind="image",
        )

    def _process_batch(
        self, payload: Any, frames: Sequence[Frame], offset: int
    ) -> List[Segment]:
        if not isinstance(payload, dict) or not isinstance(payload.get("frames"), list):
            raise GeminiValidationError("Gemini OCR response requires a frames list.")

        expected = {f"frame_{offset + index}": frame for index, frame in enumerate(frames)}
        segments: List[Segment] = []
        seen: set[Tuple[str, str, Tuple[float, ...]]] = set()
        for frame_output in payload["frames"]:
            frame_id, frame, detections = self._validate_frame_output(frame_output, expected)
            width, height = self._image_size(frame.image_path)
            for detection_index, detection in enumerate(detections):
                text, box = self._validate_detection(detection)
                key = (frame_id, text, tuple(float(value) for value in box))
                if key in seen:
                    self.observations.append(
                        {"frame_id": frame_id, "review_flags": ["duplicate_dropped"]}
                    )
                    continue
                seen.add(key)
                pixel_box = self._to_pixel_box(box, width, height)
                observation = {
                    "frame_id": frame_id,
                    "timestamp": frame.timestamp,
                    "text_verbatim": text,
                    "language": detection.get("language", ""),
                    "text_type": detection["text_type"],
                    "box_2d": box,
                    "pixel_box": pixel_box,
                    "review_flags": [],
                }

                if self._looks_translated(text, detection.get("language")):
                    observation["review_flags"].append("translation_suspected")
                    observation["review_flags"].append("REVIEW_REQUIRED")
                    self.observations.append(observation)
                    continue

                if detection.get("transcription_status") != _READABLE:
                    resolved = self._resolve_ambiguous_detection(
                        frame, frame_id, detection, pixel_box, observation
                    )
                    if resolved is None:
                        self.observations.append(observation)
                        continue
                    text = resolved
                    observation["text_verbatim"] = text

                self.observations.append(observation)
                segments.append(
                    Segment(
                        id=f"ocr_{frame_id.removeprefix('frame_')}_{detection_index}",
                        start=frame.timestamp,
                        end=frame.timestamp + 2.0,
                        text=text,
                        source="ocr",
                        confidence=0.0,
                        confidence_available=False,
                        bbox={
                            "left": pixel_box[0],
                            "top": pixel_box[1],
                            "right": pixel_box[2],
                            "bottom": pixel_box[3],
                        },
                        metadata={
                            "provider": self.__class__.__name__,
                            "model": self.model,
                            "confidence_kind": "unknown",
                            "text_type": "subtitle",
                            "frame_id": frame_id,
                            "frame_timestamp": frame.timestamp,
                            "validation_evidence": {
                                "structured_schema_valid": True,
                                "bbox_valid": True,
                                "frame_mapping_valid": True,
                            },
                        },
                    )
                )
        return segments

    @staticmethod
    def _validate_frame_output(
        item: Any, expected: Dict[str, Frame]
    ) -> Tuple[str, Frame, List[Dict[str, Any]]]:
        if not isinstance(item, dict):
            raise GeminiValidationError("Gemini OCR frame entry must be an object.")
        frame_id = item.get("frame_id")
        if frame_id not in expected:
            raise GeminiValidationError("Gemini OCR returned an unknown frame_id.")
        frame = expected[frame_id]
        timestamp = item.get("timestamp")
        if not isinstance(timestamp, (int, float)) or abs(timestamp - frame.timestamp) > _FLOAT_EPSILON:
            raise GeminiValidationError("Gemini OCR returned a mismatched frame timestamp.")
        detections = item.get("detections")
        if not isinstance(detections, list):
            raise GeminiValidationError("Gemini OCR frame detections must be a list.")
        if not all(isinstance(item, dict) for item in detections):
            raise GeminiValidationError("Gemini OCR detection must be an object.")
        return frame_id, frame, detections

    @staticmethod
    def _validate_detection(detection: Dict[str, Any]) -> Tuple[str, List[float]]:
        text = detection.get("text_verbatim")
        box = detection.get("box_2d")
        if not isinstance(text, str) or not text.strip():
            raise GeminiValidationError("Gemini OCR detection text_verbatim must be non-empty.")
        if detection.get("text_type") not in _ALLOWED_TEXT_TYPES:
            raise GeminiValidationError("Gemini OCR detection text_type is invalid.")
        if (
            not isinstance(box, list)
            or len(box) != 4
            or any(not isinstance(value, (int, float)) for value in box)
            or any(value < 0 or value > 1000 for value in box)
        ):
            raise GeminiValidationError("Gemini OCR box_2d must contain four values in 0..1000.")
        ymin, xmin, ymax, xmax = [float(value) for value in box]
        if xmin >= xmax or ymin >= ymax:
            raise GeminiValidationError("Gemini OCR box_2d has zero area or invalid ordering.")
        return text, [ymin, xmin, ymax, xmax]

    def _resolve_ambiguous_detection(
        self,
        frame: Frame,
        frame_id: str,
        detection: Dict[str, Any],
        pixel_box: List[float],
        observation: Dict[str, Any],
    ) -> Optional[str]:
        observation["review_flags"].append("primary_ambiguous")
        if not self.crop_retry_enabled:
            observation["review_flags"].append("REVIEW_REQUIRED")
            return None
        crop_bytes = self._make_crop(frame.image_path, pixel_box)
        crop_text = self._request_crop(crop_bytes, frame_id, MODE_ROLE_PRIMARY, self.model)
        if crop_text is not None:
            observation["review_flags"].append("crop_retry_resolved")
            return crop_text
        if not self.escalation_enabled or self._escalation_requests >= self.max_escalation_requests:
            observation["review_flags"].append("REVIEW_REQUIRED")
            return None
        self._escalation_requests += 1
        escalation_text = self._request_crop(
            crop_bytes, frame_id, MODE_ROLE_ESCALATION, self.escalation_model
        )
        if escalation_text is None or escalation_text != detection["text_verbatim"]:
            observation["review_flags"].append("REVIEW_REQUIRED")
            observation["review_flags"].append("escalation_conflict_or_invalid")
            return None
        observation["review_flags"].append("escalation_resolved")
        return escalation_text

    def _request_crop(
        self, crop_bytes: bytes, frame_id: str, role: str, model: str
    ) -> Optional[str]:
        prompt = (
            "Transcribe visible subtitle characters exactly from this cropped image. "
            "Do not translate, correct, infer, or add text. Return only JSON: "
            '{"text_verbatim":"string","transcription_status":"readable|ambiguous"}.'
        )
        response = self._send(
            model=model,
            role=role,
            contents=[
                {
                    "parts": [
                        {"text": f"{prompt} source_frame_id={frame_id}"},
                        {
                            "inline_data": {
                                "mime_type": "image/png",
                                "data": base64.b64encode(crop_bytes).decode("ascii"),
                            }
                        },
                    ]
                }
            ],
            media=[crop_bytes],
            source_kind="image_crop",
        )
        try:
            payload = parse_structured(response.text)
        except GeminiError:
            return None
        if not isinstance(payload, dict) or payload.get("transcription_status") != _READABLE:
            return None
        text = payload.get("text_verbatim")
        if not isinstance(text, str) or not text.strip() or self._looks_translated(text, None):
            return None
        return text

    def _send(
        self,
        *,
        model: str,
        role: str,
        contents: List[Dict[str, Any]],
        media: Sequence[bytes],
        source_kind: str,
    ):
        request = GeminiRequest(
            contents=contents,
            model=model,
            system_instruction=self.build_prompt(),
            prompt_version=self.prompt_version,
            schema_version=self.schema_version,
            source_kind=source_kind,
            media_sha256=hashlib.sha256(b"".join(media)).hexdigest(),
            api_surface=self.api_surface,
            role=role,
        )
        try:
            response = self.runtime.get_client(
                model=model, role=role, api_key=self.api_key
            ).generate_content(request)
        except GeminiError as exc:
            raise ProviderError(f"Gemini OCR request failed: {exc}") from exc
        self.provenance.append(
            {
                "model": response.model or model,
                "model_stage": response.model_stage,
                "api_surface": self.api_surface,
                "prompt_version": self.prompt_version,
                "schema_version": self.schema_version,
                "request_id": response.request_id,
                "cache_hit": response.cache_hit,
                "source_kind": source_kind,
            }
        )
        return response

    def _make_crop(self, image_path: str, pixel_box: Sequence[float]) -> bytes:
        left, top, right, bottom = pixel_box
        try:
            with Image.open(image_path) as image:
                pad_x = (right - left) * self.crop_padding_ratio
                pad_y = (bottom - top) * self.crop_padding_ratio
                crop = image.crop(
                    (
                        max(0.0, left - pad_x),
                        max(0.0, top - pad_y),
                        min(float(image.width), right + pad_x),
                        min(float(image.height), bottom + pad_y),
                    )
                )
                size = (
                    max(1, round(crop.width * self.crop_upscale_factor)),
                    max(1, round(crop.height * self.crop_upscale_factor)),
                )
                crop = crop.resize(size, Image.Resampling.LANCZOS)
                if self.enable_contrast_norm:
                    crop = ImageEnhance.Contrast(crop).enhance(1.5)
                result = io.BytesIO()
                crop.save(result, format="PNG")
                return result.getvalue()
        except OSError as exc:
            raise ProviderError(f"Gemini OCR crop failed for '{image_path}': {exc}") from exc

    @staticmethod
    def _read_bytes(path: str) -> bytes:
        try:
            with open(path, "rb") as handle:
                return handle.read()
        except OSError as exc:
            raise ProviderError(f"Gemini OCR could not read frame '{path}': {exc}") from exc

    @staticmethod
    def _image_size(path: str) -> Tuple[int, int]:
        try:
            with Image.open(path) as image:
                return image.size
        except OSError as exc:
            raise ProviderError(f"Gemini OCR could not inspect frame '{path}': {exc}") from exc

    @staticmethod
    def _mime_type(path: str) -> str:
        return "image/png" if os.path.splitext(path)[1].lower() == ".png" else "image/jpeg"

    @staticmethod
    def _to_pixel_box(box: Sequence[float], width: int, height: int) -> List[float]:
        ymin, xmin, ymax, xmax = box
        left = xmin / 1000.0 * width
        top = ymin / 1000.0 * height
        right = xmax / 1000.0 * width
        bottom = ymax / 1000.0 * height
        values = [
            min(max(left, 0.0), float(width)),
            min(max(top, 0.0), float(height)),
            min(max(right, 0.0), float(width)),
            min(max(bottom, 0.0), float(height)),
        ]
        if values[2] <= values[0] or values[3] <= values[1]:
            raise GeminiValidationError("Gemini OCR pixel box has zero area.")
        return values

    @staticmethod
    def _looks_translated(text: str, language: Any) -> bool:
        if language not in {"zh", "ja", "ko", "zh-CN", "zh-TW"}:
            return False
        return not any(
            "\u4e00" <= char <= "\u9fff"
            or "\u3040" <= char <= "\u30ff"
            or "\uac00" <= char <= "\ud7a3"
            for char in text
        )

    def write_qualification_artifacts(
        self,
        job_id: str,
        frames: List[Frame],
        segments: List[Segment],
        runtime_usage: Any = None,
        output_dir: str = "output",
    ) -> None:
        path = os.path.join(output_dir, job_id, "qualification", "gemini_ocr")
        os.makedirs(path, exist_ok=True)
        usage = runtime_usage or self.runtime.usage_recorder.totals()
        payloads = {
            "observations.json": self.observations,
            "provenance.json": self.provenance,
            "usage.json": usage,
            "comparison.json": {
                "paddle_reference": None,
                "gemini_observations": self.observations,
                "segment_count": len(segments),
                "notes": "shadow only; not merged into production",
            },
        }
        for name, payload in payloads.items():
            with open(os.path.join(path, name), "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
