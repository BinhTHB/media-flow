from __future__ import annotations

import base64
import hashlib
import io
from typing import Any, Dict, List, Optional, Sequence

from PIL import Image

from src.application.ports.ocr_provider import OCRProvider
from src.domain.entities.frame import Frame
from src.domain.entities.segment import Segment
from src.infrastructure.gemini.contracts import MODE_ROLE_PRIMARY, GeminiRequest
from src.infrastructure.gemini.exceptions import GeminiError
from src.infrastructure.gemini.runtime import GeminiRuntime
from src.infrastructure.gemini.structured_output import parse_structured
from src.shared.exceptions import ProviderError

_TEXT_TYPES = {"subtitle", "caption", "title", "scene_text", "other"}
_OUTPUT_TYPES = {"subtitle", "caption"}

class GeminiMultiStageOCRProvider(OCRProvider):
    """Gemini-only OCR adapter with detection, tracking, classification and transcription stages."""

    def __init__(
        self,
        runtime: Optional[GeminiRuntime] = None,
        model: str = "gemini-3.1-flash-lite",
        api_key: Optional[str] = None,
        verification_enabled: bool = False,
        tracking_iou_threshold: float = 0.3,
        crop_padding_ratio: float = 0.05,
        prompt_version: str = "gemini_ocr_multistage_v1",
        schema_version: str = "ocr_multistage_v1",
        api_surface: str = "generate_content",
    ) -> None:
        if not 0.0 <= tracking_iou_threshold <= 1.0:
            raise ValueError("tracking_iou_threshold must be in [0.0, 1.0].")
        if not 0.0 <= crop_padding_ratio < 0.5:
            raise ValueError("crop_padding_ratio must be in [0.0, 0.5).")
        self.runtime = runtime or GeminiRuntime.create()
        self.model = model
        self.api_key = api_key
        self.verification_enabled = verification_enabled
        self.tracking_iou_threshold = tracking_iou_threshold
        self.crop_padding_ratio = crop_padding_ratio
        self.prompt_version = prompt_version
        self.schema_version = schema_version
        self.api_surface = api_surface
        self.observations: List[Dict[str, Any]] = []
        self.provenance: List[Dict[str, Any]] = []
        self.rejected_regions: List[Dict[str, Any]] = []
        self.review_regions: List[Dict[str, Any]] = []

    def detect_text(self, frames: List[Frame]) -> List[Segment]:
        self.observations = []
        self.provenance = []
        self.rejected_regions = []
        self.review_regions = []
        regions = self._detect_regions(frames)
        tracks = self._track_regions(regions)
        return self._process_tracks(tracks)

    @staticmethod
    def _detection_prompt() -> str:
        return (
            "Detect visible text regions only. Do not read, transcribe, translate, or "
            "classify the text. Return only JSON: "
            '{"regions":[{"box_2d":[ymin,xmin,ymax,xmax]}]}. '
            "box_2d uses the inclusive 0..1000 scale."
        )

    @staticmethod
    def _classification_prompt() -> str:
        return (
            "Classify this cropped video text region. Return only JSON: "
            '{"text_type":"subtitle|caption|title|scene_text|other"}. '
            "Do not transcribe or translate."
        )

    @staticmethod
    def _transcription_prompt() -> str:
        return (
            "Transcribe visible characters exactly from this cropped image. Do not "
            "translate, correct, infer, or add text. Return only JSON: "
            '{"text_verbatim":"string","transcription_status":"readable|ambiguous"}.'
        )

    @staticmethod
    def _verification_prompt() -> str:
        return (
            "Verify the visible text in this cropped image. Transcribe exactly, without "
            "translation, correction, inference, or added text. Return only JSON: "
            '{"text_verbatim":"string","transcription_status":"readable|ambiguous"}.'
        )

    def _detect_regions(self, frames: Sequence[Frame]) -> List[Dict[str, Any]]:
        regions: List[Dict[str, Any]] = []
        for frame_index, frame in enumerate(frames):
            image_bytes = self._read_bytes(frame.image_path)
            response = self._send(
                self._detection_prompt(),
                image_bytes,
                "stage_a_region_detection",
                f"frame_{frame_index}",
            )
            payload = self._parse(response.text, "stage A")
            entries = payload.get("regions") if isinstance(payload, dict) else None
            if not isinstance(entries, list):
                raise ProviderError("Gemini OCR stage A response requires a regions list.")
            width, height = self._image_size(frame.image_path)
            for region_index, entry in enumerate(entries):
                box = self._validate_box(entry.get("box_2d") if isinstance(entry, dict) else None)
                regions.append(
                    {
                        "region_id": f"frame_{frame_index}_region_{region_index}",
                        "frame_id": f"frame_{frame_index}",
                        "frame": frame,
                        "frame_index": frame_index,
                        "box_2d": box,
                        "pixel_box": self._to_pixel_box(box, width, height),
                        "stages": ["A"],
                    }
                )
        return regions

    def _track_regions(self, regions: Sequence[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        tracks: List[List[Dict[str, Any]]] = []
        for region in regions:
            matched: Optional[List[Dict[str, Any]]] = None
            for track in reversed(tracks):
                prior = track[-1]
                if region["frame_index"] - prior["frame_index"] > 1:
                    break
                if self._iou(region["box_2d"], prior["box_2d"]) >= self.tracking_iou_threshold:
                    matched = track
                    break
            if matched is None:
                tracks.append([region])
            else:
                matched.append(region)
        return tracks

    def _process_tracks(self, tracks: Sequence[List[Dict[str, Any]]]) -> List[Segment]:
        segments: List[Segment] = []
        for track_index, track in enumerate(tracks):
            representative = track[0]
            crop = self._make_crop(representative["frame"].image_path, representative["pixel_box"])
            classification = self._classify(crop, representative["region_id"])
            for item in track:
                item["track_id"] = f"track_{track_index}"
                item["stages"].extend(["tracking", "B"])
                item["text_type"] = classification
            if classification not in _OUTPUT_TYPES:
                self.rejected_regions.extend(self._artifact(item, "not_output_text_type") for item in track)
                continue
            transcription = self._transcribe(crop, representative["region_id"], "stage_c_crop_transcription")
            if transcription is None:
                self.review_regions.extend(self._artifact(item, "ambiguous_transcription") for item in track)
                continue
            for item in track:
                item["stages"].append("C")
            if self.verification_enabled:
                verified = self._transcribe(crop, representative["region_id"], "stage_d_verification")
                if verified is None or verified != transcription:
                    self.review_regions.extend(self._artifact(item, "verification_failed") for item in track)
                    continue
                for item in track:
                    item["stages"].append("D")
            for region_index, item in enumerate(track):
                observation = self._artifact(item, "accepted")
                observation["text_verbatim"] = transcription
                self.observations.append(observation)
                left, top, right, bottom = item["pixel_box"]
                segments.append(
                    Segment(
                        id=f"ocr_{item['frame_index']}_{region_index}",
                        start=item["frame"].timestamp,
                        end=item["frame"].timestamp + 2.0,
                        text=transcription,
                        source="ocr",
                        confidence=0.0,
                        confidence_available=False,
                        bbox={"left": left, "top": top, "right": right, "bottom": bottom},
                        metadata={
                            "provider": self.__class__.__name__,
                            "model": self.model,
                            "confidence_kind": "unknown",
                            "text_type": classification,
                            "frame_id": item["frame_id"],
                            "track_id": item["track_id"],
                            "stage_provenance": list(item["stages"]),
                            "validation_evidence": {
                                "stage_a_bbox_valid": True,
                                "stage_b_output_type": True,
                                "stage_c_readable": True,
                                "stage_d_verified": self.verification_enabled,
                            },
                        },
                    )
                )
        return segments

    def _classify(self, crop: bytes, region_id: str) -> str:
        payload = self._parse(
            self._send(self._classification_prompt(), crop, "stage_b_classification", region_id).text,
            "stage B",
        )
        text_type = payload.get("text_type") if isinstance(payload, dict) else None
        if text_type not in _TEXT_TYPES:
            raise ProviderError("Gemini OCR stage B response has an invalid text_type.")
        return text_type

    def _transcribe(self, crop: bytes, region_id: str, source_kind: str) -> Optional[str]:
        prompt = self._verification_prompt() if source_kind == "stage_d_verification" else self._transcription_prompt()
        payload = self._parse(self._send(prompt, crop, source_kind, region_id).text, source_kind)
        if not isinstance(payload, dict) or payload.get("transcription_status") != "readable":
            return None
        text = payload.get("text_verbatim")
        if not isinstance(text, str) or not text.strip():
            return None
        return text

    def _send(self, prompt: str, image_bytes: bytes, source_kind: str, region_id: str):
        request = GeminiRequest(
            contents=[{"parts": [
                {"text": f"{prompt} source_id={region_id}"},
                {"inline_data": {"mime_type": "image/png", "data": base64.b64encode(image_bytes).decode("ascii")}},
            ]}],
            model=self.model,
            system_instruction=prompt,
            prompt_version=self.prompt_version,
            schema_version=self.schema_version,
            source_kind=source_kind,
            media_sha256=hashlib.sha256(image_bytes).hexdigest(),
            api_surface=self.api_surface,
            role=MODE_ROLE_PRIMARY,
        )
        try:
            response = self.runtime.get_client(
                model=self.model, role=MODE_ROLE_PRIMARY, api_key=self.api_key
            ).generate_content(request)
        except GeminiError as exc:
            raise ProviderError(f"Gemini OCR {source_kind} request failed: {exc}") from exc
        self.provenance.append({
            "stage": source_kind,
            "region_id": region_id,
            "model": response.model or self.model,
            "model_stage": response.model_stage,
            "api_surface": self.api_surface,
            "prompt_version": self.prompt_version,
            "schema_version": self.schema_version,
            "request_id": response.request_id,
            "cache_hit": response.cache_hit,
        })
        return response

    @staticmethod
    def _parse(text: str, stage: str) -> Any:
        try:
            return parse_structured(text)
        except GeminiError as exc:
            raise ProviderError(f"Gemini OCR {stage} response failed validation: {exc}") from exc

    @staticmethod
    def _validate_box(box: Any) -> List[float]:
        if not isinstance(box, list) or len(box) != 4 or any(not isinstance(value, (int, float)) for value in box):
            raise ProviderError("Gemini OCR stage A box_2d must contain four numeric values.")
        ymin, xmin, ymax, xmax = [float(value) for value in box]
        if min(ymin, xmin, ymax, xmax) < 0 or max(ymin, xmin, ymax, xmax) > 1000 or ymin >= ymax or xmin >= xmax:
            raise ProviderError("Gemini OCR stage A box_2d is outside 0..1000 or has invalid ordering.")
        return [ymin, xmin, ymax, xmax]

    @staticmethod
    def _to_pixel_box(box: Sequence[float], width: int, height: int) -> List[float]:
        ymin, xmin, ymax, xmax = box
        pixel_box = [xmin / 1000 * width, ymin / 1000 * height, xmax / 1000 * width, ymax / 1000 * height]
        if pixel_box[0] >= pixel_box[2] or pixel_box[1] >= pixel_box[3]:
            raise ProviderError("Gemini OCR stage A pixel box has zero area.")
        return pixel_box

    def _make_crop(self, image_path: str, pixel_box: Sequence[float]) -> bytes:
        left, top, right, bottom = pixel_box
        try:
            with Image.open(image_path) as image:
                pad_x = (right - left) * self.crop_padding_ratio
                pad_y = (bottom - top) * self.crop_padding_ratio
                crop = image.crop((max(0, left - pad_x), max(0, top - pad_y), min(image.width, right + pad_x), min(image.height, bottom + pad_y)))
                output = io.BytesIO()
                crop.save(output, format="PNG")
                return output.getvalue()
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
    def _image_size(path: str) -> tuple[int, int]:
        try:
            with Image.open(path) as image:
                return image.size
        except OSError as exc:
            raise ProviderError(f"Gemini OCR could not inspect frame '{path}': {exc}") from exc

    @staticmethod
    def _iou(first: Sequence[float], second: Sequence[float]) -> float:
        top = max(first[0], second[0])
        left = max(first[1], second[1])
        bottom = min(first[2], second[2])
        right = min(first[3], second[3])
        intersection = max(0.0, bottom - top) * max(0.0, right - left)
        first_area = (first[2] - first[0]) * (first[3] - first[1])
        second_area = (second[2] - second[0]) * (second[3] - second[1])
        return intersection / (first_area + second_area - intersection) if intersection else 0.0

    @staticmethod
    def _artifact(region: Dict[str, Any], status: str) -> Dict[str, Any]:
        return {
            "status": status,
            "region_id": region["region_id"],
            "track_id": region.get("track_id"),
            "frame_id": region["frame_id"],
            "timestamp": region["frame"].timestamp,
            "box_2d": region["box_2d"],
            "pixel_box": region["pixel_box"],
            "text_type": region.get("text_type"),
            "stage_provenance": list(region["stages"]),
        }
