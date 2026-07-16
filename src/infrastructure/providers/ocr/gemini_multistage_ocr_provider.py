from __future__ import annotations

import base64
import hashlib
import io
import statistics
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

_TEXT_TYPES = {
    "subtitle", "watermark", "logo", "user_interface", "scene_text",
    "warning_overlay", "other",
}

class GeminiMultiStageOCRProvider(OCRProvider):
    """Gemini-only OCR adapter with detection, tracking, classification and transcription stages."""

    def __init__(
        self,
        runtime: Optional[GeminiRuntime] = None,
        model: str = "gemini-3.1-flash-lite",
        stage_a_model: Optional[str] = None,
        stage_b_model: Optional[str] = None,
        stage_c_model: Optional[str] = None,
        stage_d_model: Optional[str] = None,
        stage_a_options: Optional[Dict[str, Any]] = None,
        stage_b_options: Optional[Dict[str, Any]] = None,
        stage_c_options: Optional[Dict[str, Any]] = None,
        stage_d_options: Optional[Dict[str, Any]] = None,
        api_key: Optional[str] = None,
        verification_enabled: bool = True,
        tracking_iou_threshold: float = 0.3,
        crop_padding_ratio: float = 0.05,
        window_size: int = 3,
        window_overlap: int = 1,
        prompt_version: str = "gemini_ocr_multistage_v1",
        schema_version: str = "ocr_multistage_v1",
        api_surface: str = "generate_content",
    ) -> None:
        if not 0.0 <= tracking_iou_threshold <= 1.0:
            raise ValueError("tracking_iou_threshold must be in [0.0, 1.0].")
        if not 0.0 <= crop_padding_ratio < 0.5:
            raise ValueError("crop_padding_ratio must be in [0.0, 0.5).")
        self.runtime = runtime or GeminiRuntime.create()
        if window_size < 2 or not 0 <= window_overlap < window_size:
            raise ValueError("window_size must be at least 2 and overlap must be smaller.")
        self.model = model
        self.stage_models = {
            "A": stage_a_model or model, "B": stage_b_model or model,
            "C": stage_c_model or model, "D": stage_d_model or model,
        }
        self.stage_options = {
            "A": dict(stage_a_options or {}), "B": dict(stage_b_options or {}),
            "C": dict(stage_c_options or {}), "D": dict(stage_d_options or {}),
        }
        self.api_key = api_key
        self.verification_enabled = verification_enabled
        self.tracking_iou_threshold = tracking_iou_threshold
        self.crop_padding_ratio = crop_padding_ratio
        self.window_size = window_size
        self.window_overlap = window_overlap
        self.prompt_version = prompt_version
        self.schema_version = schema_version
        self.api_surface = api_surface
        self.observations: List[Dict[str, Any]] = []
        self.provenance: List[Dict[str, Any]] = []
        self.rejected_regions: List[Dict[str, Any]] = []
        self.review_regions: List[Dict[str, Any]] = []
        self.region_windows: List[Dict[str, Any]] = []
        self.tracks: List[List[Dict[str, Any]]] = []
        self.classifications: List[Dict[str, Any]] = []
        self.crop_ocr_results: List[Dict[str, Any]] = []
        self.verification_results: List[Dict[str, Any]] = []

    def detect_text(self, frames: List[Frame]) -> List[Segment]:
        self.observations = []
        self.provenance = []
        self.rejected_regions = []
        self.review_regions = []
        self.region_windows = []
        self.tracks = []
        self.classifications = []
        self.crop_ocr_results = []
        self.verification_results = []
        regions = self._detect_regions(frames)
        tracks = self._track_regions(regions)
        self.tracks = tracks
        return self._process_tracks(tracks, frames)

    def debug_artifacts(self) -> Dict[str, Any]:
        """Return JSON-safe stage artifacts for the orchestrator debug directory."""
        return {
            "region_windows": list(self.region_windows),
            "tracks": [
                {
                    "track_id": track[0].get("track_id") if track else None,
                    "regions": [self._artifact(region, "tracked") for region in track],
                }
                for track in self.tracks
            ],
            "classifications": list(self.classifications),
            "crop_ocr_results": list(self.crop_ocr_results),
            "verification_results": list(self.verification_results),
            "accepted_intervals": list(self.observations),
            "review_regions": list(self.review_regions),
            "rejected_regions": list(self.rejected_regions),
        }

    @staticmethod
    def _detection_prompt() -> str:
        return (
            "For each supplied frame, detect visible text regions only. Do not read, "
            "transcribe, translate, or classify. Return only JSON: "
            '{"frames":[{"frame_id":"string","regions":[{"local_id":"string",'
            '"box_2d":[ymin,xmin,ymax,xmax]}]}]}. '
            "box_2d uses the inclusive 0..1000 scale."
        )

    @staticmethod
    def _classification_prompt() -> str:
        return (
            "Classify this cropped video text region. Return only JSON: "
            '{"text_type":"subtitle|watermark|logo|user_interface|scene_text|'
            'warning_overlay|other","evidence":"string"}. '
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
        step = self.window_size - self.window_overlap
        seen: set[tuple[int, tuple[float, ...]]] = set()
        for offset in range(0, len(frames), step):
            window = frames[offset:offset + self.window_size]
            if not window:
                continue
            response = self._send_window(window, offset)
            payload = self._parse(response.text, "stage A")
            entries = payload.get("frames") if isinstance(payload, dict) else None
            if not isinstance(entries, list):
                raise ProviderError("Gemini OCR stage A response requires a frames list.")
            self.region_windows.append(
                {
                    "window_id": f"window_{offset}",
                    "frames": [
                        {
                            "frame_id": f"frame_{index}",
                            "timestamp": frame.timestamp,
                            "image_path": frame.image_path,
                        }
                        for index, frame in enumerate(window, start=offset)
                    ],
                    "detections": entries,
                }
            )
            expected = {f"frame_{offset + index}": (offset + index, frame) for index, frame in enumerate(window)}
            for output in entries:
                if not isinstance(output, dict) or output.get("frame_id") not in expected:
                    raise ProviderError("Gemini OCR stage A returned an unknown frame_id.")
                frame_index, frame = expected[output["frame_id"]]
                detections = output.get("regions")
                if not isinstance(detections, list):
                    raise ProviderError("Gemini OCR stage A frame requires a regions list.")
                width, height = self._image_size(frame.image_path)
                for region_index, entry in enumerate(detections):
                    if not isinstance(entry, dict) or not isinstance(entry.get("local_id"), str) or not entry["local_id"]:
                        raise ProviderError("Gemini OCR stage A region requires a local_id.")
                    box = self._validate_box(entry.get("box_2d"))
                    key = (frame_index, tuple(box))
                    if key in seen:
                        continue
                    seen.add(key)
                    regions.append({"region_id": f"{output['frame_id']}:{entry['local_id']}", "local_id": entry["local_id"],
                        "frame_id": output["frame_id"], "frame": frame, "frame_index": frame_index,
                        "box_2d": box, "pixel_box": self._to_pixel_box(box, width, height), "stages": ["A"]})
        return regions

    def _track_regions(self, regions: Sequence[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        tracks: List[List[Dict[str, Any]]] = []
        for region in sorted(regions, key=lambda item: (item["frame_index"], item["region_id"])):
            candidates = [(self._track_score(region, track[-1]), index, track) for index, track in enumerate(tracks)
                          if region["frame_index"] - track[-1]["frame_index"] <= 1]
            matched = max(candidates, default=(0.0, -1, None), key=lambda item: (item[0], -item[1]))[2]
            if matched is not None and self._track_score(region, matched[-1]) < self.tracking_iou_threshold:
                matched = None
            if matched is None:
                tracks.append([region])
            else:
                matched.append(region)
        return tracks

    def _process_tracks(self, tracks: Sequence[List[Dict[str, Any]]], frames: Sequence[Frame]) -> List[Segment]:
        segments: List[Segment] = []
        for track_index, track in enumerate(tracks):
            track_id = f"track_{track_index}"
            consensus = self._median_box([item["box_2d"] for item in track])
            representative = track[len(track) // 2]
            crop = self._make_crop(representative["frame"].image_path, self._to_pixel_box(consensus, *self._image_size(representative["frame"].image_path)))
            classification = self._classify(crop, track_id)
            for item in track:
                item["track_id"] = track_id
                item["stages"].extend(["tracking", "B"])
                item["text_type"] = classification["text_type"]
                item["classification_evidence"] = classification["evidence"]
            if classification["text_type"] != "subtitle":
                self.rejected_regions.extend(self._artifact(item, "not_output_text_type") for item in track)
                continue
            readings = self._transcribe_track(track, consensus, track_id)
            readable = [text for text in readings if text is not None]
            if not readable:
                self.review_regions.extend(self._artifact(item, "ambiguous_transcription") for item in track)
                continue
            for item in track:
                item["stages"].append("C")
            unique = set(readable)
            transcription = readable[0] if len(unique) == 1 and len(readable) == len(readings) else None
            stage_d_used = False
            if transcription is None and self.verification_enabled:
                stage_d_used = True
                verified = self._transcribe(crop, track_id, "stage_d_verification")
                if verified is None or (unique and verified not in unique):
                    self.review_regions.extend(self._artifact(item, "verification_failed") for item in track)
                    continue
                transcription = verified
                for item in track:
                    item["stages"].append("D")
            if transcription is None:
                self.review_regions.extend(self._artifact(item, "verification_disabled") for item in track)
                continue
            start, end = self._interval(track, frames)
            pixel_box = self._to_pixel_box(consensus, *self._image_size(representative["frame"].image_path))
            observation = self._artifact(representative, "accepted")
            observation.update({"text_verbatim": transcription, "consensus_box_2d": consensus, "interval": [start, end]})
            self.observations.append(observation)
            segments.append(Segment(id=f"ocr_{track_id}", start=start, end=end, text=transcription, source="ocr",
                confidence=0.0, confidence_available=False,
                bbox={"left": pixel_box[0], "top": pixel_box[1], "right": pixel_box[2], "bottom": pixel_box[3]},
                metadata={"provider": self.__class__.__name__, "model": self.stage_models["C"],
                    "confidence_kind": "unknown", "text_type": "subtitle", "track_id": track_id,
                    "stage_provenance": list(representative["stages"]),
                    "validation_evidence": {"stage_a_bbox_valid": True, "stage_b_evidence": classification["evidence"],
                    "stage_c_readings": len(readings), "stage_d_verified": stage_d_used, "consensus_box_2d": consensus}}))
        return segments

    def _classify(self, crop: bytes, region_id: str) -> Dict[str, str]:
        payload = self._parse(
            self._send(self._classification_prompt(), crop, "stage_b_classification", region_id).text,
            "stage B",
        )
        text_type = payload.get("text_type") if isinstance(payload, dict) else None
        evidence = payload.get("evidence") if isinstance(payload, dict) else None
        if text_type not in _TEXT_TYPES or not isinstance(evidence, str) or not evidence.strip():
            raise ProviderError("Gemini OCR stage B response has an invalid text_type.")
        self.classifications.append(
            {
                "track_id": region_id,
                "text_type": text_type,
                "evidence": evidence,
            }
        )
        return {"text_type": text_type, "evidence": evidence}

    def _transcribe(self, crop: bytes, region_id: str, source_kind: str) -> Optional[str]:
        prompt = self._verification_prompt() if source_kind == "stage_d_verification" else self._transcription_prompt()
        payload = self._parse(self._send(prompt, crop, source_kind, region_id).text, source_kind)
        result = {
            "region_id": region_id,
            "transcription_status": (
                payload.get("transcription_status") if isinstance(payload, dict) else None
            ),
            "text_verbatim": payload.get("text_verbatim") if isinstance(payload, dict) else None,
        }
        if source_kind == "stage_d_verification":
            self.verification_results.append(result)
        else:
            self.crop_ocr_results.append(result)
        if not isinstance(payload, dict) or payload.get("transcription_status") != "readable":
            return None
        text = payload.get("text_verbatim")
        if not isinstance(text, str) or not text.strip():
            return None
        return text

    def _send_window(self, frames: Sequence[Frame], offset: int):
        parts: List[Dict[str, Any]] = [{"text": self._detection_prompt()}]
        blobs: List[bytes] = []
        for index, frame in enumerate(frames, start=offset):
            blob = self._read_bytes(frame.image_path)
            blobs.append(blob)
            parts.extend([
                {"text": f"frame_id=frame_{index}; timestamp={frame.timestamp}"},
                {"inline_data": {"mime_type": "image/png", "data": base64.b64encode(blob).decode("ascii")}},
            ])
        return self._send_with_contents(
            self._detection_prompt(), b"".join(blobs), "stage_a_region_detection",
            f"window_{offset}", contents=[{"parts": parts}],
        )

    def _send(self, prompt: str, image_bytes: bytes, source_kind: str, region_id: str):
        return self._send_with_contents(
            prompt, image_bytes, source_kind, region_id,
            [{"parts": [{"text": f"{prompt} source_id={region_id}"},
                         {"inline_data": {"mime_type": "image/png", "data": base64.b64encode(image_bytes).decode("ascii")}}]}],
        )

    def _send_with_contents(
        self, prompt: str, image_bytes: bytes, source_kind: str, region_id: str,
        contents: List[Dict[str, Any]],
    ):
        stage = {"stage_a_region_detection": "A", "stage_b_classification": "B",
                 "stage_c_crop_transcription": "C", "stage_d_verification": "D"}[source_kind]
        model = self.stage_models[stage]
        request = GeminiRequest(
            contents=contents,
            model=model,
            generation_config=self.stage_options[stage],
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
                model=model, role=MODE_ROLE_PRIMARY, api_key=self.api_key
            ).generate_content(request)
        except GeminiError as exc:
            raise ProviderError(f"Gemini OCR {source_kind} request failed: {exc}") from exc
        self.provenance.append({
            "stage": source_kind,
            "region_id": region_id,
            "model": response.model or model,
            "model_stage": response.model_stage,
            "api_surface": self.api_surface,
            "prompt_version": self.prompt_version,
            "schema_version": self.schema_version,
            "request_id": response.request_id,
            "cache_hit": response.cache_hit,
        })
        return response

    def _transcribe_track(
        self, track: Sequence[Dict[str, Any]], consensus: Sequence[float], track_id: str
    ) -> List[Optional[str]]:
        selected = [track[0], track[len(track) // 2], track[-1]]
        unique = []
        for item in selected:
            if item not in unique:
                unique.append(item)
        readings = []
        for index, item in enumerate(unique):
            pixel_box = self._to_pixel_box(consensus, *self._image_size(item["frame"].image_path))
            crop = self._make_crop(item["frame"].image_path, pixel_box, variant=index % 2)
            readings.append(self._transcribe(crop, f"{track_id}:crop_{index}", "stage_c_crop_transcription"))
        return readings

    @staticmethod
    def _median_box(boxes: Sequence[Sequence[float]]) -> List[float]:
        return [float(statistics.median(values)) for values in zip(*boxes)]

    def _interval(self, track: Sequence[Dict[str, Any]], frames: Sequence[Frame]) -> tuple[float, float]:
        start = track[0]["frame"].timestamp
        last_index = track[-1]["frame_index"]
        if last_index + 1 < len(frames):
            end = frames[last_index + 1].timestamp
        else:
            deltas = [right.timestamp - left.timestamp for left, right in zip(frames, frames[1:]) if right.timestamp > left.timestamp]
            end = track[-1]["frame"].timestamp + (statistics.median(deltas) if deltas else 1.0)
        if end <= start:
            raise ProviderError("Gemini OCR accepted interval must have end greater than start.")
        return start, end

    def _track_score(self, current: Dict[str, Any], prior: Dict[str, Any]) -> float:
        box, other = current["box_2d"], prior["box_2d"]
        iou = self._iou(box, other)
        center = lambda value: ((value[1] + value[3]) / 2, (value[0] + value[2]) / 2)
        width = lambda value: value[3] - value[1]
        height = lambda value: value[2] - value[0]
        cx, cy = center(box)
        px, py = center(other)
        center_score = max(0.0, 1 - (((cx - px) ** 2 + (cy - py) ** 2) ** .5 / 1000))
        band_score = 1.0 if abs(cy - py) <= max(height(box), height(other)) else 0.0
        size_score = min(width(box) * height(box), width(other) * height(other)) / max(width(box) * height(box), width(other) * height(other))
        orientation_score = min(width(box) / height(box), width(other) / height(other)) / max(width(box) / height(box), width(other) / height(other))
        return .45 * iou + .2 * center_score + .15 * band_score + .1 * size_score + .1 * orientation_score

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

    def _make_crop(self, image_path: str, pixel_box: Sequence[float], variant: int = 0) -> bytes:
        left, top, right, bottom = pixel_box
        try:
            with Image.open(image_path) as image:
                pad_x = (right - left) * self.crop_padding_ratio
                pad_y = (bottom - top) * self.crop_padding_ratio
                crop = image.crop((max(0, left - pad_x), max(0, top - pad_y), min(image.width, right + pad_x), min(image.height, bottom + pad_y)))
                if variant:
                    crop = crop.convert("L")
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
