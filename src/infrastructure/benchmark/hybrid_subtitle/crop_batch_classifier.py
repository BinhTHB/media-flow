"""Bounded Gemini classifier for V4 text tracks."""
from __future__ import annotations

import base64
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import cv2

from .crop_extractor import TrackCropExtractor

logger = logging.getLogger(__name__)
MAX_CLASSIFICATION_REQUESTS = 3
MAX_TRACKS_PER_TABLE = 9


class CropBatchClassifierV21:
    """Classify V4 text tracks in at most three image tables.

    Gemini receives context and crop images plus track IDs. It returns only a
    verdict, category, and confidence for each track ID. Geometry remains local.
    """

    def __init__(
        self,
        gemini_client: Any,
        model: str = "gemini-3.5-flash-lite",
        budget_limit: int = MAX_CLASSIFICATION_REQUESTS,
        artifact_dir: str | Path | None = None,
        job_id: str = "",
        run_id: str = "",
    ) -> None:
        self.client = gemini_client
        self.model = model
        self.budget_limit = min(int(budget_limit), MAX_CLASSIFICATION_REQUESTS)
        self.artifact_dir = Path(artifact_dir) if artifact_dir else None
        self.job_id = job_id
        self.run_id = run_id
        self.requests_sent = 0
        self.retry_count = 0
        self.total_candidates = 0

    def set_run_context(self, job_id: str, run_id: str = "") -> None:
        self.job_id = job_id
        self.run_id = run_id

    def classify_batch(
        self,
        tracks: List[Dict[str, Any]],
        proposal_bundle: Any,
        batch_size: int = MAX_TRACKS_PER_TABLE,
        canary_tracks: List[Dict[str, Any]] | None = None,
    ) -> Dict[str, Dict[str, Any]]:
        """Return one Gemini verdict per candidate track, fail-closed on errors.

        ``canary_tracks`` is deliberately ignored. Canary validation belongs in
        ``evaluation/`` and never participates in a production video run.
        """
        del canary_tracks
        if not tracks:
            return {}
        if self.client is None:
            logger.error("LOUD-WARN: Gemini classifier unavailable, all tracks rejected.")
            return {}
        if len(tracks) > MAX_CLASSIFICATION_REQUESTS * MAX_TRACKS_PER_TABLE:
            raise RuntimeError("Classification candidate count exceeds the three-table budget.")

        self.requests_sent = 0
        self.retry_count = 0
        self.total_candidates = len(tracks)
        results: Dict[str, Dict[str, Any]] = {}
        for index in range(0, len(tracks), MAX_TRACKS_PER_TABLE):
            table = tracks[index:index + MAX_TRACKS_PER_TABLE]
            table_number = index // MAX_TRACKS_PER_TABLE + 1
            results.update(self._classify_table(table, proposal_bundle, table_number))
        return results

    def _classify_table(
        self, tracks: List[Dict[str, Any]], proposal_bundle: Any, table_number: int
    ) -> Dict[str, Dict[str, Any]]:
        if self.requests_sent >= self.budget_limit:
            raise RuntimeError("Classification request budget exceeded.")
        parts = self._build_v4_prompt_parts(tracks, proposal_bundle)
        request_id = f"track_table_{table_number:02d}"
        response = self._request(parts, request_id)
        self.requests_sent += 1
        response_ref = self._write_response(table_number, response.text)
        try:
            from src.infrastructure.gemini.structured_output import parse_json
            payload = parse_json(response.text)
        except Exception as exc:
            logger.warning("LOUD-WARN: malformed Gemini response for %s: %s", request_id, exc)
            return self._retry_once(tracks, proposal_bundle, table_number, request_id)
        if not isinstance(payload, list):
            logger.warning("LOUD-WARN: Gemini response for %s is not a JSON list", request_id)
            return self._retry_once(tracks, proposal_bundle, table_number, request_id)
        decisions = {
            item.get("track_id"): item
            for item in payload
            if isinstance(item, dict) and isinstance(item.get("track_id"), str)
            and isinstance(item.get("is_subtitle"), bool)
        }
        result: Dict[str, Dict[str, Any]] = {}
        for track in tracks:
            track_id = track["track_id"]
            decision = decisions.get(track_id)
            if decision is None:
                logger.warning("LOUD-WARN: missing Gemini verdict for track_id=%s", track_id)
                continue
            result[track_id] = {
                "is_subtitle": decision["is_subtitle"],
                "category": str(decision.get("category", "other")),
                "confidence": decision.get("confidence"),
                "model": self.model,
                "response_ref": response_ref,
                "cache_hit": bool(getattr(response, "cache_hit", False)),
            }
        return result

    def _retry_once(
        self, tracks: List[Dict[str, Any]], proposal_bundle: Any, table_number: int, request_id: str
    ) -> Dict[str, Dict[str, Any]]:
        if self.retry_count >= 1 or self.requests_sent >= self.budget_limit:
            logger.error("LOUD-WARN: malformed Gemini response remains unresolved for %s", request_id)
            return {}
        self.retry_count += 1
        parts = self._build_v4_prompt_parts(tracks, proposal_bundle)
        response = self._request(parts, f"{request_id}_retry")
        self.requests_sent += 1
        response_ref = self._write_response(table_number, response.text, retry=True)
        try:
            from src.infrastructure.gemini.structured_output import parse_json
            payload = parse_json(response.text)
        except Exception as exc:
            logger.error("LOUD-WARN: retry response is malformed: %s", exc)
            return {}
        if not isinstance(payload, list):
            return {}
        values = {item.get("track_id"): item for item in payload if isinstance(item, dict)}
        return {
            track["track_id"]: {
                "is_subtitle": values[track["track_id"]]["is_subtitle"],
                "category": str(values[track["track_id"]].get("category", "other")),
                "confidence": values[track["track_id"]].get("confidence"),
                "model": self.model,
                "response_ref": response_ref,
                "cache_hit": bool(getattr(response, "cache_hit", False)),
            }
            for track in tracks
            if isinstance(values.get(track["track_id"]), dict)
            and isinstance(values[track["track_id"]].get("is_subtitle"), bool)
        }

    def _request(self, parts: List[Dict[str, Any]], request_id: str) -> Any:
        from src.infrastructure.gemini.contracts import GeminiRequest
        return self.client.generate_content(
            GeminiRequest(
                contents=[{"parts": parts}],
                model=self.model,
                generation_config={"temperature": 0.1},
                system_instruction=(
                    "Classify only whether each given text track is a dialogue subtitle. "
                    "Default false. Do not return coordinates or masks."
                ),
                request_id=request_id,
                source_kind="track_classifier_v4",
                api_surface="generate_content",
            ),
            use_cache=False,
        )

    def _build_v4_prompt_parts(
        self, tracks: List[Dict[str, Any]], proposal_bundle: Any
    ) -> List[Dict[str, Any]]:
        extractor = TrackCropExtractor()
        lines = [
            "Classify each track ID as a dialogue subtitle or not.",
            "The coordinates are context only. Do not return coordinates.",
            "Return raw JSON array: [{\"track_id\":\"T001\",\"is_subtitle\":true,\"category\":\"subtitle\",\"confidence\":0.0}].",
        ]
        images: List[Dict[str, Any]] = []
        image_number = 1
        for track in tracks:
            evidence = extractor.extract_evidence(track, proposal_bundle)
            bbox = track.get("temporal_union_bbox_xyxy") or track.get("median_bbox_xyxy") or []
            context = (evidence.get("clean_frames") or [None])[0]
            crop_images = evidence.get("crops", [])[:3]
            indexes: List[int] = []
            for image in [context, *crop_images]:
                if image is None:
                    continue
                _, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 85])
                images.append({"inline_data": {"mime_type": "image/jpeg", "data": base64.b64encode(encoded.tobytes()).decode("ascii")}})
                indexes.append(image_number)
                image_number += 1
            lines.append(f"track_id={track['track_id']}; bbox={bbox}; images={indexes}")
        return [{"text": "\n".join(lines)}, *images]

    def _write_response(self, table_number: int, text: str, retry: bool = False) -> str:
        suffix = "_retry" if retry else ""
        relative = f"responses/request_{table_number:02d}{suffix}.json"
        if self.artifact_dir is None:
            return relative
        destination = self.artifact_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(text, encoding="utf-8")
        return relative


CropBatchClassifier = CropBatchClassifierV21
