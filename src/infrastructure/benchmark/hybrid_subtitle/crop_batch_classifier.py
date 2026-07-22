"""Crop batch classifier with canary checks, auto-retry, and budget ceiling.
"""
from __future__ import annotations
import base64
import cv2
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List
from .crop_extractor import TrackCropExtractor

logger = logging.getLogger(__name__)

class CropBatchClassifier:
    """Multimodal batch classifier for UNCERTAIN subtitle tracks.

    Optimizes Gemini tokens by grouping 10-20 crop-items per request,
    validates model truthfulness using injected negative canaries,
    and enforces a hard request budget ceiling to prevent runaway costs.
    """

    def __init__(
        self,
        gemini_client: Any,
        model: str = "gemini-3.5-flash-lite",
        budget_limit: int = 60,
    ) -> None:
        self.client = gemini_client
        self.model = model
        self.budget_limit = budget_limit
        self.requests_sent = 0
    def classify_batch(
        self,
        tracks: List[Dict[str, Any]],
        proposal_bundle: Any,
        batch_size: int = 5,
        canary_tracks: List[Dict[str, Any]] | None = None,
    ) -> Dict[str, Any]:
        """Classify tracks in batches. Returns dict of track_id -> decision."""
        if not tracks or self.client is None:
            return {}

        results = {}
        for i in range(0, len(tracks), batch_size):
            batch = tracks[i : i + batch_size]
            res = self._classify_one_batch(
                batch, proposal_bundle, canary_tracks or []
            )
            results.update(res)
        return results

    def _classify_one_batch(
        self,
        batch: List[Dict[str, Any]],
        proposal_bundle: Any,
        canary_tracks: List[Dict[str, Any]],
        is_retry: bool = False,
    ) -> Dict[str, Any]:
        if self.requests_sent >= self.budget_limit:
            logger.error(
                "LOUD-WARN: Budget limit reached (%d requests). BLOCKED.",
                self.budget_limit,
            )
            raise RuntimeError(
                f"Budget limit reached ({self.budget_limit} requests)."
            )

        extractor = TrackCropExtractor()
        evidence_list = []
        for t in batch:
            ev = extractor.extract_evidence(t, proposal_bundle)
            if ev:
                ev["track_id"] = t["track_id"]
                evidence_list.append(ev)

        if not evidence_list:
            return {}

        injected_canaries = []
        if not canary_tracks:
            raise RuntimeError("No local negative canary supplied. BLOCKED.")
        canary_track = canary_tracks[self.requests_sent % len(canary_tracks)]
        canary_evidence = extractor.extract_evidence(canary_track, proposal_bundle)
        if not canary_evidence:
            raise RuntimeError("Canary evidence is unavailable. BLOCKED.")
        canary_id = f"canary_{canary_track['track_id']}"
        canary_evidence["track_id"] = canary_id
        evidence_list.append(canary_evidence)
        injected_canaries.append(canary_id)

        prompt_parts = []
        prompt_text = [
            "You are a vision-language classifier for on-screen text in video crops.",
            "Classify whether each candidate track is a subtitle.",
            "Subtitle definition: text representing spoken dialogue, narration, or captions.",
            "Non-subtitle: logos, UI text, watermarks, scene text, decorative text.",
            "",
            "Analyze each track using its trajectory metadata and the sequence of crops provided.",
            "Crops are clean unannotated images showing the text region over time.",
            "",
            "Candidate tracks to classify:",
        ]

        images_in_request = []
        img_counter = 1

        for ev in evidence_list:
            tid = ev["track_id"]
            traj = ev["trajectory"]
            start_img = img_counter
            for crop in ev["crops"]:
                _, png = cv2.imencode(".png", crop)
                images_in_request.append(
                    {
                        "inline_data": {
                            "mime_type": "image/png",
                            "data": base64.b64encode(png.tobytes()).decode("ascii"),
                        }
                    }
                )
                img_counter += 1
            end_img = img_counter - 1
            prompt_text.append(
                f"- Track {tid}: bottom_y_ratio={traj['bottom_y_ratio']:.3f}, "
                f"duration={traj['duration']:.2f}s, displacement={traj['displacement']:.1f}px. "
                f"Visual crops are shown in Image {start_img} to Image {end_img}."
            )

        prompt_text.extend(
            [
                "",
                "Return a JSON array of decisions matching this schema:",
                "[",
                "  {",
                "    \"track_id\": \"<track_id>\",",
                "    \"is_subtitle\": true/false,",
                "    \"category\": \"dialogue_subtitle\" or other allowed category,",
                "    \"confidence\": 0.0-1.0,",
                "    \"transcription\": \"extracted text\",",
                "    \"reason\": \"short explanation\"",
                "  }",
                "]",
                "Do NOT include markdown wrapping or other text. Return raw JSON array.",
            ]
        )

        prompt_parts.append({"text": "\n".join(prompt_text)})
        prompt_parts.extend(images_in_request)

        try:
            from src.infrastructure.gemini.contracts import GeminiRequest

            req = GeminiRequest(
                contents=[{"parts": prompt_parts}],
                model=self.model,
                generation_config={"temperature": 0.1},
                system_instruction="Classify video text tracks as subtitle or non-subtitle.",
                request_id=f"crop_batch_{self.requests_sent}",
                source_kind="crop_classifier",
                api_surface="benchmark",
            )
            self.requests_sent += 1
            resp = self.client.generate_content(req, use_cache=True)
            text = resp.text if hasattr(resp, "text") else ""

            from src.infrastructure.gemini.structured_output import parse_json
            data = parse_json(text)
            if not isinstance(data, list):
                raise ValueError("Response is not a JSON list")

            expected_ids = {
                evidence["track_id"] for evidence in evidence_list
            }
            actual_ids = [
                decision.get("track_id") for decision in data
                if isinstance(decision, dict)
            ]
            if (
                len(actual_ids) != len(set(actual_ids))
                or set(actual_ids) != expected_ids
            ):
                raise ValueError(
                    "Batch response track IDs are missing, extra, or duplicated."
                )
            decisions = {d["track_id"]: d for d in data}

            canary_passed = True
            for cid in injected_canaries:
                decision = decisions.get(cid)
                if decision and decision.get("is_subtitle") is True:
                    logger.warning(
                        "LOUD-WARN: Canary %s failed (flagged as subtitle)!", cid
                    )
                    canary_passed = False

            if not canary_passed:
                if is_retry:
                    logger.error(
                        "LOUD-WARN: Canary failed on retry batch. BLOCKED."
                    )
                    raise RuntimeError("Canary validation failed twice. BLOCKED.")

                logger.warning(
                    "LOUD-WARN: Canary check failed. Retrying batch with half size."
                )
                half = max(1, len(batch) // 2)
                res1 = self._classify_one_batch(
                    batch[:half], proposal_bundle, canary_tracks, is_retry=True
                )
                res2 = self._classify_one_batch(
                    batch[half:], proposal_bundle, canary_tracks, is_retry=True
                ) if len(batch) > half else {}
                res1.update(res2)
                return res1

            self._log_ledger(resp, len(batch), canary_passed)

            mapped_decisions = {}
            for t in batch:
                tid = t["track_id"]
                decision = decisions.get(tid)
                if decision:
                    mapped_decisions[tid] = decision
                else:
                    logger.warning(
                        "LOUD-WARN: Missing decision for track %s. Fail-closed.",
                        tid,
                    )
            return mapped_decisions

        except Exception as exc:
            logger.error("LOUD-WARN: Crop batch classification failed: %s", exc)
            if "BLOCKED" in str(exc) or "validation failed" in str(exc):
                raise exc
            return {}

    def _log_ledger(self, resp: Any, batch_size: int, canary_passed: bool) -> None:
        try:
            ledger_dir = Path("artifacts/round23")
            ledger_dir.mkdir(parents=True, exist_ok=True)
            ledger_path = ledger_dir / "crop_classifier_ledger.jsonl"

            log_entry = {
                "timestamp": time.time(),
                "live_or_cache": "cache"
                if getattr(resp, "cache_hit", False)
                else "live",
                "batch_size": batch_size,
                "canary_pass": canary_passed,
                "model": self.model,
                "finish_reason": getattr(resp, "finish_reason", "STOP"),
            }
            with open(ledger_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry) + "\n")
        except Exception as exc:
            logger.warning(
                "LOUD-WARN: Failed to write to crop classifier ledger: %s", exc
            )
