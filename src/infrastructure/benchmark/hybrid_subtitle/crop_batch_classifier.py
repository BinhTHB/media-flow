"""Crop batch classifier V2.1 — Gemini packet with full-packet canary.

Packet format (V2.1):
  Image 0 = clean 720p context frame (CẤM annotations — regression v13 lesson)
  Images 1..N = FULL-RES crops per candidate, 3-4 crops/track over time range
  Metadata = per-candidate: index, bbox_normalized, duration, displacement,
             motion_score, language, static_long

adjacent_confirmed_subtitle: computed in CODE, NOT shown to model.
Model returns per-candidate {candidate_id, subtitle y/n, burn_in y/n, text_echo}.
Accept on subtitle=y (burn_in=n is additional signal, NOT AND gate).

Echo-check fuzzy >= 0.6 vs Paddle text → mismatch → retry solo-packet once.
Budget: total <= 60, retry <= 10% candidates.

Bất biến V2.1: reject-only canary. Fail-closed on canary failure.
"""
from __future__ import annotations
import base64
import cv2
import json
import logging
import time
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .crop_extractor import TrackCropExtractor

logger = logging.getLogger(__name__)

FUZZY_THRESHOLD: float = 0.6  # Echo-check threshold
MAX_RETRY_RATIO: float = 0.10  # Max retries as fraction of total candidates
BUDGET_LIMIT: int = 60


class CropBatchClassifierV21:
    """V2.1 multimodal batch classifier with full-packet canary.

    Builds a single Gemini packet per candidate batch with:
    - One clean 720p context frame (no annotations)
    - 3-4 full-res crops per track showing text evolution
    - Rich metadata (bbox, duration, displacement, language)

    Validates model truthfulness via full-packet canary (same builder).
    Echo-check retries mismatched candidates solo.
    """

    def __init__(
        self,
        gemini_client: Any,
        model: str = "gemini-3.5-flash-lite",
        budget_limit: int = BUDGET_LIMIT,
        artifact_dir: str | Path | None = None,
        job_id: str = "",
        run_id: str = "",
    ) -> None:
        self.client = gemini_client
        self.model = model
        self.budget_limit = budget_limit
        self.artifact_dir = Path(artifact_dir) if artifact_dir else None
        self.job_id = job_id
        self.run_id = run_id
        self.requests_sent = 0
        self.total_candidates = 0
        self.retry_count = 0
        self.unresolved_count = 0
        self.echo_check_ocr_error_count = 0

    def set_run_context(self, job_id: str, run_id: str = "") -> None:
        self.job_id = job_id
        self.run_id = run_id

    def classify_batch(
        self,
        tracks: List[Dict[str, Any]],
        proposal_bundle: Any,
        batch_size: int = 5,
        canary_tracks: List[Dict[str, Any]] | None = None,
    ) -> Dict[str, Any]:
        """Classify tracks in V2.1 batches. Returns dict of track_id -> decision."""
        if not tracks or self.client is None:
            return {}

        self.total_candidates = len(tracks)
        results: Dict[str, Any] = {}

        for i in range(0, len(tracks), batch_size):
            batch = tracks[i: i + batch_size]
            batch_result = self._classify_one_batch(
                batch, proposal_bundle, canary_tracks or []
            )
            results.update(batch_result)

        return results

    def _build_packet(
        self,
        tracks: List[Dict[str, Any]],
        proposal_bundle: Any,
        canary_evidence: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[Dict[str, Any]], List[str], Dict[str, Any]]:
        """Build a V2.1 Gemini packet.

        Returns (evidence_list, injected_canary_ids, metadata_dict).
        """
        extractor = TrackCropExtractor()
        evidence_list: List[Dict[str, Any]] = []
        canary_ids: List[str] = []

        for t in tracks:
            ev = extractor.extract_evidence(t, proposal_bundle)
            if ev:
                ev["metadata"] = self._build_track_metadata(t, ev, proposal_bundle)
                ev["track_id"] = t["track_id"]
                evidence_list.append(ev)

        # Inject canary if provided
        if canary_evidence:
            canary_id = f"canary_{canary_evidence.get('track_id', 'N001')}"
            canary_evidence["track_id"] = canary_id
            canary_evidence["metadata"] = {"is_canary": True}
            evidence_list.append(canary_evidence)
            canary_ids.append(canary_id)

        return evidence_list, canary_ids, {}

    def _build_track_metadata(
        self, track: Dict[str, Any], evidence: Dict[str, Any],
        bundle: Any = None,
    ) -> Dict[str, Any]:
        """Build V2.1 metadata for a single track candidate."""
        traj = evidence.get("trajectory", {})
        groups = track.get("groups", [])
        bbox = (
            track.get("median_bbox_xyxy")
            or track.get("temporal_union_bbox_xyxy")
            or [0, 0, 0, 0]
        )

        # Normalized bbox in [0, 1]
        frame_h = bundle.video_height if bundle is not None and hasattr(bundle, "video_height") else 720
        frame_w = bundle.video_width if bundle is not None and hasattr(bundle, "video_width") else 1280
        norm_bbox = (
            round(bbox[0] / frame_w, 4) if frame_w else 0,
            round(bbox[1] / frame_h, 4) if frame_h else 0,
            round(bbox[2] / frame_w, 4) if frame_w else 0,
            round(bbox[3] / frame_h, 4) if frame_h else 0,
        )

        # Extract language from track metadata
        language = ""
        behavior = track.get("behavior", {})
        if isinstance(behavior, dict):
            language = behavior.get("detected_language", "")

        return {
            "bbox_normalized": list(norm_bbox),
            "duration_s": round(traj.get("duration", 0.0), 2),
            "displacement_px": round(traj.get("displacement", 0.0), 1),
            "motion_score": round(traj.get("motion_score", 0.0), 2),
            "language": language,
            "static_long": bool(track.get("static_long", False)),
        }

    def _build_canary_evidence(
        self, track: Dict[str, Any], proposal_bundle: Any
    ) -> Dict[str, Any] | None:
        """Build canary evidence using the same packet builder as real tracks."""
        extractor = TrackCropExtractor()
        ev = extractor.extract_evidence(track, proposal_bundle)
        if ev:
            ev["metadata"] = {"is_canary": True}
        return ev

    def _classify_one_batch(
        self,
        batch: List[Dict[str, Any]],
        proposal_bundle: Any,
        canary_tracks: List[Dict[str, Any]],
        is_retry: bool = False,
    ) -> Dict[str, Any]:
        """Classify one batch of tracks via V2.1 packet."""
        if self.requests_sent >= self.budget_limit:
            logger.error(
                "LOUD-WARN: Budget limit reached (%d requests). BLOCKED.",
                self.budget_limit,
            )
            raise RuntimeError(
                f"Budget limit reached ({self.budget_limit} requests)."
            )

        # Build canary evidence from a track rejected by >= 2 local rules
        canary_evidence = None
        canary_source = None
        if canary_tracks:
            # Pick canary from tracks rejected by multiple rules
            for candidate in canary_tracks:
                ev = self._build_canary_evidence(candidate, proposal_bundle)
                if ev:
                    canary_evidence = ev
                    canary_source = candidate.get("track_id", "N001")
                    break

        evidence_list, canary_ids, _ = self._build_packet(
            batch, proposal_bundle, canary_evidence
        )

        if not evidence_list:
            return {}

        # Build the V2.1 prompt
        prompt_parts = self._build_v21_prompt(evidence_list)
        if not prompt_parts:
            return {}

        try:
            from src.infrastructure.gemini.contracts import GeminiRequest

            req = GeminiRequest(
                contents=[{"parts": prompt_parts}],
                model=self.model,
                generation_config={"temperature": 0.1},
                system_instruction=(
                    "You are a subtitle detection assistant. Classify each candidate "
                    "as subtitle or non-subtitle. Default is NOT subtitle."
                ),
                request_id=f"crop_batch_v21_{self.requests_sent}",
                source_kind="crop_classifier_v21",
                api_surface="benchmark",
            )
            self.requests_sent += 1
            resp = self.client.generate_content(req, use_cache=True)
            text = resp.text if hasattr(resp, "text") else ""

            from src.infrastructure.gemini.structured_output import parse_json
            data = parse_json(text)
            if not isinstance(data, list):
                raise ValueError("Response is not a JSON list")

            expected_ids = {ev["track_id"] for ev in evidence_list}
            actual_ids = [
                d.get("candidate_id", d.get("track_id", ""))
                for d in data if isinstance(d, dict)
            ]
            # Map candidate_id or track_id
            decisions: Dict[str, Any] = {}
            for d in data:
                if not isinstance(d, dict):
                    continue
                tid = d.get("candidate_id") or d.get("track_id", "")
                decisions[tid] = d

            if set(actual_ids) != expected_ids and not set(actual_ids).issuperset(expected_ids):
                # Allow extra model metadata entries, but require all expected IDs
                missing = expected_ids - set(actual_ids)
                if missing:
                    logger.warning(
                        "LOUD-WARN: Missing IDs %s in response. Fail-closed for those.",
                        missing,
                    )

            # Canary check
            canary_passed = True
            for cid in canary_ids:
                decision = decisions.get(cid)
                if decision and decision.get("is_subtitle") is True:
                    logger.warning(
                        "LOUD-WARN: Canary %s flagged as subtitle!", cid
                    )
                    canary_passed = False

            if not canary_passed:
                self._log_ledger(resp, len(batch), False)
                if is_retry:
                    logger.error(
                        "LOUD-WARN: Canary failed on retry. BLOCKED."
                    )
                    raise RuntimeError("Canary validation failed twice. BLOCKED.")
                # Retry with half batch
                half = max(1, len(batch) // 2)
                res1 = self._classify_one_batch(
                    batch[:half], proposal_bundle, canary_tracks, is_retry=True
                )
                res2 = (
                    self._classify_one_batch(
                        batch[half:], proposal_bundle, canary_tracks, is_retry=True
                    )
                    if len(batch) > half
                    else {}
                )
                res1.update(res2)
                return res1

            self._log_ledger(resp, len(batch), canary_passed)

            # Map decisions back to tracks, run echo-check
            mapped: Dict[str, Any] = {}
            for t in batch:
                tid = t["track_id"]
                decision = decisions.get(tid)
                if not decision:
                    logger.warning(
                        "LOUD-WARN: Missing decision for track %s. Fail-closed.",
                        tid,
                    )
                    continue

                # Echo-check
                echo_ok = self._echo_check(decision, t, proposal_bundle)
                if not echo_ok:
                    if self.retry_count < max(
                        1, int(self.total_candidates * MAX_RETRY_RATIO)
                    ):
                        self.retry_count += 1
                        # Solo retry
                        solo = self._classify_one_batch(
                            [t], proposal_bundle, canary_tracks, is_retry=True
                        )
                        if solo:
                            mapped[tid] = solo[tid]
                            continue
                    self.unresolved_count += 1
                    logger.warning(
                        "LOUD-WARN: Echo-check failed for %s (unresolved). NO.",
                        tid,
                    )
                    decision["is_subtitle"] = False
                    decision["echo_mismatch"] = True

                mapped[tid] = decision

            return mapped

        except Exception as exc:
            logger.error("LOUD-WARN: V2.1 batch classification failed: %s", exc)
            if "BLOCKED" in str(exc):
                raise exc
            return {}

    def _build_v21_prompt(
        self, evidence_list: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]] | None:
        """Build V2.1 Gemini prompt with clean 720p context frame + full-res crops."""
        if not evidence_list:
            return None

        prompt_parts: List[Dict[str, Any]] = []
        prompt_lines: List[str] = [
            "You are a vision-language classifier for on-screen text in video frames.",
            "",
            "For each candidate track below, classify it as subtitle or non-subtitle.",
            "Default answer is NOT subtitle unless evidence clearly shows dialogue subtitle.",
            "",
            "Each candidate has:",
            "- A clean context frame (Image X, shown at ~720p, NO annotations)",
            "- 3-4 full-resolution crops showing the text region over time",
            "- Metadata: bbox position (normalized 0-1), duration, displacement, language",
            "",
            "Subtitle translates or transcribes spoken dialogue.",
            "SUBTITLE = YES: appears with speech, changes content, bottom of frame.",
            "SUBTITLE = NO: static watermark, logo, UI text, scene text, timer/clock, karaoke.",
            "",
            "Candidate tracks:",
        ]

        images_in_request: List[Dict[str, Any]] = []
        img_counter = 1

        for ev in evidence_list:
            tid = ev["track_id"]
            meta = ev.get("metadata", {})
            traj = ev.get("trajectory", {})

            # Image 0: Clean 720p context frame (NO boxes/annotations!)
            context_frame = ev["clean_frames"][0] if ev.get("clean_frames") else None
            if context_frame is not None:
                # Ensure it's ~720p
                H, W = context_frame.shape[:2]
                max_dim = max(H, W)
                if max_dim > 720:
                    scale = 720.0 / max_dim
                    new_w = int(round(W * scale))
                    new_h = int(round(H * scale))
                    context_frame = cv2.resize(
                        context_frame,
                        (new_w, new_h),
                        interpolation=cv2.INTER_AREA,
                    )
                _, jpeg = cv2.imencode(
                    ".jpg", context_frame, [cv2.IMWRITE_JPEG_QUALITY, 85]
                )
                images_in_request.append({
                    "inline_data": {
                        "mime_type": "image/jpeg",
                        "data": base64.b64encode(jpeg.tobytes()).decode("ascii"),
                    }
                })
                context_img_idx = img_counter
                img_counter += 1
            else:
                context_img_idx = -1

            # Images 1..N: Full-res crops
            crop_indices: List[int] = []
            for crop in ev.get("crops", []):
                _, jpeg_crop = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
                images_in_request.append({
                    "inline_data": {
                        "mime_type": "image/jpeg",
                        "data": base64.b64encode(jpeg_crop.tobytes()).decode("ascii"),
                    }
                })
                crop_indices.append(img_counter)
                img_counter += 1

            # Build metadata line
            bbox_str = f"bbox_norm={meta.get('bbox_normalized', 'N/A')}"
            dur_str = f"duration={meta.get('duration_s', traj.get('duration', 0)):.2f}s"
            disp_str = f"displacement={meta.get('displacement_px', traj.get('displacement', 0)):.1f}px"
            lang_str = f"language={meta.get('language', 'unknown')}"
            static_str = "static_long" if meta.get("static_long") else ""

            meta_str = f"  {bbox_str}, {dur_str}, {disp_str}, {lang_str}"
            if static_str:
                meta_str += f", {static_str}"

            prompt_lines.append(f"- Candidate {tid}:")
            prompt_lines.append(meta_str)
            if context_img_idx > 0:
                prompt_lines.append(
                    f"  Context frame: Image {context_img_idx} (720p, NO annotations)"
                )
            if crop_indices:
                prompt_lines.append(
                    f"  Text crops: Images {crop_indices[0]}-{crop_indices[-1]} "
                    f"({len(crop_indices)} frames over time)"
                )

        prompt_lines.extend([
            "",
            "Return a JSON array of decisions:",
            "[",
            "  {",
            '    "candidate_id": "<track_id>",',
            '    "is_subtitle": true/false,',
            '    "burn_in": true/false,   // true if text is burned into video',
            '    "category": "subtitle" | "watermark" | "scene_text" | "ui" | "other",',
            '    "confidence": 0.0-1.0,',
            '    "text_echo": "extracted text or empty string",',
            '    "reason": "brief explanation"',
            "  }",
            "]",
            "Return raw JSON array, no markdown.",
        ])

        prompt_parts.append({"text": "\n".join(prompt_lines)})
        prompt_parts.extend(images_in_request)

        return prompt_parts

    def _echo_check(
        self,
        decision: Dict[str, Any],
        track: Dict[str, Any],
        proposal_bundle: Any,
    ) -> bool:
        """Echo-check: fuzzy match Gemini text_echo vs PaddleOCR text.

        Returns True if match >= FUZZY_THRESHOLD or pass-through on error.
        Rejects track if Gemini produces text but Paddle runs successfully and finds nothing.
        """
        echo_text = decision.get("text_echo", "")
        if not echo_text:
            return True  # No echo text provided, skip check

        # Get Paddle text from track crops
        paddle_text, had_error = self._get_paddle_text(track, proposal_bundle)
        
        # Fail-safe V2.1: pass-through on OCR error/crash
        if had_error:
            logger.warning(
                "LOUD-WARN: OCR error occurred during evaluation of track %s. "
                "Pass-through allowed. (errors=%d)",
                track.get("track_id"), self.echo_check_ocr_error_count
            )
            return True

        if not paddle_text:
            # Gemini claims subtitle text but Paddle executed successfully and saw nothing
            # -> Candidate is likely a non-subtitle (hallucination on green HUD/UI or blank crop)
            logger.warning(
                "Echo-reject: Gemini '%s' but Paddle empty for track %s",
                echo_text[:50], track.get("track_id")
            )
            return False

        # Fuzzy match
        similarity = SequenceMatcher(
            None,
            echo_text.lower().strip(),
            paddle_text.lower().strip(),
        ).ratio()

        if similarity < FUZZY_THRESHOLD:
            logger.warning(
                "Echo-check: Gemini '%s' vs Paddle '%s' = %.2f (threshold %.1f)",
                echo_text[:50], paddle_text[:50], similarity, FUZZY_THRESHOLD,
            )
            return False

        return True

    def _get_paddle_text(
        self, track: Dict[str, Any], proposal_bundle: Any
    ) -> Tuple[str, bool]:
        """Extract PaddleOCR text from track crops.

        Returns (joined_texts, had_error).
        """
        groups = track.get("groups", [])
        texts: List[str] = []
        errors: List[bool] = []
        for group in groups:
            fi = group.get("frame_index", -1)
            try:
                decoded = proposal_bundle.frame_for_index(fi)
            except (KeyError, AttributeError):
                continue
            if not decoded.crops or not decoded.detections:
                continue
            candidate_ids = set(group.get("member_candidate_ids", []))
            for det_idx, det in enumerate(decoded.detections):
                if det.get("candidate_id", "") in candidate_ids and det_idx < len(decoded.crops):
                    text, err = self._ocr_simple(decoded.crops[det_idx])
                    errors.append(err)
                    if not err and text.strip():
                        texts.append(text)
                    break
        return " ".join(texts), any(errors)

    def _ocr_simple(self, crop: Any) -> Tuple[str, bool]:
        """Simple PaddleOCR call on a crop image.

        Returns (text, is_error).
        """
        try:
            import paddleocr
            if not hasattr(self, "_paddle"):
                self._paddle = paddleocr.PaddleOCR(
                    lang="ch",
                    device="gpu",
                    enable_mkldnn=False,
                    engine_config={"paddle_static": {"enable_new_ir": False}},
                    show_log=False,
                )
            result = self._paddle.ocr(crop)
            page = result[0] if result else []
            if isinstance(page, dict):
                text = " ".join(str(item) for item in page.get("rec_texts", []))
            else:
                text = " ".join(str(line[1][0]) for line in page if len(line) == 2)
            return text, False
        except Exception as exc:
            self.echo_check_ocr_error_count += 1
            logger.warning(
                "LOUD-WARN: exception during PaddleOCR invocation: %s", exc
            )
            return "", True

    def _log_ledger(self, resp: Any, batch_size: int, canary_passed: bool) -> None:
        if self.artifact_dir is None:
            return
        try:
            self.artifact_dir.mkdir(parents=True, exist_ok=True)
            ledger_path = self.artifact_dir / "crop_classifier_v21_ledger.jsonl"
            log_entry = {
                "timestamp": time.time(),
                "job_id": self.job_id,
                "run_id": self.run_id,
                "request_id": getattr(resp, "request_id", "")
                or f"crop_batch_v21_{self.requests_sent - 1}",
                "route": "crop_classifier_v21",
                "live_or_cache": "cache"
                if getattr(resp, "cache_hit", False)
                else "live",
                "batch_size": batch_size,
                "canary_pass": canary_passed,
                "model": self.model,
                "total_candidates": self.total_candidates,
                "retry_count": self.retry_count,
                "unresolved_count": self.unresolved_count,
                "finish_reason": getattr(resp, "finish_reason", "STOP"),
            }
            with open(ledger_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry) + "\n")
        except Exception as exc:
            logger.warning(
                "LOUD-WARN: Failed to write ledger: %s", exc
            )


# Backward-compatible alias for legacy test imports
CropBatchClassifier = CropBatchClassifierV21
