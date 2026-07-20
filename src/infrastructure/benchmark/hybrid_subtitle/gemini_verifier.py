from __future__ import annotations
import json, logging, os
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import cv2
from .config import DEFAULT_GEMINI_MODEL
from .video_utils import draw_bboxes, COLOR_MAP
from .core import validate_gemini_decision

logger = logging.getLogger(__name__)

ALLOWED_CATEGORIES = {
    "dialogue_subtitle", "narration_subtitle", "sound_caption",
    "karaoke_or_lyric_subtitle", "scene_text", "watermark", "logo",
    "ui_text", "social_overlay", "title_card", "credits",
    "decorative_text", "unknown"
}

SYSTEM_PROMPT = (
    "You are a vision-language classifier for on-screen text in video frames.\n"
    "Geometry is provided by a local detector. Do NOT re-evaluate or modify "
    "coordinates. Only classify each candidate ID as subtitle or non-subtitle.\n\n"
    "Subtitle definition: text added to the video to represent dialogue, "
    "narration, audio content, captions, on-screen transcripts that replace speech.\n\n"
    "NOT subtitle: scene text, product labels, signs, usernames, watermarks, "
    "logos, UI controls, title cards, decorative text.\n\n"
    "Allowed categories: dialogue_subtitle, narration_subtitle, sound_caption, "
    "karaoke_or_lyric_subtitle, scene_text, watermark, logo, ui_text, "
    "social_overlay, title_card, credits, decorative_text, unknown.\n\n"
    "Return JSON only. Candidate IDs must come from the input."
)

class RepresentativeFrameSelector:
    def __init__(self, max_frames: int = 5):
        self._max = max_frames

    def select(self, track_frames: List[int], track_bboxes: List[List[float]],
               confidences: List[float]) -> List[Tuple[int, List[float], str]]:
        if not track_frames:
            return []
        result = []
        first_fi = track_frames[0]
        last_fi = track_frames[-1]
        first_bbox = track_bboxes[0] if track_bboxes else None
        last_bbox = track_bboxes[-1] if track_bboxes else None
        result.append((first_fi, first_bbox, "first"))
        if last_fi != first_fi:
            result.append((last_fi, last_bbox, "last"))
        if len(track_frames) >= 3:
            mid_idx = len(track_frames) // 2
            result.append((track_frames[mid_idx], track_bboxes[mid_idx], "middle"))
        if confidences and len(confidences) > 1:
            best_idx = max(range(len(confidences)), key=lambda i: confidences[i])
            best_fi = track_frames[best_idx]
            if not any(r[0] == best_fi for r in result):
                result.append((best_fi, track_bboxes[best_idx], "highest_conf"))
        return result[:self._max]


class GeminiTrackVerifier:
    """Unified Gemini verifier for subtitle tracks.

    Exposes both public methods with their original signatures so existing call
    sites keep working.  The verifier must never produce geometry:
    validate_gemini_decision rejects bbox keys in responses.
    """

    def __init__(self, runtime_client, model: str = DEFAULT_GEMINI_MODEL):
        self._client = runtime_client
        self._model = model
        # Diagnostic counters for A2 / STEP F audit.
        self.tracks_received: int = 0
        self.requests_sent: int = 0
        self.cache_hits: int = 0
        self.responses_parsed_ok: int = 0
        self.parse_failures: int = 0
        self.fail_open_count: int = 0

    # ----- classify_track (original GeminiSubtitleSemanticVerifier path) -----

    def classify_track(self, track, frame_bgr: List[Any],
                       candidate_ids: List[str],
                       candidate_bboxes: List[List[float]]) -> Dict[str, Any]:
        """Classify track using pre-decoded BGR frames."""
        if not frame_bgr:
            self.tracks_received += 1
            self.fail_open_count += 1
            return {"track_id": track.track_id, "decisions": [], "uncertain": True}

        self.tracks_received += 1
        user_prompt = self._build_user_prompt(candidate_ids, candidate_bboxes)
        try:
            import base64
            content_parts = [{"text": user_prompt}]
            for frame in frame_bgr:
                _, png = cv2.imencode(".png", frame)
                content_parts.append({
                    "inline_data": {
                        "mime_type": "image/png",
                        "data": base64.b64encode(png.tobytes()).decode("ascii")
                    }
                })
            from src.infrastructure.gemini.contracts import GeminiRequest
            req = GeminiRequest(
                contents=[{"parts": content_parts}],
                model=self._model,
                generation_config={"temperature": 0.1},
                system_instruction=SYSTEM_PROMPT,
                request_id=track.track_id,
                source_kind="benchmark_subtitle_verifier",
                api_surface="benchmark",
            )
            self.requests_sent += 1
            resp = self._client.generate_content(req, use_cache=True)
            if getattr(resp, "cache_hit", False):
                self.cache_hits += 1
            text = resp.text if hasattr(resp, "text") else ""
            return self._parse_response_text(text, track.track_id, candidate_ids, resp)
        except Exception as exc:
            self.parse_failures += 1
            self.fail_open_count += 1
            logger.warning(
                "LOUD-WARN: Gemini classification/network failed for track %s (fail-open used): %s",
                track.track_id,
                str(exc)[:200],
            )
            return {"track_id": track.track_id, "decisions": [],
                    "uncertain": True, "error": str(exc)[:200]}

    def _build_user_prompt(self, candidate_ids: List[str],
                            candidate_bboxes: Optional[List[List[float]]] = None) -> str:
        lines = [
            "Classify the following candidate IDs in the provided video frames.",
            f"Candidate IDs: {json.dumps(candidate_ids)}",
        ]
        if candidate_bboxes:
            lines.append("Each candidate's bounding box [x1, y1, x2, y2] in pixel "
                        "coordinates (top-left origin), in the same order as the IDs above:")
            lines.append(json.dumps(candidate_bboxes))
            lines.append("Frames are annotated with a yellow rectangle and the candidate ID label.")
        else:
            lines.append("Frames are shown in order.")
        lines.append("Return a JSON object matching this schema:")
        lines.append("{")
        lines.append('  "track_id": "<same track_id>",')
        lines.append('  "decisions": [')
        lines.append("    {")
        lines.append('      "candidate_id": "<from input list>",')
        lines.append('      "is_subtitle": true/false,')
        lines.append('      "category": "<one of allowed categories>",')
        lines.append('      "confidence": 0.0-1.0,')
        lines.append('      "transcription": null or string,')
        lines.append('      "reason_code": "<short reason>"')
        lines.append("    }")
        lines.append("  ],")
        lines.append('  "uncertain": true/false')
        lines.append("}")
        return "\n".join(lines)

    def _parse_response_text(self, text: str, track_id: str,
                                  valid_ids: List[str], resp: Any) -> Dict[str, Any]:
            """Parse and validate Gemini response; reject geometry via validate_gemini_decision."""
            from src.infrastructure.gemini.structured_output import parse_json
            try:
                data = parse_json(text)
            except Exception as exc:
                self.parse_failures += 1
                self.fail_open_count += 1
                logger.warning(
                    "LOUD-WARN: Verifier parse_failed for track %s (fail-open): "
                    "first_200_chars='%s'",
                    track_id,
                    text[:200],
                )
                return {"track_id": track_id, "decisions": [],
                        "uncertain": True, "error": f"parse_failed: {exc}"}

            if isinstance(data, dict) and "decisions" in data:
                decisions = data["decisions"]
                uncertain = bool(data.get("uncertain", False))
            elif isinstance(data, list):
                decisions = data
                uncertain = False
            else:
                self.parse_failures += 1
                self.fail_open_count += 1
                logger.warning(
                    "LOUD-WARN: Verifier schema_invalid for track %s (fail-open): "
                    "first_200_chars='%s'",
                    track_id,
                    text[:200],
                )
                return {"track_id": track_id, "decisions": [],
                        "uncertain": True, "error": "schema_invalid"}

            valid_set = set(valid_ids)
            validated = []
            for d in decisions:
                cid = d.get("candidate_id")
                if cid not in valid_set:
                    continue
                try:
                    validate_gemini_decision(
                        {"track_id": track_id, "decisions": [d]},
                        track_id,
                        valid_set,
                    )
                except ValueError:
                    continue
                if d.get("category") not in ALLOWED_CATEGORIES:
                    d["category"] = "unknown"
                validated.append(d)

            self.responses_parsed_ok += 1
            if uncertain or not validated:
                self.fail_open_count += 1
            return {
                "track_id": track_id,
                "decisions": validated,
                "uncertain": uncertain or not validated,
                "model": getattr(resp, "model", None) if resp else None,
                "cache_hit": getattr(resp, "cache_hit", False) if resp else None,
                "finish_reason": getattr(resp, "finish_reason", None) if resp else None,
            }

    # ----- classify (original LiveGeminiTrackVerifier path) -----

    def classify(self, track: Dict, rep_frames: List[Tuple[int, Any, str]],
                 candidate_ids: List[str], cap) -> Dict:
        """Classify track using representative frames read from video capture."""
        if not rep_frames:
            return {"track_id": track["track_id"], "decisions": [],
                    "uncertain": True}
        import base64
        content_parts = [{"text": self._build_prompt(candidate_ids, track)}]
        for fi, bbox, role in rep_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ret, frame = cap.read()
            if not ret:
                continue
            annotated = draw_bboxes(frame, [{"bbox_xyxy": bbox, "color_key": "yellow",
                                            "label": f"{role[0].upper()}", "thickness": 2}], COLOR_MAP)
            _, png_buf = cv2.imencode(".png", annotated, [cv2.IMWRITE_PNG_COMPRESSION, 6])
            content_parts.append({
                "inline_data": {
                    "mime_type": "image/png",
                    "data": base64.b64encode(png_buf.tobytes()).decode("ascii")
                }
            })
        try:
            from src.infrastructure.gemini.contracts import GeminiRequest
            req = GeminiRequest(
                contents=[{"parts": content_parts}],
                model=self._model,
                generation_config={"temperature": 0.1},
                system_instruction=(
                    "Geometry is provided by a detector. Do NOT modify or re-evaluate "
                    "coordinates. Only classify candidate IDs. Return valid JSON. "
                    "Allowed categories: dialogue_subtitle, narration_subtitle, "
                    "sound_caption, karaoke_or_lyric_subtitle, scene_text, watermark, "
                    "logo, ui_text, social_overlay, title_card, credits, decorative_text, unknown."
                ),
                request_id=track["track_id"],
                source_kind="benchmark_subtitle_verifier",
                api_surface="benchmark",
            )
            resp = self._client.generate_content(req, use_cache=True)
            text = resp.text if hasattr(resp, "text") else ""
            parsed = json.loads(text)
            validated = validate_gemini_decision(parsed, track["track_id"], set(candidate_ids))
            return validated
        except Exception as exc:
            return {"track_id": track["track_id"], "decisions": [],
                    "uncertain": True, "error": str(exc)[:200]}

    def _build_prompt(self, candidate_ids: List[str], track: Dict) -> str:
        return (
            "Classify each candidate ID as subtitle or non-subtitle.\\n"
            "Candidate IDs in this track (" + track["track_id"] + "):\\n"
            + json.dumps(candidate_ids) + "\\n\\n"
            "Track frames: " + str(len(track["groups"])) + " frames "
            "from frame " + str(track.get("start_frame")) + " to " + str(track.get("end_frame")) + ".\\n\\n"
            "Return: {\"track_id\": \"" + track["track_id"] + "\", "
            "\"decisions\": [{\"candidate_id\": \"<id>\", "
            "\"is_subtitle\": true/false, \"category\": \"<category>\", "
            "\"confidence\": 0.0-1.0, \"transcription\": null/str, "
            "\"reason_code\": \"<short>\"}], \"uncertain\": true/false}"
        )


# Backward-compatible aliases so existing call sites keep working.
GeminiSubtitleSemanticVerifier = GeminiTrackVerifier
LiveGeminiTrackVerifier = GeminiTrackVerifier
