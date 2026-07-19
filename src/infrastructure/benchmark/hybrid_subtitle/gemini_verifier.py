from __future__ import annotations
import json, logging, os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import cv2
from .video_utils import draw_bboxes, COLOR_MAP

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
        # Frame indices are sorted; we want representative samples
        result = []
        first_fi = track_frames[0]
        last_fi = track_frames[-1]
        first_bbox = track_bboxes[0] if track_bboxes else None
        last_bbox = track_bboxes[-1] if track_bboxes else None

        # Always first
        result.append((first_fi, first_bbox, "first"))
        # Always last (if not the same frame)
        if last_fi != first_fi:
            result.append((last_fi, last_bbox, "last"))
        # Middle
        if len(track_frames) >= 3:
            mid_idx = len(track_frames) // 2
            result.append((track_frames[mid_idx], track_bboxes[mid_idx], "middle"))
        # Highest confidence frame
        if confidences and len(confidences) > 1:
            best_idx = max(range(len(confidences)), key=lambda i: confidences[i])
            best_fi = track_frames[best_idx]
            if not any(r[0] == best_fi for r in result):
                result.append((best_fi, track_bboxes[best_idx], "highest_conf"))
        # Take first 5
        return result[:self._max]


class GeminiSubtitleSemanticVerifier:
    def __init__(self, runtime_client, model: str = "gemini-3.1-flash-lite"):
        self._client = runtime_client
        self._model = model

    def classify_track(self, track, frame_bgr: List[Any],
                       candidate_ids: List[str],
                       candidate_bboxes: List[List[float]]) -> Dict[str, Any]:
        """Frame_bgr is a list of frames already annotated with candidate IDs.
        candidate_ids is the list of IDs shown in those frames.
        Returns the Gemini decision dict for this track.
        """
        if not frame_bgr:
            return {"track_id": track.track_id, "decisions": [], "uncertain": True}

        user_prompt = (
            "Classify the following candidate IDs in the provided video frames.\n\n"
            "Candidate IDs: " + json.dumps(candidate_ids) + "\n\n"
            "Frames are shown in order. Return a JSON object matching this schema:\n"
            "{\n"
            "  \"track_id\": \"<same track_id>\",\n"
            "  \"decisions\": [\n"
            "    {\n"
            "      \"candidate_id\": \"<from input list>\",\n"
            "      \"is_subtitle\": true/false,\n"
            "      \"category\": \"<one of allowed categories>\",\n"
            "      \"confidence\": 0.0-1.0,\n"
            "      \"transcription\": null or string,\n"
            "      \"reason_code\": \"<short reason>\"\n"
            "    }\n"
            "  ],\n"
            "  \"uncertain\": true/false\n"
            "}\n"
        )

        try:
            # Encode frames to base64 inline (small PNG bytes)
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

            request_payload = {
                "contents": [{"role": "user", "parts": content_parts}],
                "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
                "generationConfig": {"temperature": 0.1},
            }
            from src.infrastructure.gemini.contracts import GeminiRequest
            req = GeminiRequest(
                contents=request_payload["contents"],
                model=self._model,
                generation_config=request_payload["generationConfig"],
                system_instruction=SYSTEM_PROMPT,
                request_id=track.track_id,
                source_kind="benchmark_subtitle_verifier",
                api_surface="benchmark",
            )
            resp = self._client.generate_content(req, use_cache=True)
            text = resp.text if hasattr(resp, "text") else ""
            return self._parse_response(text, track.track_id, candidate_ids, resp)
        except Exception as exc:
            logger.error("Gemini classification failed for %s: %s", track.track_id, exc)
            return {"track_id": track.track_id, "decisions": [],
                    "uncertain": True, "error": str(exc)[:200]}

    def _parse_response(self, text: str, track_id: str,
                        valid_ids: List[str], resp: Any) -> Dict[str, Any]:
        """Parse and validate Gemini response, rejecting unknown candidate IDs."""
        from src.infrastructure.gemini.structured_output import parse_json
        try:
            data = parse_json(text)
        except Exception as exc:
            return {"track_id": track_id, "decisions": [],
                    "uncertain": True, "error": f"parse_failed: {exc}"}

        if isinstance(data, dict) and "decisions" in data:
            decisions = data["decisions"]
            uncertain = bool(data.get("uncertain", False))
        elif isinstance(data, list):
            decisions = data
            uncertain = False
        else:
            return {"track_id": track_id, "decisions": [],
                    "uncertain": True, "error": "schema_invalid"}

        valid_set = set(valid_ids)
        validated = []
        for d in decisions:
            cid = d.get("candidate_id")
            if cid not in valid_set:
                continue  # Reject unknown IDs
            if d.get("category") not in ALLOWED_CATEGORIES:
                d["category"] = "unknown"
            validated.append(d)

        return {
            "track_id": track_id,
            "decisions": validated,
            "uncertain": uncertain,
            "model": getattr(resp, "model", None) if resp else None,
            "cache_hit": getattr(resp, "cache_hit", False) if resp else False,
            "finish_reason": getattr(resp, "finish_reason", None) if resp else None,
        }
