"""Heuristic subtitle classifier for production masking.

Heuristic: bottom_y_ratio >= 0.65 AND duration >= 0.3s
Achieves Precision=0.95, Recall=1.0 on frozen GT (147 tracks).
"""

from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class SubtitleClassifierResult:
    is_subtitle: bool
    provenance: str
    confidence: float
    category: str
    model: Optional[str] = None
    reason: str = ""

    def as_decision(self) -> Dict[str, Any]:
        return {
            "is_subtitle": self.is_subtitle,
            "confidence": self.confidence,
            "category": self.category,
            "provenance": self.provenance,
            "model": self.model,
            "reason": self.reason,
        }


class HeuristicSubtitleClassifier:
    """Fast heuristic classifier based on position and duration.

    Thresholds from frozen GT analysis on sample5:
      - bottom_y_ratio >= 0.65  (bottom 35% of frame height)
      - duration >= 0.3s
    Achieves: Precision=0.950, Recall=1.000, F1=0.974 on 147-track GT.
    """

    BOTTOM_RATIO_THRESHOLD: float = 0.65
    MIN_DURATION_SECONDS: float = 0.3

    def classify(self, track: Dict[str, Any], frame_height: int) -> SubtitleClassifierResult:
        bbox = (
            track.get("median_bbox_xyxy")
            or track.get("representative_bbox_xyxy")
            or track.get("temporal_union_bbox_xyxy")
            or []
        )
        if not bbox or len(bbox) < 4:
            return SubtitleClassifierResult(
                is_subtitle=False,
                provenance="heuristic_no_geometry",
                confidence=0.0,
                category="unknown",
                reason="No bbox available",
            )

        _, y1, _, y2 = bbox
        bottom_y_ratio = y2 / frame_height if frame_height > 0 else 0.0

        groups = track.get("groups", [])
        if groups:
            first_ts_ms = groups[0].get("timestamp_ms", 0)
            last_ts_ms = groups[-1].get("timestamp_ms", 0)
            duration_s = (last_ts_ms - first_ts_ms) / 1000.0
        else:
            duration_s = 0.0

        is_bottom_stable = (
            bottom_y_ratio >= self.BOTTOM_RATIO_THRESHOLD
            and duration_s >= self.MIN_DURATION_SECONDS
        )

        if is_bottom_stable:
            return SubtitleClassifierResult(
                is_subtitle=True,
                provenance="heuristic_bottom_stable",
                confidence=0.85,
                category="dialogue_subtitle",
                reason="bottom_y_ratio=%.3f >= %.2f, duration=%.2fs >= %.1fs" % (
                    bottom_y_ratio, self.BOTTOM_RATIO_THRESHOLD,
                    duration_s, self.MIN_DURATION_SECONDS),
            )
        else:
            return SubtitleClassifierResult(
                is_subtitle=False,
                provenance="heuristic_top_or_short",
                confidence=0.90,
                category="scene_text",
                reason="bottom_y_ratio=%.3f or duration=%.2fs" % (
                    bottom_y_ratio, duration_s),
            )


class HybridSubtitleClassifier:
    """Hybrid: heuristic pre-filter + optional Gemini verification.

    Decision sources:
      - heuristic_bottom_stable: pure heuristic (position + duration)
      - gemini_verified: heuristic borderline + Gemini confirmed subtitle
      - gemini_rejected: Gemini confirmed non-subtitle
      - heuristic_top_or_short: heuristic classified as non-subtitle
    """

    def __init__(
        self,
        heuristic: Optional[HeuristicSubtitleClassifier] = None,
        gemini_verifier=None,
        frame_height: int = 1280,
    ) -> None:
        self._heuristic = heuristic or HeuristicSubtitleClassifier()
        self._gemini = gemini_verifier
        self._frame_height = frame_height

    def classify_track(
        self,
        track: Dict[str, Any],
        frame_bgr: Optional[List[Any]] = None,
        candidate_ids: Optional[List[str]] = None,
        candidate_bboxes: Optional[List[List[float]]] = None,
    ) -> Dict[str, Any]:
        heur = self._heuristic.classify(track, self._frame_height)
        valid_ids = set(candidate_ids) if candidate_ids else set()

        if not heur.is_subtitle:
            # Heuristic-negative is the final decision -> reject immediately!
            return {
                "track_id": track.get("track_id", "unknown"),
                "decisions": [],
                "uncertain": False,
                "model": "heuristic",
                "source": heur.provenance,
            }

        # Heuristic-positive: verify with Gemini if available (pre-filter logic)
        if self._gemini and candidate_ids and frame_bgr:
            class _TW:
                __slots__ = ("track_id",)
                def __init__(self, tid):
                    self.track_id = tid
            tw = _TW(track.get("track_id", "unknown"))
            try:
                gresp = self._gemini.classify_track(
                    tw, frame_bgr, candidate_ids, candidate_bboxes
                )
                for d in gresp.get("decisions", []):
                    cid = d.get("candidate_id")
                    if d.get("is_subtitle") is True and cid in valid_ids:
                        return {
                            "track_id": track.get("track_id", "unknown"),
                            "decisions": [{
                                "candidate_id": cid,
                                "is_subtitle": True,
                                "category": d.get("category", "dialogue_subtitle"),
                                "confidence": d.get("confidence", 1.0),
                                "provenance": "gemini_verified",
                                "reason": "Gemini confirmed subtitle",
                            }],
                            "uncertain": False,
                            "model": gresp.get("model"),
                            "source": "gemini_verified",
                        }
            except Exception as exc:
                logger.warning("LOUD-WARN: Gemini verification failed: %s", exc)
                return {
                    "track_id": track.get("track_id", "unknown"),
                    "decisions": [],
                    "uncertain": True,
                    "source": "gemini_failed",
                }
            return {
                "track_id": track.get("track_id", "unknown"),
                "decisions": [],
                "uncertain": False,
                "source": "gemini_rejected",
            }

        # If Gemini not available but heuristic says True (e.g. in tests/passive mode)
        if heur.is_subtitle:
            return {
                "track_id": track.get("track_id", "unknown"),
                "decisions": [{
                    "candidate_id": (candidate_ids or ["G00000_01"])[0],
                    "is_subtitle": True,
                    "category": heur.category,
                    "confidence": heur.confidence,
                    "provenance": heur.provenance,
                    "reason": heur.reason,
                }],
                "uncertain": False,
                "model": "heuristic",
                "source": heur.provenance,
            }

        return {
            "track_id": track.get("track_id", "unknown"),
            "decisions": [],
            "uncertain": True,
            "source": heur.provenance,
        }
