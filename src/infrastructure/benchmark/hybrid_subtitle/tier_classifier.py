"""Three-level tier classifier for subtitle tracks (V2.1 architecture).

Separates tracks into:
  - sure-YES (THU HẸP): cadence position-stack only, must pass R1+R3
  - sure-NO: explicit rejection criteria (persistence/confidence)
  - gray zone: everything else → R1 → R2 → R3 → STEP 5

Bất biến V2.1: reject-only, neutral on insufficient evidence.
"""

from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class TierResult:
    tier: str  # "sure_yes" | "sure_no" | "gray_zone"
    provenance: str
    reason: str
    confidence: float


class TierClassifier:
    """Three-level tier classification for subtitle track candidates.

    sure-YES (THU HẸP): only tracks with cadence position-stack (multiple
    consecutive tracks appearing at the same vertical position over time).
    These bypass R2 (language loop) but must pass R1 + R3 before final accept.

    sure-NO: tracks that are clearly non-subtitle by persistence/confidence
    heuristics (same as prior rounds — rejected by position+duration rules).

    gray zone: everything uncertain proceeds through R1 → R2 → R3 → Gemini.
    """

    # Number of consecutive tracks at same vertical band to qualify as cadence
    CADENCE_MIN_COUNT: int = 2
    # Vertical band height (fraction of frame height) for same-position matching
    CADENCE_BAND_FRAC: float = 0.10
    # Max time gap between cadence entries (seconds)
    CADENCE_MAX_GAP_SEC: float = 5.0

    # sure-NO thresholds
    SURE_NO_MIN_CONFIDENCE: float = 0.95  # Not used; see HeuristicSubtitleClassifier
    SURE_NO_MIN_DURATION: float = 0.3

    def __init__(self, frame_height: int) -> None:
        self._frame_height = frame_height

    def classify(self, track: Dict[str, Any], all_tracks: List[Dict[str, Any]]) -> TierResult:
        """Classify a single track into one of three tiers.

        Parameters
        ----------
        track : dict
            A single track dict with 'groups', 'median_bbox_xyxy', 'track_id', etc.
        all_tracks : list[dict]
            All tracks from the same video, used for cadence detection.

        Returns
        -------
        TierResult with tier, provenance, reason, confidence.
        """
        groups = track.get("groups", [])
        if not groups:
            return TierResult("gray_zone", "no_groups", "No groups available", 0.0)

        bbox = (
            track.get("median_bbox_xyxy")
            or track.get("temporal_union_bbox_xyxy")
            or []
        )
        if not bbox or len(bbox) < 4:
            return TierResult("gray_zone", "no_bbox", "No bbox available", 0.0)

        # --- sure-NO check: position+duration heuristic ---
        _, y1, _, y2 = bbox
        bottom_y_ratio = y2 / self._frame_height if self._frame_height > 0 else 0.0

        first_ts_ms = groups[0].get("timestamp_ms", 0)
        last_ts_ms = groups[-1].get("timestamp_ms", 0)
        duration_s = (last_ts_ms - first_ts_ms) / 1000.0 if last_ts_ms >= first_ts_ms else 0.0

        is_top_or_short = (
            bottom_y_ratio < 0.65 or duration_s < self.SURE_NO_MIN_DURATION
        )
        if is_top_or_short:
            return TierResult(
                "sure_no",
                "tier_position_duration",
                f"bottom_y_ratio={bottom_y_ratio:.3f} or duration={duration_s:.3f}s",
                0.90,
            )

        # --- sure-YES check: cadence position-stack ---
        cadence_info = self._detect_cadence(track, all_tracks)
        if cadence_info["has_cadence"]:
            return TierResult(
                "sure_yes",
                "tier_cadence_stack",
                f"Cadence stack with {cadence_info['cadence_count']} tracks "
                f"at band {cadence_info['band_key']}",
                0.85,
            )

        # --- gray zone ---
        return TierResult(
            "gray_zone",
            "tier_no_cadence",
            f"Bottom stable ({bottom_y_ratio:.3f}, {duration_s:.2f}s) but no cadence",
            0.50,
        )

    def _detect_cadence(self, track: Dict[str, Any], all_tracks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Detect if this track is part of a cadence position-stack.

        A cadence is defined as >= CADENCE_MIN_COUNT tracks at the same
        vertical band, appearing consecutively over time.

        Returns dict with has_cadence, cadence_count, band_key.
        """
        bbox = (
            track.get("median_bbox_xyxy")
            or track.get("temporal_union_bbox_xyxy")
            or []
        )
        if not bbox or len(bbox) < 4:
            return {"has_cadence": False, "cadence_count": 0, "band_key": ""}

        # Get this track's vertical band
        _, y1, _, y2 = bbox
        y_center = (y1 + y2) / 2.0
        band_size = self._frame_height * self.CADENCE_BAND_FRAC
        band_key = f"band_{int(y_center / band_size)}" if band_size > 0 else "band_0"

        groups = track.get("groups", [])
        track_start_ms = groups[0].get("timestamp_ms", 0) if groups else 0
        track_end_ms = groups[-1].get("timestamp_ms", 0) if groups else 0

        # Count tracks in same band that are temporally adjacent
        same_band_tracks = []
        for other in all_tracks:
            if other.get("track_id") == track.get("track_id"):
                continue
            other_bbox = (
                other.get("median_bbox_xyxy")
                or other.get("temporal_union_bbox_xyxy")
                or []
            )
            if not other_bbox or len(other_bbox) < 4:
                continue
            _, oy1, _, oy2 = other_bbox
            oy_center = (oy1 + oy2) / 2.0
            oband_key = f"band_{int(oy_center / band_size)}" if band_size > 0 else "band_0"
            if oband_key != band_key:
                continue

            other_groups = other.get("groups", [])
            other_start_ms = other_groups[0].get("timestamp_ms", 0) if other_groups else 0
            other_end_ms = other_groups[-1].get("timestamp_ms", 0) if other_groups else 0

            # Check temporal adjacency: gap between end of one and start of next
            gap1 = abs(other_end_ms - track_start_ms)
            gap2 = abs(track_end_ms - other_start_ms)
            min_gap_ms = min(gap1, gap2)
            if min_gap_ms / 1000.0 <= self.CADENCE_MAX_GAP_SEC:
                same_band_tracks.append(other)

        has_cadence = len(same_band_tracks) + 1 >= self.CADENCE_MIN_COUNT

        return {
            "has_cadence": has_cadence,
            "cadence_count": len(same_band_tracks) + 1,
            "band_key": band_key,
        }
