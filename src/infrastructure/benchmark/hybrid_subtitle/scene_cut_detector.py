"""Scene-cut detector for V2.1 R3 static rule.

Detects hard cuts, fades, and flashes between consecutive frames.
Used by R3 to distinguish static UI elements (which survive cuts) from
long-duration subtitles (which reset at cuts).

Bất biến V2.1: any error → NEUTRAL (safe on both FP and FN sides).
"""

from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SceneCutResult:
    has_cut: bool
    cut_type: str  # "hard_cut" | "fade" | "flash" | "none"
    confidence: float
    frame_pairs_analyzed: int = 0
    cut_positions: list[int] = field(default_factory=list)


class SceneCutDetector:
    """Detect scene transitions between video frames.

    Uses three complementary methods:
    1. Histogram difference — robust to motion, sensitive to scene change
    2. Pixel-wise absolute difference — detects flashes/fades
    3. Edge change ratio — detects structural scene changes

    NEUTRAL on any error (exception → return no cut).
    """

    HIST_CUT_THRESHOLD: float = 0.5  # Normalized histogram intersection
    PIXEL_CUT_THRESHOLD: float = 30.0  # Mean pixel diff for flash detection
    FADE_THRESHOLD: float = 15.0  # Mean pixel diff for fade detection
    EDGE_CUT_THRESHOLD: float = 0.4  # Edge change ratio

    def analyze(
        self,
        track_groups: List[Dict[str, Any]],
        proposal_bundle: Any,
    ) -> SceneCutResult:
        """Analyze whether a track spans a scene cut.

        Parameters
        ----------
        track_groups : list[dict]
            Track groups with frame_index entries.
        proposal_bundle : ProposalBundle
            Decoded frame cache.

        Returns
        -------
        SceneCutResult with cut detection.
        """
        if len(track_groups) < 2:
            return SceneCutResult(
                has_cut=False, cut_type="none", confidence=0.0,
            )

        frame_indices = []
        for g in track_groups:
            fi = g.get("frame_index", -1)
            if fi >= 0:
                frame_indices.append(fi)

        if len(frame_indices) < 2:
            return SceneCutResult(
                has_cut=False, cut_type="none", confidence=0.0,
            )

        consecutive_pairs = 0
        cut_positions: List[int] = []

        for i in range(len(frame_indices) - 1):
            fi0 = frame_indices[i]
            fi1 = frame_indices[i + 1]

            if fi1 - fi0 > 3:
                continue

            try:
                frame0 = proposal_bundle.frame_for_index(fi0)
                frame1 = proposal_bundle.frame_for_index(fi1)
            except (KeyError, AttributeError):
                continue

            img0 = getattr(frame0, "image", None)
            img1 = getattr(frame1, "image", None)
            if img0 is None or img1 is None:
                continue

            pair_result = self._check_pair(img0, img1)
            if pair_result["is_cut"]:
                cut_positions.append(fi1)
            consecutive_pairs += 1

        if not cut_positions:
            return SceneCutResult(
                has_cut=False,
                cut_type="none",
                confidence=0.0,
                frame_pairs_analyzed=consecutive_pairs,
            )

        confidence = min(1.0, len(cut_positions) / max(consecutive_pairs, 1) * 2.0)
        # Dominant type determined by threshold crossing (all cuts are hard_cut)
        # In practice: any cut > PIXEL_CUT_THRESHOLD is meaningful

        return SceneCutResult(
            has_cut=True,
            cut_type="hard_cut",
            confidence=confidence,
            frame_pairs_analyzed=consecutive_pairs,
            cut_positions=cut_positions,
        )

    def _check_pair(self, img0: Any, img1: Any) -> Dict[str, Any]:
        """Check a single frame pair for scene transitions.

        Returns dict with is_cut, type, confidence.
        """
        H0, W0 = img0.shape[:2]
        H1, W1 = img1.shape[:2]
        if H0 != H1 or W0 != W1:
            return {"is_cut": False, "type": "none", "confidence": 0.0}

        try:
            # Convert to grayscale
            if len(img0.shape) == 3:
                gray0 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
                gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            else:
                gray0 = img0
                gray1 = img1

            # Method 1: Histogram difference
            hist0 = cv2.calcHist([gray0], [0], None, [64], [0, 256])
            hist1 = cv2.calcHist([gray1], [0], None, [64], [0, 256])
            cv2.normalize(hist0, hist0)
            cv2.normalize(hist1, hist1)
            hist_sim = cv2.compareHist(hist0, hist1, cv2.HISTCMP_INTERSECT)

            # Method 2: Pixel-wise absolute difference
            diff = cv2.absdiff(gray0, gray1)
            mean_diff = float(diff.mean())

            # Method 3: Edge change ratio
            edges0 = cv2.Canny(gray0, 50, 150)
            edges1 = cv2.Canny(gray1, 50, 150)
            edge_diff = cv2.absdiff(edges0, edges1)
            edge_change = float(edge_diff.mean()) / 255.0 if edge_diff.size > 0 else 0.0

            # Classification
            if mean_diff > self.PIXEL_CUT_THRESHOLD and edge_change > self.EDGE_CUT_THRESHOLD:
                return {
                    "is_cut": True,
                    "type": "hard_cut",
                    "confidence": min(1.0, mean_diff / 255.0),
                }
            elif mean_diff > self.FADE_THRESHOLD and hist_sim > self.HIST_CUT_THRESHOLD * 0.7:
                return {
                    "is_cut": True,
                    "type": "fade",
                    "confidence": min(1.0, mean_diff / 60.0),
                }
            elif mean_diff > self.PIXEL_CUT_THRESHOLD * 0.8 and hist_sim < 0.1:
                return {
                    "is_cut": True,
                    "type": "flash",
                    "confidence": min(1.0, mean_diff / 255.0 * 2.0),
                }

            return {"is_cut": False, "type": "none", "confidence": 0.0}

        except Exception as exc:
            logger.warning("SceneCutDetector error on frame pair: %s", exc)
            return {"is_cut": False, "type": "none", "confidence": 0.0}
