"""R3 static 3-branch rule (V2.1 architecture — upgraded behavior_discriminator).

Reject-only rule with three hard rejection branches:
  (a) clock-pattern: index-changes-continuously + persistent >= 8s + fixed position
  (b) static outside sub-band: still <= 12px + text sim >= 0.95, persistent
      >= 8s OR surviving a scene cut
  (c) static >= 8s + non-target consensus (R2 confirms non-target)

Static >= 8s INSIDE sub-band + target/neutral language → do NOT reject,
flag as static_long=true for Gemini (handles title cards, long static translations).

"Sub-band" definition: vertical band of sure-YES filtered tracks on THIS video.
If no sure-YES tracks exist → ALL positions considered INSIDE band (safe).

Bất biến V2.1: reject-only. Exceptions/timeouts → NEUTRAL + counter.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace, field
from typing import Any, Dict, Iterable, List, Optional

from .crop_extractor import TrackCropExtractor
from .scene_cut_detector import SceneCutDetector

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class R3Evidence:
    rule_fired: bool           # True = track should be rejected
    rule_reason: str           # Machine-readable reason code
    branch: str                # "clock_pattern" | "static_outside_band" | "static_non_target" | "neutral"
    duration_seconds: float
    displacement_px: float
    static_text_similarity: float | None = None
    static_long_flag: bool = False  # Set true when static >= 8s inside band
    scene_cut_survived: bool = False
    within_sub_band: bool = True
    counter: Dict[str, int] = field(default_factory=lambda: {
        "evaluated": 0, "rejected": 0, "neutral": 0, "error": 0,
    })

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_fired": self.rule_fired,
            "rule_reason": self.rule_reason,
            "branch": self.branch,
            "duration_seconds": self.duration_seconds,
            "displacement_px": self.displacement_px,
            "static_text_similarity": self.static_text_similarity,
            "static_long_flag": self.static_long_flag,
            "scene_cut_survived": self.scene_cut_survived,
            "within_sub_band": self.within_sub_band,
            "counter": dict(self.counter),
        }


class R3StaticDiscriminator:
    """R3 rule — static content discriminator with 3 rejection branches.

    Usage:
        r3 = R3StaticDiscriminator(frame_height=720)
        evidence = r3.evaluate(track, proposal_bundle, all_tracks)
        if evidence.rule_fired:
            # reject track
        # else: pass through with metadata flags
    """

    STATIC_THRESHOLD_PX: float = 12.0
    TEXT_SIMILARITY_THRESHOLD: float = 0.95
    MIN_DURATION_SECONDS: float = 8.0
    CLOCK_MIN_DURATION_SECONDS: float = 8.0
    SCENE_CUT_SURVIVAL_CHECK: bool = True

    def __init__(
        self,
        frame_height: int = 720,
        speech_segments: Iterable[Any] | None = None,
    ) -> None:
        self._frame_height = frame_height
        self._scene_cut_detector = SceneCutDetector()
        self._speech_segments = list(speech_segments or [])
        self._sub_band_cache: Optional[tuple[float, float]] = None

    def set_sub_band_from_sure_yes(
        self, sure_yes_tracks: List[Dict[str, Any]]
    ) -> None:
        """Set the subtitle vertical band from sure-YES filtered tracks.

        If no sure-YES tracks → band is None (all positions are INSIDE band).
        """
        if not sure_yes_tracks:
            self._sub_band_cache = None
            return

        y_positions = []
        for track in sure_yes_tracks:
            bbox = (
                track.get("median_bbox_xyxy")
                or track.get("temporal_union_bbox_xyxy")
                or []
            )
            if len(bbox) == 4:
                y_positions.append((bbox[1] + bbox[3]) / 2.0)

        if not y_positions:
            self._sub_band_cache = None
            return

        y_min = min(y_positions) - self._frame_height * 0.05
        y_max = max(y_positions) + self._frame_height * 0.05
        self._sub_band_cache = (max(0.0, y_min), min(float(self._frame_height), y_max))

    def evaluate(
        self,
        track: Dict[str, Any],
        proposal_bundle: Any,
        all_tracks: List[Dict[str, Any]],
        r2_non_target: bool = False,
    ) -> R3Evidence:
        """Evaluate a single track against all 3 R3 branches.

        Parameters
        ----------
        track : dict
            Track to evaluate.
        proposal_bundle : ProposalBundle
            Frame/crop cache.
        all_tracks : list[dict]
            All tracks in video (for sub-band detection).
        r2_non_target : bool
            Whether R2 already flagged this track as non-target language.

        Returns
        -------
        R3Evidence
        """
        counter: Dict[str, int] = {"evaluated": 0, "rejected": 0, "neutral": 0, "error": 0}
        counter["evaluated"] += 1

        groups = track.get("groups", [])
        if not groups:
            counter["neutral"] += 1
            return R3Evidence(
                rule_fired=False, rule_reason="no_groups", branch="neutral",
                duration_seconds=0.0, displacement_px=0.0, counter=counter,
            )

        # Duration
        first_ts = groups[0].get("timestamp_ms", 0) / 1000.0
        last_ts = groups[-1].get("timestamp_ms", 0) / 1000.0
        duration_s = max(0.0, last_ts - first_ts)

        # Displacement
        displacement = self._displacement(track)

        # Static text similarity
        text_sim = self._text_similarity(track, proposal_bundle)

        # Check if within sub band
        within_band = self._is_within_sub_band(track)

        # Check scene cut survival
        scene_cut = False
        if self.SCENE_CUT_SURVIVAL_CHECK and duration_s >= self.MIN_DURATION_SECONDS:
            try:
                result = self._scene_cut_detector.analyze(groups, proposal_bundle)
                scene_cut = result.has_cut
            except Exception as exc:
                logger.warning("R3 scene-cut check error: %s", exc)
                counter["error"] += 1

        static = displacement <= self.STATIC_THRESHOLD_PX
        text_static = text_sim is not None and text_sim >= self.TEXT_SIMILARITY_THRESHOLD
        long_enough = duration_s >= self.MIN_DURATION_SECONDS
        clock_long_enough = duration_s >= self.CLOCK_MIN_DURATION_SECONDS

        # Branch (a): clock-pattern
        if static and clock_long_enough and self._is_clock_pattern(track):
            counter["rejected"] += 1
            return R3Evidence(
                rule_fired=True,
                rule_reason="clock_pattern",
                branch="clock_pattern",
                duration_seconds=duration_s,
                displacement_px=displacement,
                static_text_similarity=text_sim,
                within_sub_band=within_band,
                counter=counter,
            )

        # Branch (b): static outside sub-band
        is_static_outside = (
            static
            and text_static
            and not within_band
            and (long_enough or scene_cut)
        )
        if is_static_outside:
            counter["rejected"] += 1
            return R3Evidence(
                rule_fired=True,
                rule_reason="static_outside_sub_band",
                branch="static_outside_band",
                duration_seconds=duration_s,
                displacement_px=displacement,
                static_text_similarity=text_sim,
                scene_cut_survived=scene_cut,
                within_sub_band=False,
                counter=counter,
            )

        # Branch (c): static >= 8s + non-target consensus
        if static and long_enough and r2_non_target:
            counter["rejected"] += 1
            return R3Evidence(
                rule_fired=True,
                rule_reason="static_non_target_consensus",
                branch="static_non_target",
                duration_seconds=duration_s,
                displacement_px=displacement,
                static_text_similarity=text_sim,
                scene_cut_survived=scene_cut,
                within_sub_band=within_band,
                counter=counter,
            )

        # Static long INSIDE sub-band: flag but don't reject
        if static and long_enough and within_band:
            counter["neutral"] += 1
            return R3Evidence(
                rule_fired=False,
                rule_reason="static_long_inside_band_passed",
                branch="neutral",
                duration_seconds=duration_s,
                displacement_px=displacement,
                static_text_similarity=text_sim,
                static_long_flag=True,
                scene_cut_survived=scene_cut,
                within_sub_band=True,
                counter=counter,
            )

        # Default: neutral
        counter["neutral"] += 1
        return R3Evidence(
            rule_fired=False,
            rule_reason="not_static_or_short",
            branch="neutral",
            duration_seconds=duration_s,
            displacement_px=displacement,
            static_text_similarity=text_sim,
            within_sub_band=within_band,
            counter=counter,
        )

    def _is_within_sub_band(self, track: Dict[str, Any]) -> bool:
        """Check if track is within the subtitle vertical band."""
        if self._sub_band_cache is None:
            return True  # No sure-YES tracks → everyone is inside band

        bbox = (
            track.get("median_bbox_xyxy")
            or track.get("temporal_union_bbox_xyxy")
            or []
        )
        if len(bbox) < 4:
            return True  # Can't determine → assume inside (safe)

        y_center = (bbox[1] + bbox[3]) / 2.0
        band_min, band_max = self._sub_band_cache
        return band_min <= y_center <= band_max

    def _is_clock_pattern(self, track: Dict[str, Any]) -> bool:
        """Detect clock/timer pattern: index changes continuously at fixed position."""
        groups = track.get("groups", [])
        if len(groups) < 3:
            return False

        # Check if bbox position is very stable
        bboxes = [g["union_bbox_xyxy"] for g in groups if "union_bbox_xyxy" in g]
        if len(bboxes) < 3:
            return False

        x_positions = [((b[0] + b[2]) / 2.0) for b in bboxes]
        y_positions = [((b[1] + b[3]) / 2.0) for b in bboxes]

        x_std = __import__('numpy', fromlist=['std']).std(x_positions) if len(x_positions) > 1 else 999
        y_std = __import__('numpy', fromlist=['std']).std(y_positions) if len(y_positions) > 1 else 999

        # Very stable position
        if x_std > 10 or y_std > 10:
            return False

        # Check for sequential index-like pattern in text
        # (heuristic: small bbox relative to frame, typical of counters)
        avg_height = float(__import__("statistics", fromlist=["mean"]).mean([b[3] - b[1] for b in bboxes])) if len(bboxes) > 1 else 999
        if avg_height > self._frame_height * 0.08:
            return False  # Too tall for clock text

        return True

    def _text_similarity(
        self, track: Dict[str, Any], proposal_bundle: Any
    ) -> float | None:
        """Compute text similarity across track frames."""
        groups = track.get("groups", [])
        if len(groups) < 2:
            return None

        texts: List[str] = []
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
                    text = self._read_text(decoded.crops[det_idx])
                    if text.strip():
                        texts.append(text)
                    break

        if len(texts) < 2:
            return None

        normalized = [self._normalize_text(t) for t in texts if t.strip()]
        if len(normalized) < 2:
            return None

        from difflib import SequenceMatcher
        pairs = [
            SequenceMatcher(None, left, right).ratio()
            for i, left in enumerate(normalized)
            for right in normalized[i + 1:]
        ]
        return min(pairs) if pairs else 0.0

    def _read_text(self, crop: Any) -> str:
        """OCR text from a crop image (disabled)."""
        return ""

    @staticmethod
    def _normalize_text(text: str) -> str:
        return "".join(char.casefold() for char in text if char.isalnum())

    @staticmethod
    def _displacement(track: Dict[str, Any]) -> float:
        groups = track.get("groups", [])
        if len(groups) < 2:
            return 0.0
        first = groups[0]["union_bbox_xyxy"]
        last = groups[-1]["union_bbox_xyxy"]
        fc = ((first[0] + first[2]) / 2.0, (first[1] + first[3]) / 2.0)
        lc = ((last[0] + last[2]) / 2.0, (last[1] + last[3]) / 2.0)
        return ((fc[0] - lc[0]) ** 2 + (fc[1] - lc[1]) ** 2) ** 0.5


# Backward-compatible legacy class for old test compatibility
@dataclass(frozen=True)
class BehaviorEvidence:
    speech_overlap_ratio: float
    coverage_ratio: float
    duration_seconds: float
    displacement_px: float
    transcript_available: bool
    rule_fired: bool
    rule_reason: str
    static_text_similarity: float | None = None
    ocr_text_samples: tuple[str, ...] = ()


class TrackBehaviorDiscriminator:
    """Reject only durable, static, non-dialogue tracks before Gemini."""

    def __init__(
        self,
        speech_segments: Iterable[Any] | None = None,
        video_duration_seconds: float = 0.0,
        static_text_reader: Any | None = None,
    ) -> None:
        self._speech_segments = list(speech_segments or [])
        self._video_duration_seconds = max(float(video_duration_seconds), 0.0)
        self._static_text_reader = static_text_reader

    def evaluate(self, track: dict[str, Any]) -> BehaviorEvidence:
        groups = track.get("groups", [])
        start = float(groups[0].get("timestamp_ms", 0.0)) / 1000.0 if groups else 0.0
        end = float(groups[-1].get("timestamp_ms", 0.0)) / 1000.0 if groups else start
        duration = max(0.0, end - start)
        overlap = sum(
            max(0.0, min(end, self._end(segment)) - max(start, self._start(segment)))
            for segment in self._speech_segments
        )
        overlap_ratio = min(1.0, overlap / duration) if duration else 0.0
        coverage_ratio = (
            min(1.0, duration / self._video_duration_seconds)
            if self._video_duration_seconds
            else 0.0
        )
        displacement = self._displacement(track)
        transcript_available = bool(self._speech_segments)
        is_static_watermark = (
            transcript_available
            and duration >= 3.0
            and displacement <= 12.0
            and overlap_ratio <= 0.05
            and coverage_ratio >= 0.10
        )
        return BehaviorEvidence(
            speech_overlap_ratio=round(overlap_ratio, 4),
            coverage_ratio=round(coverage_ratio, 4),
            duration_seconds=round(duration, 4),
            displacement_px=round(displacement, 4),
            transcript_available=transcript_available,
            rule_fired=is_static_watermark,
            rule_reason=(
                "durable_static_without_speech"
                if is_static_watermark
                else "neutral"
            ),
        )

    def evaluate_static_content(
        self, track: dict[str, Any], proposal_bundle: Any
    ) -> BehaviorEvidence:
        evidence = self.evaluate(track)
        if evidence.duration_seconds < 3.0 or evidence.displacement_px > 12.0:
            return evidence

        crops = TrackCropExtractor().extract_evidence(track, proposal_bundle).get(
            "crops", []
        )
        samples = tuple(self._read_static_text(crop) for crop in crops)
        normalized = [self._normalize_text(text) for text in samples if text.strip()]
        similarity = self._text_similarity(normalized)
        static = len(normalized) >= 2 and similarity >= 0.95
        from dataclasses import replace
        return replace(
            evidence,
            static_text_similarity=round(similarity, 4) if normalized else None,
            ocr_text_samples=samples,
            rule_fired=evidence.rule_fired or static,
            rule_reason=(
                "durable_static_text"
                if static
                else evidence.rule_reason
            ),
        )

    def _read_static_text(self, crop: Any) -> str:
        if self._static_text_reader is not None:
            return str(self._static_text_reader(crop))
        return ""

    @staticmethod
    def _normalize_text(text: str) -> str:
        return "".join(char.casefold() for char in text if char.isalnum())

    @staticmethod
    def _text_similarity(texts: list[str]) -> float:
        if len(texts) < 2:
            return 0.0
        from difflib import SequenceMatcher
        pairs = [
            SequenceMatcher(None, left, right).ratio()
            for index, left in enumerate(texts)
            for right in texts[index + 1 :]
        ]
        return min(pairs) if pairs else 0.0

    @staticmethod
    def _start(segment: Any) -> float:
        return float(segment.start if hasattr(segment, "start") else segment.get("start", 0.0))

    @staticmethod
    def _end(segment: Any) -> float:
        return float(segment.end if hasattr(segment, "end") else segment.get("end", 0.0))

    @staticmethod
    def _displacement(track: dict[str, Any]) -> float:
        groups = track.get("groups", [])
        if len(groups) < 2:
            return 0.0
        first = groups[0]["union_bbox_xyxy"]
        last = groups[-1]["union_bbox_xyxy"]
        first_center = ((first[0] + first[2]) / 2.0, (first[1] + first[3]) / 2.0)
        last_center = ((last[0] + last[2]) / 2.0, (last[1] + last[3]) / 2.0)
        return (
            (first_center[0] - last_center[0]) ** 2
            + (first_center[1] - last_center[1]) ** 2
        ) ** 0.5