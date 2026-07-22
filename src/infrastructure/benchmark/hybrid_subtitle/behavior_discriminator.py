"""Fail-closed local behavioral rejection for uncertain text tracks."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class BehaviorEvidence:
    speech_overlap_ratio: float
    coverage_ratio: float
    duration_seconds: float
    displacement_px: float
    transcript_available: bool
    rule_fired: bool
    rule_reason: str


class TrackBehaviorDiscriminator:
    """Reject only durable, static, non-dialogue tracks before Gemini."""

    def __init__(
        self,
        speech_segments: Iterable[Any] | None = None,
        video_duration_seconds: float = 0.0,
    ) -> None:
        self._speech_segments = list(speech_segments or [])
        self._video_duration_seconds = max(float(video_duration_seconds), 0.0)

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
