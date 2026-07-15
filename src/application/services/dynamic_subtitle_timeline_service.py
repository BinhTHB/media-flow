from dataclasses import dataclass, field
from typing import Mapping, Sequence

from src.application.services.translation_quality_gate import (
    TranslationQualityGate,
    TranslationQualityResult,
)

BBox = Mapping[str, float]


@dataclass(frozen=True)
class DynamicSourceInterval:
    interval_id: str
    start: float
    end: float
    source_text: str
    source_bbox: BBox | None
    expanded_cover_bbox: BBox | None
    safety_status: str
    safety_reason: str | None
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class TranslatedSourceSegment:
    segment_id: str
    start: float
    end: float
    source_text: str
    target_text: str
    provenance: str
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class TimelineMappingResult:
    interval_id: str
    start: float
    end: float
    source_text: str
    target_text: str
    source_bbox: BBox | None
    expanded_cover_bbox: BBox | None
    mapping_status: str
    mapping_reason: str
    matched_segment_id: str | None
    intersection_duration: float
    interval_coverage: float
    segment_coverage: float
    temporal_iou: float
    overlap_score: float
    safety_status: str
    safety_reason: str | None
    quality_result: TranslationQualityResult | None
    metadata: Mapping[str, object]


@dataclass(frozen=True)
class _Candidate:
    segment: TranslatedSourceSegment
    quality: TranslationQualityResult
    intersection_duration: float
    interval_coverage: float
    segment_coverage: float
    temporal_iou: float
    score: float


class DynamicSubtitleTimelineService:
    def __init__(
        self,
        *,
        quality_gate: TranslationQualityGate | None = None,
        minimum_interval_coverage: float = 0.50,
        minimum_segment_coverage: float = 0.10,
        ambiguity_margin: float = 0.10,
    ) -> None:
        for name, value in (
            ("minimum_interval_coverage", minimum_interval_coverage),
            ("minimum_segment_coverage", minimum_segment_coverage),
            ("ambiguity_margin", ambiguity_margin),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1.")
        self.quality_gate = quality_gate or TranslationQualityGate()
        self.minimum_interval_coverage = minimum_interval_coverage
        self.minimum_segment_coverage = minimum_segment_coverage
        self.ambiguity_margin = ambiguity_margin

    def map(
        self,
        intervals: Sequence[DynamicSourceInterval],
        segments: Sequence[TranslatedSourceSegment],
    ) -> list[TimelineMappingResult]:
        ordered_intervals = sorted(
            intervals, key=lambda item: (item.start, item.end, item.interval_id)
        )
        ordered_segments = sorted(
            segments, key=lambda item: (item.start, item.end, item.segment_id)
        )
        return [
            self._map_interval(interval, ordered_segments)
            for interval in ordered_intervals
        ]

    def _map_interval(
        self,
        interval: DynamicSourceInterval,
        segments: Sequence[TranslatedSourceSegment],
    ) -> TimelineMappingResult:
        if interval.safety_reason:
            return self._review(interval, interval.safety_reason)
        if not self._valid_bbox(interval.source_bbox):
            return self._review(interval, "missing_source_bbox")
        if not interval.source_text.strip():
            return self._review(interval, "missing_source_text")

        candidates = []
        rejected_quality = False
        rejected_provenance = False
        for segment in segments:
            if segment.provenance != "ocr_source_text":
                if self._intersection(interval, segment) > 0:
                    rejected_provenance = True
                continue
            intersection = self._intersection(interval, segment)
            if intersection <= 0:
                continue
            quality = self.quality_gate.evaluate(
                source_text=segment.source_text,
                target_text=segment.target_text,
            )
            if not quality.accepted:
                rejected_quality = True
                continue
            interval_duration = interval.end - interval.start
            segment_duration = segment.end - segment.start
            if interval_duration <= 0 or segment_duration <= 0:
                continue
            interval_coverage = intersection / interval_duration
            segment_coverage = intersection / segment_duration
            union = interval_duration + segment_duration - intersection
            temporal_iou = intersection / union if union > 0 else 0.0
            if (
                interval_coverage < self.minimum_interval_coverage
                or segment_coverage < self.minimum_segment_coverage
            ):
                continue
            score = (interval_coverage + segment_coverage + temporal_iou) / 3
            candidates.append(
                _Candidate(
                    segment=segment,
                    quality=quality,
                    intersection_duration=intersection,
                    interval_coverage=interval_coverage,
                    segment_coverage=segment_coverage,
                    temporal_iou=temporal_iou,
                    score=score,
                )
            )

        candidates.sort(
            key=lambda item: (-item.score, item.segment.start, item.segment.segment_id)
        )
        if not candidates:
            if rejected_quality:
                return self._review(interval, "translation_quality_rejected")
            if rejected_provenance:
                return self._review(interval, "unsupported_source_provenance")
            return self._review(
                interval, "missing_or_ambiguous_timeline_match"
            )
        if (
            len(candidates) > 1
            and candidates[0].score - candidates[1].score < self.ambiguity_margin
        ):
            return self._review(
                interval, "missing_or_ambiguous_timeline_match"
            )
        best = candidates[0]
        return TimelineMappingResult(
            interval_id=interval.interval_id,
            start=interval.start,
            end=interval.end,
            source_text=interval.source_text,
            target_text=best.segment.target_text,
            source_bbox=interval.source_bbox,
            expanded_cover_bbox=interval.expanded_cover_bbox,
            mapping_status="mapped",
            mapping_reason="timestamp_overlap",
            matched_segment_id=best.segment.segment_id,
            intersection_duration=best.intersection_duration,
            interval_coverage=best.interval_coverage,
            segment_coverage=best.segment_coverage,
            temporal_iou=best.temporal_iou,
            overlap_score=best.score,
            safety_status=interval.safety_status,
            safety_reason=interval.safety_reason,
            quality_result=best.quality,
            metadata=interval.metadata,
        )

    @staticmethod
    def _intersection(
        interval: DynamicSourceInterval, segment: TranslatedSourceSegment
    ) -> float:
        return max(
            0.0, min(interval.end, segment.end) - max(interval.start, segment.start)
        )

    @staticmethod
    def _valid_bbox(bbox: BBox | None) -> bool:
        return bool(
            bbox
            and bbox.get("left", 0) < bbox.get("right", 0)
            and bbox.get("top", 0) < bbox.get("bottom", 0)
        )

    @staticmethod
    def _review(
        interval: DynamicSourceInterval, reason: str
    ) -> TimelineMappingResult:
        return TimelineMappingResult(
            interval_id=interval.interval_id,
            start=interval.start,
            end=interval.end,
            source_text=interval.source_text,
            target_text="",
            source_bbox=interval.source_bbox,
            expanded_cover_bbox=interval.expanded_cover_bbox,
            mapping_status="review_required",
            mapping_reason=reason,
            matched_segment_id=None,
            intersection_duration=0.0,
            interval_coverage=0.0,
            segment_coverage=0.0,
            temporal_iou=0.0,
            overlap_score=0.0,
            safety_status=interval.safety_status,
            safety_reason=interval.safety_reason,
            quality_result=None,
            metadata=interval.metadata,
        )
