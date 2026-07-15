from dataclasses import dataclass
from typing import Mapping

from src.application.services.dynamic_ass_event_service import DynamicAssEvent
from src.application.services.dynamic_subtitle_text_layout_service import (
    DynamicSubtitleLayoutResult,
    FIT,
    SKIPPED as LAYOUT_SKIPPED,
)
from src.application.services.dynamic_subtitle_timeline_service import (
    TimelineMappingResult,
)

@dataclass(frozen=True)
class DynamicReplacementDecision:
    interval_id: str
    start: float
    end: float
    status: str
    reason: str
    mapping_status: str
    layout_status: str
    source_bbox: object | None
    cover_bbox: object | None
    target_text: str
    ass_event: DynamicAssEvent | None
    safety_status: str
    safety_reason: str | None
    candidate_level_override: bool
    override_reason: str | None
    metadata: Mapping[str, object]

STATUS_REPLACED = 'REPLACED'
STATUS_SKIPPED = 'SKIPPED'
STATUS_REVIEW = 'REVIEW_REQUIRED'
STATUS_FALLBACK = 'STATIC_FALLBACK'

HARD_SAFETY_REASONS = {
    'true_person_overlap_risk',
    'large_person_overlap',
    'person_overlap_only',
    'face_overlap',
    'watermark_overlap',
    'scene_object_overlap',
}

class DynamicSubtitleReplacementService:
    def decide(
        self,
        mapping: TimelineMappingResult,
        layout: DynamicSubtitleLayoutResult,
        event: DynamicAssEvent | None,
        *,
        force_static_fallback: bool = False,
    ) -> DynamicReplacementDecision:
        # Resolve overrides
        meta = mapping.metadata or {}
        candidate_level_override = bool(
            meta.get('candidate_level_override', False)
            or meta.get('static_region_person_overlap_override', False)
            or meta.get('local_safety_override', False)
        )
        override_reason = meta.get('override_reason')
        if not isinstance(override_reason, str) or not override_reason:
            override_reason = None

        # 1. Skipped due to safety risk (Fail-Closed)
        is_layout_skipped = layout.layout_status == LAYOUT_SKIPPED
        safety_reason = mapping.safety_reason or (
            layout.layout_reason if is_layout_skipped else None
        )
        if (
            mapping.safety_status == 'skipped'
            or is_layout_skipped
            or (safety_reason and safety_reason in HARD_SAFETY_REASONS)
        ):
            final_reason = safety_reason or 'true_person_overlap_risk'
            return self._build(
                mapping, layout, None, STATUS_SKIPPED, 'hard_safety_reject',
                safety_reason=final_reason,
                candidate_level_override=candidate_level_override,
                override_reason=override_reason,
            )

        # 2. Static Fallback requested
        if force_static_fallback:
            if mapping.start >= 0 and mapping.end > mapping.start:
                return self._build(
                    mapping, layout, None, STATUS_FALLBACK, 'static_fallback_requested',
                    candidate_level_override=candidate_level_override,
                    override_reason=override_reason,
                )
            return self._build(
                mapping, layout, None, STATUS_REVIEW, 'invalid_interval',
                candidate_level_override=candidate_level_override,
                override_reason=override_reason,
            )

        # 3. Validation / Review Required checks
        if mapping.mapping_status != 'mapped':
            return self._build(
                mapping, layout, None, STATUS_REVIEW, 'mapping_not_eligible',
                candidate_level_override=candidate_level_override,
                override_reason=override_reason,
            )
        if mapping.quality_result and not mapping.quality_result.accepted:
            return self._build(
                mapping, layout, None, STATUS_REVIEW, 'translation_quality_failed',
                candidate_level_override=candidate_level_override,
                override_reason=override_reason,
            )
        if layout.layout_status != FIT:
            return self._build(
                mapping, layout, None, STATUS_REVIEW, 'layout_not_fit',
                candidate_level_override=candidate_level_override,
                override_reason=override_reason,
            )
        if not mapping.source_bbox:
            return self._build(
                mapping, layout, None, STATUS_REVIEW, 'missing_source_bbox',
                candidate_level_override=candidate_level_override,
                override_reason=override_reason,
            )
        if not mapping.expanded_cover_bbox:
            return self._build(
                mapping, layout, None, STATUS_REVIEW, 'missing_cover_bbox',
                candidate_level_override=candidate_level_override,
                override_reason=override_reason,
            )
        if event is None:
            return self._build(
                mapping, layout, None, STATUS_REVIEW, 'missing_ass_event',
                candidate_level_override=candidate_level_override,
                override_reason=override_reason,
            )

        # 4. Success - REPLACED
        return self._build(
            mapping, layout, event, STATUS_REPLACED, 'replacement_ready',
            candidate_level_override=candidate_level_override,
            override_reason=override_reason,
        )

    def _build(
        self,
        mapping: TimelineMappingResult,
        layout: DynamicSubtitleLayoutResult,
        event: DynamicAssEvent | None,
        status: str,
        reason: str,
        safety_reason: str | None = None,
        candidate_level_override: bool = False,
        override_reason: str | None = None,
    ) -> DynamicReplacementDecision:
        return DynamicReplacementDecision(
            interval_id=mapping.interval_id,
            start=mapping.start,
            end=mapping.end,
            status=status,
            reason=reason,
            mapping_status=mapping.mapping_status,
            layout_status=layout.layout_status,
            source_bbox=mapping.source_bbox,
            cover_bbox=mapping.expanded_cover_bbox,
            target_text=mapping.target_text,
            ass_event=event,
            safety_status=mapping.safety_status,
            safety_reason=safety_reason or mapping.safety_reason,
            candidate_level_override=candidate_level_override,
            override_reason=override_reason,
            metadata=mapping.metadata or {},
        )