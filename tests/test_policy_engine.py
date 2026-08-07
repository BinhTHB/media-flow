"""
Unit and Integration Tests for Policy-Driven Engine
"""

import os
import pytest
from pathlib import Path

from src.domain.entities.canonical_detection import CanonicalDetection, ClassAwareTrack
from src.domain.entities.processing_policy import (
    ActionType,
    ClassPolicyConfig,
    ProcessingPolicyConfig,
    PRESET_BLUR_ALL,
    PRESET_TRANSLATE_SUBTITLE,
    PRESET_PRESERVE_ALL,
    get_preset_policy,
)
from src.domain.services.class_aware_tracker import ClassAwareTracker
from src.application.services.action_policy_engine import ActionPolicyEngine
from src.infrastructure.services.lazy_enrichment_service import LazyEnrichmentService
from src.application.dto.processing_request import ProcessingRequest


def test_processing_policy_defaults_and_presets():
    policy = PRESET_BLUR_ALL
    assert policy.subtitle.action == "blur"
    assert policy.watermark.action == "blur"
    assert policy.logo.action == "blur"
    assert policy.other_burn_in_text.action == "blur"

    preset_trans = get_preset_policy("translate-subtitle")
    assert preset_trans.subtitle.action == "replace_text"
    assert preset_trans.watermark.action == "preserve"

    with pytest.raises(ValueError, match="Invalid ActionType"):
        ActionType.from_str("invalid_action")

    with pytest.raises(ValueError, match="Unknown preset"):
        get_preset_policy("invalid-preset-name")


def test_class_aware_tracker_preserves_class_metadata():
    dets = [
        CanonicalDetection(
            class_id=0,
            class_name="subtitle",
            confidence=0.9,
            bbox_xyxy=(10.0, 10.0, 100.0, 50.0),
            bbox_normalized=(0.1, 0.1, 0.5, 0.2),
            frame_index=1,
            timestamp_sec=0.033,
        ),
        CanonicalDetection(
            class_id=1,
            class_name="watermark",
            confidence=0.85,
            bbox_xyxy=(200.0, 10.0, 300.0, 50.0),
            bbox_normalized=(0.6, 0.1, 0.9, 0.2),
            frame_index=1,
            timestamp_sec=0.033,
        ),
    ]

    tracker = ClassAwareTracker(video_width=1000, video_height=500)
    tracks = tracker.track(dets)

    assert len(tracks) == 2
    class_names = {t.class_name for t in tracks}
    assert class_names == {"subtitle", "watermark"}


def test_action_policy_engine_planning():
    tracker = ClassAwareTracker(video_width=1000, video_height=500)
    dets = [
        CanonicalDetection(
            class_id=2,
            class_name="logo",
            confidence=0.8,
            bbox_xyxy=(10.0, 10.0, 50.0, 50.0),
            bbox_normalized=(0.01, 0.01, 0.05, 0.05),
            frame_index=0,
            timestamp_sec=0.0,
        )
    ]
    tracks = tracker.track(dets)

    engine = ActionPolicyEngine(PRESET_BLUR_ALL)
    plan = engine.create_rendering_plan(
        job_id="test_job",
        tracks=tracks,
        video_width=1000,
        video_height=500,
        duration=1.0,
        fps=30.0,
    )

    assert len(plan.actions) == 1
    action = plan.actions[0]
    assert action.class_name == "logo"
    assert action.action_type == "blur"


def test_lazy_enrichment_skips_expensive_providers_for_blur():
    engine = ActionPolicyEngine(PRESET_BLUR_ALL)
    plan = engine.create_rendering_plan(
        job_id="test_job",
        tracks=[],
        video_width=1000,
        video_height=500,
        duration=1.0,
        fps=30.0,
    )

    lazy_service = LazyEnrichmentService()
    # Should complete without raising credential errors even if GEMINI_API_KEY is missing
    enriched = lazy_service.enrich_plan(
        rendering_plan=plan,
        tracks=[],
        input_video_path="dummy.mp4",
    )
    assert enriched == plan
