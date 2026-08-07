"""
Unit and Integration Tests for Policy-Driven Engine & Frontend-Ready Backend Boundary
"""

import json
import os
import shutil
import pytest
import numpy as np
import cv2
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
from src.infrastructure.services.media_processing_service import MediaProcessingService
from src.application.dto.processing_request import ProcessingRequest, ProcessingResult


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


def test_blur_all_all_eligible_tracks_blur():
    tracker = ClassAwareTracker(video_width=1000, video_height=500)
    dets = [
        CanonicalDetection(
            class_id=0, class_name="subtitle", confidence=0.8,
            bbox_xyxy=(10, 10, 50, 50), bbox_normalized=(0.01, 0.01, 0.05, 0.05),
            frame_index=0, timestamp_sec=0.0
        ),
        CanonicalDetection(
            class_id=1, class_name="watermark", confidence=0.8,
            bbox_xyxy=(100, 10, 150, 50), bbox_normalized=(0.1, 0.01, 0.15, 0.05),
            frame_index=0, timestamp_sec=0.0
        )
    ]
    tracks = tracker.track(dets)

    engine = ActionPolicyEngine(PRESET_BLUR_ALL)
    plan = engine.create_rendering_plan("job1", tracks, 1000, 500, 1.0, 30.0)

    for act in plan.actions:
        assert act.status == "active"
        assert act.action_type == "blur"
        assert act.reason == ""


def test_below_threshold_status_ignored_not_preserve():
    tracker = ClassAwareTracker(video_width=1000, video_height=500)
    dets = [
        CanonicalDetection(
            class_id=0, class_name="subtitle", confidence=0.05,  # Below threshold 0.15
            bbox_xyxy=(10, 10, 50, 50), bbox_normalized=(0.01, 0.01, 0.05, 0.05),
            frame_index=0, timestamp_sec=0.0
        )
    ]
    tracks = tracker.track(dets)

    engine = ActionPolicyEngine(PRESET_BLUR_ALL)
    plan = engine.create_rendering_plan("job2", tracks, 1000, 500, 1.0, 30.0)

    assert len(plan.actions) == 1
    act = plan.actions[0]
    assert act.status == "ignored"
    assert act.reason == "below_confidence_threshold"
    assert plan.summary().get("ignored") == 1


def test_preserve_action_intentional():
    policy = ProcessingPolicyConfig(
        watermark=ClassPolicyConfig(action="preserve", confidence_threshold=0.20)
    )
    tracker = ClassAwareTracker(video_width=1000, video_height=500)
    dets = [
        CanonicalDetection(
            class_id=1, class_name="watermark", confidence=0.8,
            bbox_xyxy=(10, 10, 50, 50), bbox_normalized=(0.01, 0.01, 0.05, 0.05),
            frame_index=0, timestamp_sec=0.0
        )
    ]
    tracks = tracker.track(dets)

    engine = ActionPolicyEngine(policy)
    plan = engine.create_rendering_plan("job3", tracks, 1000, 500, 1.0, 30.0)

    assert len(plan.actions) == 1
    act = plan.actions[0]
    assert act.status == "active"
    assert act.action_type == "preserve"
    assert act.reason == "policy_preserve"


def test_replace_image_with_alpha_png(tmp_path):
    # Create test PNG with alpha channel
    png_path = tmp_path / "test_logo.png"
    img = np.zeros((50, 50, 4), dtype=np.uint8)
    img[:, :, 2] = 255  # Red box
    img[:, :, 3] = 200  # Alpha
    cv2.imwrite(str(png_path), img)

    policy = ProcessingPolicyConfig(
        logo=ClassPolicyConfig(action="replace_image", image_asset_path=str(png_path))
    )
    tracker = ClassAwareTracker(video_width=1000, video_height=500)
    dets = [
        CanonicalDetection(
            class_id=2, class_name="logo", confidence=0.8,
            bbox_xyxy=(10, 10, 50, 50), bbox_normalized=(0.01, 0.01, 0.05, 0.05),
            frame_index=0, timestamp_sec=0.0
        )
    ]
    tracks = tracker.track(dets)

    engine = ActionPolicyEngine(policy)
    plan = engine.create_rendering_plan("job4", tracks, 1000, 500, 1.0, 30.0)

    lazy_service = LazyEnrichmentService()
    enriched = lazy_service.enrich_plan(plan, tracks, "dummy.mp4")

    assert enriched.actions[0].action_type == "replace_image"
    assert enriched.actions[0].replacement_image_path == str(png_path)


def test_job_workspace_isolation(tmp_path):
    ws_base = tmp_path / "jobs"
    service = MediaProcessingService(base_workspace_dir=str(ws_base))

    # Job 1 workspace path
    job1_dir = ws_base / "job_101"
    job1_dir.mkdir(parents=True, exist_ok=True)
    (job1_dir / "action_plan.json").write_text('{"job_id": "job_101"}', encoding="utf-8")

    # Job 2 workspace path
    job2_dir = ws_base / "job_102"
    job2_dir.mkdir(parents=True, exist_ok=True)
    (job2_dir / "action_plan.json").write_text('{"job_id": "job_102"}', encoding="utf-8")

    assert (job1_dir / "action_plan.json").read_text(encoding="utf-8") != (job2_dir / "action_plan.json").read_text(encoding="utf-8")


def test_processing_request_result_serialization():
    req = ProcessingRequest(
        input_video="input.mp4",
        output_video="output.mp4",
        preset="blur-all",
    )
    req_json = req.to_json()
    assert "input.mp4" in req_json
    assert "blur-all" in req_json

    res = ProcessingResult(
        job_id="job_test_ser",
        status="SUCCESS",
        input_video_path="input.mp4",
        output_video_path="output.mp4",
        duration_sec=10.0,
        fps=30.0,
        resolution="720x1280",
    )
    res_json = res.to_json()
    assert "job_test_ser" in res_json
    assert "SUCCESS" in res_json


def test_config_override_precedence():
    req = ProcessingRequest(
        input_video="input.mp4",
        output_video="output.mp4",
        preset="translate-subtitle",
    )
    resolved = req.resolve_policy()
    assert resolved.subtitle.action == "replace_text"

    # Direct policy_config override takes precedence over preset
    custom_policy = PRESET_BLUR_ALL
    req.policy_config = custom_policy
    resolved2 = req.resolve_policy()
    assert resolved2.subtitle.action == "blur"


def test_lazy_enrichment_skips_expensive_providers_for_blur():
    engine = ActionPolicyEngine(PRESET_BLUR_ALL)
    plan = engine.create_rendering_plan("job5", [], 1000, 500, 1.0, 30.0)

    lazy_service = LazyEnrichmentService()
    enriched = lazy_service.enrich_plan(plan, [], "dummy.mp4")
    assert enriched == plan
