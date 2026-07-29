"""Tests for the bounded V4 track classifier."""

import pytest
import numpy as np
from unittest.mock import MagicMock
from src.infrastructure.benchmark.hybrid_subtitle.crop_batch_classifier import (
    CropBatchClassifier,
    MAX_TRACKS_PER_TABLE,
)
from src.application.dto.proposal_bundle import ProposalBundle


class FakeGeminiResponse:
    def __init__(self, text, model="gemini-2.5-flash", cache_hit=False):
        self.text = text
        self.model = model
        self.cache_hit = cache_hit
        self.finish_reason = "STOP"


def _make_frame():
    img = np.zeros((720, 1280, 3), dtype=np.uint8)
    img[600:650, 100:300] = [0, 255, 0]
    return img


def _make_bundle():
    bundle = ProposalBundle("video.mp4", fps=30.0, stride=1, video_width=1280, video_height=720)
    img = _make_frame()
    bundle.add_frame(
        frame_index=0,
        timestamp_sec=0.0,
        image=img,
        image_path=None,
        detections=[{
            "frame_index": 0,
            "timestamp_ms": 0,
            "candidate_id": "F00000_C01",
            "bbox_xyxy": [100.0, 600.0, 300.0, 650.0],
            "confidence": 0.9,
        }],
    )
    bundle.add_frame(
        frame_index=5,
        timestamp_sec=5/30.0,
        image=img,
        image_path=None,
        detections=[{
            "frame_index": 5,
            "timestamp_ms": 166.7,
            "candidate_id": "F00005_C01",
            "bbox_xyxy": [150.0, 600.0, 350.0, 650.0],
            "confidence": 0.9,
        }],
    )
    return bundle


def _track(track_id="T001"):
    return {
        "track_id": track_id,
        "temporal_union_bbox_xyxy": [100.0, 600.0, 350.0, 650.0],
        "groups": [
            {
                "frame_index": 0,
                "timestamp_ms": 0.0,
                "candidate_group_id": "G00000_01",
                "member_candidate_ids": ["F00000_C01"],
                "union_bbox_xyxy": [100.0, 600.0, 300.0, 650.0],
            },
            {
                "frame_index": 5,
                "timestamp_ms": 166.7,
                "candidate_group_id": "G00005_01",
                "member_candidate_ids": ["F00005_C01"],
                "union_bbox_xyxy": [150.0, 600.0, 350.0, 650.0],
            },
        ],
    }

def test_batch_returns_decisions_for_all_tracks():
    """Classifier returns a decision for every track in the batch."""
    client = MagicMock()
    client.generate_content.return_value = FakeGeminiResponse(
        '[{"track_id":"T001","is_subtitle":true,"category":"dialogue_subtitle",'
        '"confidence":0.9}]'
    )

    classifier = CropBatchClassifier(client)
    bundle = _make_bundle()
    tracks = [_track()]

    decisions = classifier.classify_batch(tracks, bundle)

    assert "T001" in decisions
    assert decisions["T001"]["is_subtitle"] is True
    assert decisions["T001"]["model"] == classifier.model
    assert decisions["T001"]["response_ref"] == "responses/request_01.json"


def test_batch_sends_clean_full_frame_context_with_crops():
    client = MagicMock()
    client.generate_content.return_value = FakeGeminiResponse(
        '[{"track_id":"T001","is_subtitle":true,"category":"dialogue_subtitle",'
        '"confidence":0.9}]'
    )
    classifier = CropBatchClassifier(client)

    classifier.classify_batch([_track()], _make_bundle())

    parts = client.generate_content.call_args.args[0].contents[0]["parts"]
    assert "Classify each track ID as a dialogue subtitle or not." in parts[0]["text"]
    assert "track_id=T001" in parts[0]["text"]
    assert len([part for part in parts[1:] if "inline_data" in part]) == 3


def test_canary_tracks_are_ignored_by_v4_classifier():
    """V4 accepts only real text tracks; canary validation is evaluation-only."""
    client = MagicMock()
    client.generate_content.return_value = FakeGeminiResponse(
        '[{"track_id":"T001","is_subtitle":true,"category":"dialogue_subtitle","confidence":0.95},'
        '{"track_id":"T002","is_subtitle":true,"category":"dialogue_subtitle","confidence":0.95}]'
    )

    classifier = CropBatchClassifier(client)
    bundle = _make_bundle()
    tracks = [_track("T001"), _track("T002")]

    decisions = classifier.classify_batch(
        tracks, bundle, canary_tracks=[_track("canary_N001")]
    )

    assert client.generate_content.call_count == 1
    assert set(decisions) == {"T001", "T002"}
    prompt = client.generate_content.call_args.args[0].contents[0]["parts"][0]["text"]
    assert "canary_N001" not in prompt


def test_budget_limit_blocks_requests():
    """A table count above the V4 request limit raises RuntimeError."""
    client = MagicMock()

    client.generate_content.return_value = FakeGeminiResponse(
        '[{"track_id":"T001","is_subtitle":false,"category":"logo","confidence":0.9}]'
    )
    classifier = CropBatchClassifier(client, budget_limit=1)
    bundle = _make_bundle()
    tracks = [_track() for _ in range(MAX_TRACKS_PER_TABLE + 1)]

    with pytest.raises(RuntimeError, match="Classification request budget exceeded"):
        classifier.classify_batch(tracks, bundle)
    assert classifier.requests_sent == 1


def test_missing_decision_fail_closed():
    """Track with no decision in Gemini response is fail-closed (not included)."""
    client = MagicMock()

    def fake_generate(req, **kwargs):
        return FakeGeminiResponse(
            '[{"track_id":"T002","is_subtitle":true,"category":"dialogue_subtitle",'
            '"confidence":0.9}]'
        )

    client.generate_content = fake_generate
    classifier = CropBatchClassifier(client)
    bundle = _make_bundle()

    decisions = classifier.classify_batch([_track("T001")], bundle)

    # T001 not in decisions -> fail-closed
    assert "T001" not in decisions


def test_empty_tracks_returns_empty():
    """Empty track list returns empty decisions."""
    client = MagicMock()
    classifier = CropBatchClassifier(client)
    bundle = _make_bundle()
    decisions = classifier.classify_batch([], bundle)
    assert decisions == {}
