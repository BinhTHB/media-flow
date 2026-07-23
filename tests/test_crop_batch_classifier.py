"""Tests for CropBatchClassifier: canary logic, budget ceiling, and batch correctness."""
import json
import re

import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from src.infrastructure.benchmark.hybrid_subtitle.crop_batch_classifier import CropBatchClassifier
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

def _canary_track():
    return _track("N001")


def test_batch_returns_decisions_for_all_tracks():
    """Classifier returns a decision for every track in the batch."""
    client = MagicMock()
    client.generate_content.return_value = FakeGeminiResponse(
        '[{"track_id":"T001","is_subtitle":true,"category":"dialogue_subtitle",'
        '"confidence":0.9,"transcription":"Hello","reason":"bottom subtitle"},'
        '{"track_id":"canary_N001","is_subtitle":false,"category":"logo",'
        '"confidence":0.9,"transcription":"","reason":"negative canary"}]'
    )

    classifier = CropBatchClassifier(client)
    bundle = _make_bundle()
    tracks = [_track()]

    decisions = classifier.classify_batch(
        tracks, bundle, batch_size=5, canary_tracks=[_canary_track()]
    )

    assert "T001" in decisions
    assert decisions["T001"]["is_subtitle"] is True
    assert decisions["T001"]["transcription"] == "Hello"


def test_batch_sends_clean_full_frame_context_with_crops():
    client = MagicMock()
    client.generate_content.return_value = FakeGeminiResponse(
        '[{"track_id":"T001","is_subtitle":true,"category":"dialogue_subtitle",'
        '"confidence":0.9,"transcription":"Hello","reason":"bottom subtitle"},'
        '{"track_id":"canary_N001","is_subtitle":false,"category":"logo",'
        '"confidence":0.9,"transcription":"","reason":"negative canary"}]'
    )
    classifier = CropBatchClassifier(client)

    classifier.classify_batch(
        [_track()], _make_bundle(), canary_tracks=[_canary_track()]
    )

    parts = client.generate_content.call_args.args[0].contents[0]["parts"]
    assert "clean context frame" in parts[0]["text"]
    assert len([part for part in parts[1:] if "inline_data" in part]) == 6


def test_canary_injection_detects_rubber_stamp():
    """Canary track flagged as subtitle triggers retry + half-size."""
    client = MagicMock()
    call_count = [0]

    def fake_generate(req, **kwargs):
        call_count[0] += 1
        # First call: model is untrusted, flags canary as subtitle
        # Subsequent calls: model is correct, flags canary as non-subtitle
        is_sub = True if call_count[0] == 1 else False
        prompt = req.contents[0]["parts"][0]["text"]
        track_ids = re.findall(r"- Candidate (\w+):", prompt)
        decisions = [
            {
                "track_id": track_id,
                "is_subtitle": is_sub if track_id == "canary_N001" else True,
                "category": "logo" if track_id == "canary_N001" else "dialogue_subtitle",
                "confidence": 0.95,
                "transcription": "" if track_id == "canary_N001" else "Hi",
                "reason": "test",
            }
            for track_id in track_ids
        ]
        resp = FakeGeminiResponse(json.dumps(decisions), cache_hit=False)
        return resp

    client.generate_content = fake_generate

    classifier = CropBatchClassifier(client, budget_limit=10)
    bundle = _make_bundle()
    tracks = [_track("T001"), _track("T002")]

    decisions = classifier.classify_batch(
        tracks, bundle, batch_size=5, canary_tracks=[_canary_track()]
    )

    # First call: full batch (T001 + T002 + canary). Canary failed.
    # Retry calls: half-sized batches (size 1). Canary passed because is_sub = False.
    # Total calls: 1 (initial) + 2 (retries for halves) = 3 calls
    assert call_count[0] == 3
    assert "T001" in decisions
    assert "T002" in decisions
    assert decisions["T001"]["is_subtitle"] is True


def test_budget_limit_blocks_requests():
    """Hitting the budget ceiling raises RuntimeError."""
    client = MagicMock()

    def fake_generate(req, **kwargs):
        return FakeGeminiResponse(
            '[{"track_id":"T001","is_subtitle":false,"category":"logo","confidence":0.9},'
            '{"track_id":"canary_N001","is_subtitle":false,"category":"logo","confidence":0.9}]'
        )

    client.generate_content = fake_generate

    classifier = CropBatchClassifier(client, budget_limit=2)
    bundle = _make_bundle()
    tracks = [_track() for _ in range(10)]

    # First 2 batches succeed
    classifier.classify_batch(
        tracks[:2], bundle, batch_size=1, canary_tracks=[_canary_track()]
    )
    assert classifier.requests_sent == 2

    # Third batch raises RuntimeError
    with pytest.raises(RuntimeError, match="Budget limit reached"):
        classifier.classify_batch(
            [tracks[2]], bundle, batch_size=1, canary_tracks=[_canary_track()]
        )


def test_missing_decision_fail_closed():
    """Track with no decision in Gemini response is fail-closed (not included)."""
    client = MagicMock()

    def fake_generate(req, **kwargs):
        return FakeGeminiResponse(
            '[{"track_id":"T002","is_subtitle":true,"category":"dialogue_subtitle",'
            '"confidence":0.9,"transcription":"OK","reason":"ok"},'
            '{"track_id":"canary_N001","is_subtitle":false,"category":"logo",'
            '"confidence":0.9,"transcription":"","reason":"negative canary"}]'
        )

    client.generate_content = fake_generate
    classifier = CropBatchClassifier(client)
    bundle = _make_bundle()

    decisions = classifier.classify_batch(
        [_track("T001")], bundle, batch_size=5, canary_tracks=[_canary_track()]
    )

    # T001 not in decisions -> fail-closed
    assert "T001" not in decisions


def test_empty_tracks_returns_empty():
    """Empty track list returns empty decisions."""
    client = MagicMock()
    classifier = CropBatchClassifier(client)
    bundle = _make_bundle()
    decisions = classifier.classify_batch([], bundle)
    assert decisions == {}
