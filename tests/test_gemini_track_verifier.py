# Unified Gemini verifier tests
import pytest
import numpy as np
import json

from src.infrastructure.benchmark.hybrid_subtitle.gemini_verifier import (
    GeminiTrackVerifier,
    GeminiSubtitleSemanticVerifier,
    LiveGeminiTrackVerifier,
)

class FakeGeminiResponse:
    def __init__(self, text: str, model: str = "gemini-2.5-flash", cache_hit: bool = False):
        self.text = text
        self.model = model
        self.cache_hit = cache_hit
        self.finish_reason = "STOP"

class FakeRuntimeClient:
    def __init__(self, responses: list):
        self._responses = responses
        self._index = 0

    def generate_content(self, req, use_cache: bool = True):
        if self._index < len(self._responses):
            resp = self._responses[self._index]
            self._index += 1
            return resp
        return FakeGeminiResponse("{}")

class MockTrackObj:
    def __init__(self, track_id: str):
        self.track_id = track_id

def test_classify_track_returns_decisions():
    fake_resp = FakeGeminiResponse(json.dumps({
        "track_id": "T001",
        "decisions": [
            {
                "candidate_id": "F00000_C01",
                "is_subtitle": True,
                "category": "dialogue_subtitle",
                "confidence": 0.9,
                "transcription": None,
                "reason_code": "bottom_position"
            }
        ],
        "uncertain": False
    }))
    client = FakeRuntimeClient([fake_resp])
    verifier = GeminiTrackVerifier(client, "gemini-2.5-flash")

    mock_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    result = verifier.classify_track(
        MockTrackObj("T001"),
        frame_bgr=[mock_frame],
        candidate_ids=["F00000_C01"],
        candidate_bboxes=[[100, 600, 300, 650]],
    )
    assert result["track_id"] == "T001"
    assert len(result["decisions"]) == 1
    assert result["decisions"][0]["candidate_id"] == "F00000_C01"
    assert result["decisions"][0]["is_subtitle"] is True

def test_classify_returns_decisions():
    fake_resp = FakeGeminiResponse(json.dumps({
        "track_id": "T001",
        "decisions": [
            {
                "candidate_id": "F00000_C01",
                "is_subtitle": True,
                "category": "dialogue_subtitle",
                "confidence": 0.9,
                "transcription": None,
                "reason_code": "bottom_position"
            }
        ],
        "uncertain": False
    }))
    client = FakeRuntimeClient([fake_resp])
    verifier = GeminiTrackVerifier(client, "gemini-2.5-flash")

    track_dict = {
        "track_id": "T001",
        "groups": [{"frame_index": 0, "candidate_group_id": "F00000_C01"}],
        "start_frame": 0,
        "end_frame": 4,
    }

    class FakeVideoCapture:
        def set(self, prop, value):
            pass
        def read(self):
            frame = np.zeros((720, 1280, 3), dtype=np.uint8)
            return True, frame

    result = verifier.classify(
        track_dict,
        rep_frames=[(0, [100, 600, 300, 650], "first")],
        candidate_ids=["F00000_C01"],
        cap=FakeVideoCapture(),
    )
    assert result["track_id"] == "T001"
    assert len(result["decisions"]) == 1
    assert result["decisions"][0]["candidate_id"] == "F00000_C01"
    assert result["decisions"][0]["is_subtitle"] is True

def test_bbox_key_rejected_in_response():
    fake_resp = FakeGeminiResponse(json.dumps({
        "track_id": "T001",
        "decisions": [
            {
                "candidate_id": "F00000_C01",
                "is_subtitle": True,
                "category": "dialogue_subtitle",
                "confidence": 0.9,
                "transcription": None,
                "reason_code": "bottom_position",
                "bbox": [100, 600, 300, 650]
            }
        ],
        "uncertain": False
    }))
    client = FakeRuntimeClient([fake_resp])
    verifier = GeminiTrackVerifier(client, "gemini-2.5-flash")

    track_dict = {
        "track_id": "T001",
        "groups": [{"frame_index": 0, "candidate_group_id": "F00000_C01"}],
        "start_frame": 0,
        "end_frame": 4,
    }

    class FakeVideoCapture:
        def set(self, prop, value):
            pass
        def read(self):
            frame = np.zeros((720, 1280, 3), dtype=np.uint8)
            return True, frame

    result = verifier.classify(
        track_dict,
        rep_frames=[(0, [100, 600, 300, 650], "first")],
        candidate_ids=["F00000_C01"],
        cap=FakeVideoCapture(),
    )
    assert result["decisions"] == []

def test_aliases_point_to_same_class():
    assert GeminiSubtitleSemanticVerifier is GeminiTrackVerifier
    assert LiveGeminiTrackVerifier is GeminiTrackVerifier
