import pytest
import numpy as np
import tempfile
import cv2
import os

from src.domain.entities.frame import Frame
from src.domain.entities.segment import Segment
from src.domain.entities.subtitle_region import SubtitleRegion
from src.infrastructure.providers.ocr.hybrid_gemini_ocr_provider import HybridGeminiOCRProvider

class FakeProposal:
    def __init__(self, detections):
        self._dets = detections

    def detect_images(self, image_paths):
        return list(self._dets)

class FakeVerifier:
    def __init__(self, decisions_by_track):
        self._decisions = decisions_by_track

    def classify_track(self, track, rep_frames, rep_ids, cap=None):
        return {
            "track_id": track.track_id,
            "decisions": self._decisions.get(track.track_id, []),
            "uncertain": False,
        }

class CountingVerifier(FakeVerifier):
    def __init__(self, decisions_by_track):
        super().__init__(decisions_by_track)
        self.calls = 0

    def classify_track(self, track, rep_frames, rep_ids, cap=None):
        self.calls += 1
        return super().classify_track(track, rep_frames, rep_ids, cap)

def test_hybrid_gemini_ocr_emits_segments_for_transcribed_subtitles():
    with tempfile.TemporaryDirectory() as tmp_dir:
        img_path = os.path.join(tmp_dir, "frame_0.png")
        img = np.zeros((720, 1280, 3), dtype=np.uint8)
        cv2.imwrite(img_path, img)
        frames = [Frame(timestamp=1.0, image_path=img_path), Frame(timestamp=2.0, image_path=img_path)]

        dets = [
            {
                "frame_index": 0,
                "timestamp_ms": 1000.0,
                "candidate_id": "F00000_C01",
                "bbox_xyxy": [100.0, 600.0, 300.0, 650.0],
                "confidence": 0.95,
                "class": "text",
            },
            {
                "frame_index": 1,
                "timestamp_ms": 2000.0,
                "candidate_id": "F00001_C01",
                "bbox_xyxy": [100.0, 600.0, 300.0, 650.0],
                "confidence": 0.95,
                "class": "text",
            },
        ]

        decisions = {
            "T001": [
                {
                    "candidate_id": "G00000_01",
                    "is_subtitle": True,
                    "category": "dialogue_subtitle",
                    "confidence": 0.9,
                    "transcription": "Hello World Subtitle",
                },
                {
                    "candidate_id": "G00001_01",
                    "is_subtitle": True,
                    "category": "dialogue_subtitle",
                    "confidence": 0.9,
                    "transcription": "Hello World Subtitle",
                }
            ]
        }

        provider = HybridGeminiOCRProvider(
            proposal_provider=FakeProposal(dets),
            semantic_verifier=FakeVerifier(decisions),
        )

        segments = provider.detect_text(frames)
        assert len(segments) == 1
        seg = segments[0]
        assert isinstance(seg, Segment)
        assert seg.text == "Hello World Subtitle"
        assert seg.source == "ocr"
        assert seg.confidence == 0.9
        assert seg.start <= 1.0


def test_hybrid_gemini_ocr_track_survives_normalization():
    from src.application.use_cases.normalize_segments import NormalizeSegmentsUseCase

    with tempfile.TemporaryDirectory() as tmp_dir:
        image_path = os.path.join(tmp_dir, "frame_0.png")
        cv2.imwrite(image_path, np.zeros((720, 1280, 3), dtype=np.uint8))
        frames = [
            Frame(timestamp=1.0, image_path=image_path),
            Frame(timestamp=2.0, image_path=image_path),
        ]
        detections = [
            {
                "frame_index": 0,
                "timestamp_ms": 1000.0,
                "candidate_id": "F00000_C01",
                "bbox_xyxy": [100.0, 600.0, 300.0, 650.0],
                "confidence": 0.95,
                "class": "text",
            },
            {
                "frame_index": 1,
                "timestamp_ms": 2000.0,
                "candidate_id": "F00001_C01",
                "bbox_xyxy": [100.0, 600.0, 300.0, 650.0],
                "confidence": 0.95,
                "class": "text",
            },
        ]
        decisions = {"T001": [{
            "candidate_id": "G00000_01",
            "is_subtitle": True,
            "category": "dialogue_subtitle",
            "confidence": 0.9,
            "transcription": "Track subtitle",
        }, {
            "candidate_id": "G00001_01",
            "is_subtitle": True,
            "category": "dialogue_subtitle",
            "confidence": 0.9,
            "transcription": "Track subtitle",
        }]}

        segments = HybridGeminiOCRProvider(
            proposal_provider=FakeProposal(detections),
            semantic_verifier=FakeVerifier(decisions),
        ).detect_text(frames)

        normalized = NormalizeSegmentsUseCase().execute(segments, video_duration=3.0)

        assert len(normalized) == 1
        assert normalized[0].end > normalized[0].start

def test_heuristic_negative_track_skips_gemini_verifier(tmp_path):
    image_path = tmp_path / "frame.png"
    assert cv2.imwrite(str(image_path), np.zeros((720, 1280, 3), dtype=np.uint8))
    frames = [
        Frame(timestamp=0.0, image_path=str(image_path)),
        Frame(timestamp=0.5, image_path=str(image_path)),
    ]
    detections = [
        {
            "frame_index": index,
            "timestamp_ms": index * 500,
            "candidate_id": f"F{index:05d}_C01",
            "bbox_xyxy": [100.0, 50.0, 300.0, 100.0],
            "confidence": 0.95,
            "class": "text",
        }
        for index in range(2)
    ]
    verifier = CountingVerifier({})

    segments = HybridGeminiOCRProvider(
        proposal_provider=FakeProposal(detections),
        semantic_verifier=verifier,
    ).detect_text(frames)

    assert segments == []
    assert verifier.calls == 0


def test_short_bottom_track_routes_to_semantic_verifier(tmp_path):
    image_path = tmp_path / "frame.png"
    assert cv2.imwrite(str(image_path), np.zeros((720, 1280, 3), dtype=np.uint8))
    frames = [
        Frame(timestamp=0.0, image_path=str(image_path)),
        Frame(timestamp=0.1, image_path=str(image_path)),
    ]
    detections = [
        {
            "frame_index": index,
            "timestamp_ms": index * 100,
            "candidate_id": f"F{index:05d}_C01",
            "bbox_xyxy": [100.0, 650.0, 300.0, 710.0],
            "confidence": 0.95,
            "class": "text",
        }
        for index in range(2)
    ]
    verifier = CountingVerifier({
        "T001": [{
            "candidate_id": "G00000_01",
            "is_subtitle": True,
            "category": "dialogue_subtitle",
            "confidence": 0.9,
            "transcription": "Short caption",
        }],
    })

    segments = HybridGeminiOCRProvider(
        proposal_provider=FakeProposal(detections),
        semantic_verifier=verifier,
    ).detect_text(frames)

    assert verifier.calls == 1
    assert len(segments) == 1


def test_v4_confirmed_tracks_transcribed_in_one_request(tmp_path):
    img_path = tmp_path / "frame.png"
    cv2.imwrite(str(img_path), np.zeros((720, 1280, 3), dtype=np.uint8))
    frames = [Frame(timestamp=1.0, image_path=str(img_path))]
    regions = [
        SubtitleRegion(
            start_time=1.0, end_time=2.0, x=0.1, y=0.8, width=0.5, height=0.1,
            confidence=0.9, track_id="T003"
        ),
        SubtitleRegion(
            start_time=1.0, end_time=2.0, x=0.2, y=0.8, width=0.4, height=0.1,
            confidence=0.9, track_id="T006"
        )
    ]

    class MockClientResponse:
        def __init__(self, text):
            self.text = text

    class MockClient:
        def __init__(self):
            self.requests = []

        def generate_content(self, req, use_cache=False):
            self.requests.append(req)
            return MockClientResponse('[{"track_id":"T003","text":"transcription 3"},{"track_id":"T006","text":"transcription 6"}]')

    class FakeVerifierWithClient(FakeVerifier):
        def __init__(self):
            super().__init__({})
            self.client = MockClient()
            self._model = "gemini-3.5-flash-lite"

        def _resolve_client(self):
            return self.client

    verifier = FakeVerifierWithClient()
    provider = HybridGeminiOCRProvider(
        proposal_provider=None,
        semantic_verifier=verifier
    )

    segments = provider.read_confirmed_tracks(regions, frames, 1280, 720)

    assert len(verifier.client.requests) == 1
    req = verifier.client.requests[0]
    assert req.request_id == "confirmed_track_ocr_batch"
    assert req.source_kind == "confirmed_track_ocr_v4"

    assert len(segments) == 2
    assert segments[0].id == "ocr_T003"
    assert segments[0].text == "transcription 3"
    assert segments[1].id == "ocr_T006"
    assert segments[1].text == "transcription 6"