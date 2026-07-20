import pytest
import numpy as np
import tempfile
import cv2
import os

from src.domain.entities.frame import Frame
from src.domain.entities.segment import Segment
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
            }
        ]

        decisions = {
            "T001": [
                {
                    "candidate_id": "F00000_C01",
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


def test_hybrid_gemini_single_frame_track_survives_normalization():
    from src.application.use_cases.normalize_segments import NormalizeSegmentsUseCase

    with tempfile.TemporaryDirectory() as tmp_dir:
        image_path = os.path.join(tmp_dir, "frame_0.png")
        cv2.imwrite(image_path, np.zeros((720, 1280, 3), dtype=np.uint8))
        frames = [
            Frame(timestamp=1.0, image_path=image_path),
            Frame(timestamp=2.0, image_path=image_path),
        ]
        detections = [{
            "frame_index": 0,
            "timestamp_ms": 1000.0,
            "candidate_id": "F00000_C01",
            "bbox_xyxy": [100.0, 600.0, 300.0, 650.0],
            "confidence": 0.95,
            "class": "text",
        }]
        decisions = {"T001": [{
            "candidate_id": "F00000_C01",
            "is_subtitle": True,
            "category": "dialogue_subtitle",
            "confidence": 0.9,
            "transcription": "Single frame subtitle",
        }]}

        segments = HybridGeminiOCRProvider(
            proposal_provider=FakeProposal(detections),
            semantic_verifier=FakeVerifier(decisions),
        ).detect_text(frames)

        normalized = NormalizeSegmentsUseCase().execute(segments, video_duration=3.0)

        assert len(normalized) == 1
        assert normalized[0].end > normalized[0].start
