import cv2
import numpy as np

from src.application.services.mask_authority import DecisionStore
from src.domain.entities.frame import Frame
from src.infrastructure.providers.subtitle.hybrid_subtitle_region_detector_provider import (
    HybridSubtitleRegionDetectorProvider,
)


class FakeProposalProvider:
    def __init__(self, boxes, confidence=0.9):
        self.boxes = boxes
        self.confidence = confidence

    def detect_images(self, image_paths):
        boxes = self.boxes if isinstance(self.boxes[0], list) else [self.boxes] * len(image_paths)
        return [
            {
                "frame_index": index,
                "timestamp_ms": index * 500.0,
                "candidate_id": f"F{index:05d}_C01",
                "bbox_xyxy": box,
                "confidence": self.confidence,
            }
            for index, box in enumerate(boxes)
        ]


class FakeClassifier:
    def __init__(self, verdict=True):
        self.verdict = verdict
        self.calls = []

    def classify_batch(self, tracks, _bundle):
        self.calls.append(tracks)
        return {
            track["track_id"]: {
                "is_subtitle": self.verdict,
                "model": "fake-gemini",
                "response_ref": "responses/request_01.json",
                "cache_hit": False,
            }
            for track in tracks
        }


def _frames(tmp_path, timestamps=(0.0, 0.5, 1.0)):
    frames = []
    for index, timestamp in enumerate(timestamps):
        path = tmp_path / f"frame_{index}.png"
        cv2.imwrite(str(path), np.zeros((720, 1280, 3), dtype=np.uint8))
        frames.append(Frame(timestamp=timestamp, image_path=str(path)))
    return frames


def _provider(tmp_path, proposal, classifier):
    response = tmp_path / "responses" / "request_01.json"
    response.parent.mkdir(exist_ok=True)
    response.write_text("[]", encoding="utf-8")
    return HybridSubtitleRegionDetectorProvider(
        proposal_provider=proposal,
        crop_batch_classifier=classifier,
        decision_store=DecisionStore("v4-test", path=tmp_path / "mask_decisions.jsonl"),
        static_variance_threshold=-1.0,
    )


def test_detect_regions_empty_frames(tmp_path):
    provider = _provider(tmp_path, FakeProposalProvider([100, 600, 300, 650]), FakeClassifier())
    assert provider.detect_regions([]) == []


def test_v4_positive_track_emits_track_scoped_geometry(tmp_path):
    frames = _frames(tmp_path)
    provider = _provider(tmp_path, FakeProposalProvider([300, 620, 900, 700]), FakeClassifier(True))

    regions = provider.detect_regions(frames)

    assert len(regions) == 3
    assert {region.track_id for region in regions} == {"T001"}
    assert regions[0].x == 300 / 1280
    assert regions[0].y == 620 / 720
    assert all(provider._decision_store.permits_mask(region.track_id) for region in regions)


def test_v4_negative_track_emits_no_geometry(tmp_path):
    frames = _frames(tmp_path)
    provider = _provider(tmp_path, FakeProposalProvider([300, 620, 900, 700]), FakeClassifier(False))

    assert provider.detect_regions(frames) == []


def test_v4_top_band_is_rejected_before_gemini(tmp_path):
    frames = _frames(tmp_path)
    classifier = FakeClassifier(True)
    provider = _provider(tmp_path, FakeProposalProvider([300, 100, 900, 180]), classifier)

    assert provider.detect_regions(frames) == []
    assert classifier.calls == []


def test_v4_missing_decision_fails_closed(tmp_path):
    class MissingDecisionClassifier(FakeClassifier):
        def classify_batch(self, tracks, _bundle):
            return {}

    frames = _frames(tmp_path)
    provider = _provider(
        tmp_path, FakeProposalProvider([300, 620, 900, 700]), MissingDecisionClassifier()
    )

    assert provider.detect_regions(frames) == []


def test_v4_timestamps_pad_half_frame_gap(tmp_path):
    frames = _frames(tmp_path, (0.25, 0.75, 1.25))
    provider = _provider(tmp_path, FakeProposalProvider([300, 620, 900, 700]), FakeClassifier())

    regions = provider.detect_regions(frames)

    assert regions[0].start_time == 0.0
    assert regions[-1].end_time == 1.5


def test_v4_moving_track_keeps_per_frame_geometry(tmp_path):
    frames = _frames(tmp_path, (0.25, 0.75, 1.25))
    boxes = [[100, 600, 300, 650], [200, 600, 400, 650], [300, 600, 500, 650]]
    provider = _provider(tmp_path, FakeProposalProvider(boxes), FakeClassifier())

    regions = provider.detect_regions(frames)

    assert [region.x for region in regions] == [100 / 1280, 200 / 1280, 300 / 1280]


def test_v4_classifier_failure_fails_closed(tmp_path):
    class BrokenClassifier:
        def classify_batch(self, _tracks, _bundle):
            raise RuntimeError("Gemini unavailable")

    frames = _frames(tmp_path)
    provider = _provider(tmp_path, FakeProposalProvider([300, 620, 900, 700]), BrokenClassifier())

    assert provider.detect_regions(frames) == []
