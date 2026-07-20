import os
import pytest
import cv2
import numpy as np
import tempfile

from src.domain.entities.frame import Frame
from src.infrastructure.providers.subtitle.hybrid_subtitle_region_detector_provider import (
    HybridSubtitleRegionDetectorProvider,
)
from src.domain.entities.subtitle_region import SubtitleRegion

from src.infrastructure.registry.provider_registry import ProviderRegistry

requires_yolo_weights = pytest.mark.skipif(
    not os.path.exists("best.pt"),
    reason="requires local YOLO weights best.pt",
)

class FakeProposalProvider:
    def __init__(self, bbox_xyxy, confidence=0.9):
        self._bbox = bbox_xyxy
        self._confidence = confidence

    def detect_images(self, image_paths):
        dets = []
        for fi, p in enumerate(image_paths):
            dets.append({
                "frame_index": fi,
                "timestamp_ms": round(fi * 1000.0, 1),
                "candidate_id": f"F{fi:05d}_C01",
                "bbox_xyxy": self._bbox,
                "bbox_normalized": [
                    self._bbox[0]/1280, self._bbox[1]/720,
                    self._bbox[2]/1280, self._bbox[3]/720
                ],
                "confidence": self._confidence,
                "class": "text",
            })
        return dets

def test_detect_regions_empty_frames():
    provider = HybridSubtitleRegionDetectorProvider(proposal_provider=FakeProposalProvider([100, 100, 200, 200]))
    assert provider.detect_regions([]) == []

def test_detect_regions_returns_normalized_bottom_region():
    with tempfile.TemporaryDirectory() as tmp_dir:
        paths = []
        for i in range(3):
            img_path = os.path.join(tmp_dir, f"frame_{i}.png")
            img = np.zeros((720, 1280, 3), dtype=np.uint8)
            cv2.imwrite(img_path, img)
            paths.append(img_path)

        frames = [
            Frame(timestamp=0.0, image_path=paths[0]),
            Frame(timestamp=0.5, image_path=paths[1]),
            Frame(timestamp=1.0, image_path=paths[2]),
        ]

        fake_proposal = FakeProposalProvider([300.0, 620.0, 900.0, 700.0], 0.9)
        provider = HybridSubtitleRegionDetectorProvider(proposal_provider=fake_proposal)

        regions = provider.detect_regions(frames)
        assert len(regions) >= 1
        r = regions[0]
        assert 0.0 <= r.x <= 1.0
        assert 0.0 <= r.y <= 1.0
        assert r.y >= 0.5
        assert r.end_time >= r.start_time
        assert r.confidence == 0.9

def test_scene_text_top_region_filtered():
    with tempfile.TemporaryDirectory() as tmp_dir:
        paths = []
        for i in range(3):
            img_path = os.path.join(tmp_dir, f"frame_{i}.png")
            img = np.zeros((720, 1280, 3), dtype=np.uint8)
            cv2.imwrite(img_path, img)
            paths.append(img_path)

        frames = [
            Frame(timestamp=0.0, image_path=paths[0]),
            Frame(timestamp=0.5, image_path=paths[1]),
            Frame(timestamp=1.0, image_path=paths[2]),
        ]

        fake_proposal = FakeProposalProvider([300.0, 100.0, 900.0, 180.0], 0.9)
        provider = HybridSubtitleRegionDetectorProvider(proposal_provider=fake_proposal)

        regions = provider.detect_regions(frames)
        assert regions == []

def test_registry_creates_hybrid_provider():
    registry = ProviderRegistry()
    provider = registry.create("subtitle_region_detector", "hybrid_yolo", {})
    assert provider.__class__.__name__ == "HybridSubtitleRegionDetectorProvider"

@requires_yolo_weights
def test_detect_regions_with_real_yolo_weights(tmp_path):
    frames = []
    for i in range(5):
        image_path = tmp_path / f"frame_{i}.png"
        image = np.zeros((720, 1280, 3), dtype=np.uint8)
        cv2.putText(
            image, "SUBTITLE TEXT", (300, 680),
            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2,
        )
        cv2.imwrite(str(image_path), image)
        frames.append(Frame(timestamp=i * 0.5, image_path=str(image_path)))

    regions = HybridSubtitleRegionDetectorProvider().detect_regions(frames)
    assert all(region.__class__.__name__ == "SubtitleRegion" for region in regions)

@requires_yolo_weights
def test_detect_regions_real_yolo_guarded():
    with tempfile.TemporaryDirectory() as tmp_dir:
        paths = []
        for i in range(5):
            img_path = os.path.join(tmp_dir, f"frame_{i}.png")
            img = np.zeros((720, 1280, 3), dtype=np.uint8)
            cv2.putText(img, "SUBTITLE TEST", (300, 680), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            cv2.imwrite(img_path, img)
            paths.append(img_path)

        frames = [
            Frame(timestamp=i * 0.5, image_path=paths[i])
            for i in range(5)
        ]

        provider = HybridSubtitleRegionDetectorProvider()
        regions = provider.detect_regions(frames)
        assert isinstance(regions, list)
        for r in regions:
            assert isinstance(r, SubtitleRegion)
            assert 0.0 <= r.x <= 1.0
            assert 0.0 <= r.y <= 1.0
            assert 0.0 <= r.width <= 1.0
            assert 0.0 <= r.height <= 1.0
            assert r.end_time >= r.start_time
            assert 0.0 <= r.confidence <= 1.0

def test_detect_regions_pads_half_frame_gap():
    with tempfile.TemporaryDirectory() as tmp_dir:
        paths = []
        for i in range(3):
            img_path = os.path.join(tmp_dir, f"frame_{i}.png")
            img = np.zeros((720, 1280, 3), dtype=np.uint8)
            cv2.imwrite(img_path, img)
            paths.append(img_path)

        frames = [
            Frame(timestamp=0.25, image_path=paths[0]),
            Frame(timestamp=0.75, image_path=paths[1]),
            Frame(timestamp=1.25, image_path=paths[2]),
        ]

        fake_proposal = FakeProposalProvider([300.0, 620.0, 900.0, 700.0], 0.9)
        provider = HybridSubtitleRegionDetectorProvider(proposal_provider=fake_proposal)

        regions = provider.detect_regions(frames)
        assert len(regions) >= 1
        r = regions[0]
        assert 0.0 <= r.x <= 1.0
        assert 0.0 <= r.y <= 1.0
        assert r.y >= 0.5
        assert r.end_time >= r.start_time
        assert r.confidence == 0.9
        assert r.start_time == 0.0
        assert abs(r.end_time - 1.5) < 1e-6

def test_provider_accepts_device_option():
    provider = HybridSubtitleRegionDetectorProvider(
        proposal_provider=FakeProposalProvider([100, 100, 200, 200]),
        device="cuda",
    )
    assert provider._device == "cuda"

class FakeDriftingProposalProvider:
    def __init__(self, bboxes, confidence=0.9):
        self._bboxes = bboxes
        self._confidence = confidence

    def detect_images(self, image_paths):
        dets = []
        for fi, p in enumerate(image_paths):
            bbox = self._bboxes[fi]
            dets.append({
                "frame_index": fi,
                "timestamp_ms": round(fi * 1000.0, 1),
                "candidate_id": f"F{fi:05d}_C01",
                "bbox_xyxy": bbox,
                "bbox_normalized": [
                    bbox[0]/1280, bbox[1]/720,
                    bbox[2]/1280, bbox[3]/720
                ],
                "confidence": self._confidence,
                "class": "text",
            })
        return dets

def test_detect_regions_moving_track_emits_per_frame_regions():
    with tempfile.TemporaryDirectory() as tmp_dir:
        paths = []
        for i in range(3):
            img_path = os.path.join(tmp_dir, f"frame_{i}.png")
            img = np.zeros((720, 1280, 3), dtype=np.uint8)
            cv2.imwrite(img_path, img)
            paths.append(img_path)

        frames = [
            Frame(timestamp=0.25, image_path=paths[0]),
            Frame(timestamp=0.75, image_path=paths[1]),
            Frame(timestamp=1.25, image_path=paths[2]),
        ]

        # Drifting x coords that stay in same track (adjacent IoU > 0.25)
        # but first-last IoU < 0.30 to trigger per-frame region emission.
        bboxes = [
            [100.0, 600.0, 300.0, 650.0],
            [200.0, 600.0, 400.0, 650.0],
            [300.0, 600.0, 500.0, 650.0],
        ]
        fake_proposal = FakeDriftingProposalProvider(bboxes, 0.9)
        provider = HybridSubtitleRegionDetectorProvider(proposal_provider=fake_proposal)

        regions = provider.detect_regions(frames)
        assert len(regions) == 3
        assert abs(regions[0].x - 100.0/1280) < 1e-6
        assert abs(regions[1].x - 200.0/1280) < 1e-6
        assert abs(regions[2].x - 300.0/1280) < 1e-6
        assert abs(regions[0].start_time - 0.0) < 1e-6
        assert abs(regions[0].end_time - 0.5) < 1e-6
        assert abs(regions[1].start_time - 0.5) < 1e-6
        assert abs(regions[1].end_time - 1.0) < 1e-6
        assert abs(regions[2].start_time - 1.0) < 1e-6
        assert abs(regions[2].end_time - 1.5) < 1e-6

class FakeVerifier:
    def __init__(self, decisions):
        self._decisions = decisions

    def classify_track(self, track, frame_bgr, candidate_ids, candidate_bboxes):
        return {"track_id": track.track_id, "decisions": self._decisions, "uncertain": False}

def test_detect_regions_rule_rejected_track_accepted_with_fake_verifier():
    with tempfile.TemporaryDirectory() as tmp_dir:
        paths = []
        for i in range(3):
            img_path = os.path.join(tmp_dir, f"frame_{i}.png")
            img = np.zeros((720, 1280, 3), dtype=np.uint8)
            cv2.imwrite(img_path, img)
            paths.append(img_path)

        frames = [
            Frame(timestamp=0.0, image_path=paths[0]),
            Frame(timestamp=0.5, image_path=paths[1]),
            Frame(timestamp=1.0, image_path=paths[2]),
        ]

        # Middle-of-frame bbox: y2=380, H=720, ratio=0.528 <= 0.65
        # Rule classify_track_by_position labels it scene_text -> is_subtitle=False -> rule rejects.
        middle_bbox = [300.0, 300.0, 900.0, 380.0]

        # Without verifier: rule rejects all -> 0 regions.
        provider_no_verifier = HybridSubtitleRegionDetectorProvider(
            proposal_provider=FakeProposalProvider(middle_bbox, 0.9),
        )
        regions_no_v = provider_no_verifier.detect_regions(frames)
        assert len(regions_no_v) == 0

        # With verifier that says it IS subtitle -> track accepted.
        fake_decisions = [
            {"candidate_id": "G00000_01", "is_subtitle": True, "category": "subtitle", "confidence": 0.9}
        ]
        fake_verifier = FakeVerifier(fake_decisions)
        provider_with_v = HybridSubtitleRegionDetectorProvider(
            proposal_provider=FakeProposalProvider(middle_bbox, 0.9),
            semantic_verifier=fake_verifier,
        )
        regions_with_v = provider_with_v.detect_regions(frames)
        assert len(regions_with_v) >= 1
