import pytest
import numpy as np

from src.application.dto.proposal_bundle import ProposalBundle
from src.infrastructure.benchmark.hybrid_subtitle.crop_extractor import TrackCropExtractor

def test_crop_extractor_correctly_crops_with_padding():
    # 1. Setup mock source frame
    # Create a 720x1280 (H=720, W=1280) source frame with a distinct pattern
    img = np.zeros((720, 1280, 3), dtype=np.uint8)
    # Fill target crop area [100, 600, 300, 650] (x1=100, y1=600, x2=300, y2=650)
    # with a distinct color (e.g. green)
    img[600:650, 100:300] = [0, 255, 0]
    
    # 2. Setup proposal bundle
    bundle = ProposalBundle(video_path="video.mp4", fps=30.0, stride=2, video_width=1280, video_height=720)
    bundle.add_frame(
        frame_index=0,
        timestamp_sec=0.0,
        image=img,
        image_path=None,
        detections=[{
            "frame_index": 0,
            "timestamp_ms": 0.0,
            "candidate_id": "F00000_C01",
            "bbox_xyxy": [100.0, 600.0, 300.0, 650.0],
            "confidence": 0.9,
        }]
    )
    bundle.add_frame(
        frame_index=1,
        timestamp_sec=1.0,
        image=img,
        image_path=None,
        detections=[{
            "frame_index": 1,
            "timestamp_ms": 1000.0,
            "candidate_id": "F00001_C01",
            "bbox_xyxy": [150.0, 600.0, 350.0, 650.0],
            "confidence": 0.9,
        }]
    )
    
    # 3. Setup track
    track = {
        "track_id": "T001",
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
                "frame_index": 1,
                "timestamp_ms": 1000.0,
                "candidate_group_id": "G00001_01",
                "member_candidate_ids": ["F00001_C01"],
                "union_bbox_xyxy": [150.0, 600.0, 350.0, 650.0],
            }
        ]
    }
    
    extractor = TrackCropExtractor()
    evidence = extractor.extract_evidence(track, bundle, padding=8)
    
    assert "crops" in evidence
    assert len(evidence["crops"]) == 2
    
    # Verify crop 1 has correct shape with padding (padding=8)
    # x1=100-8=92, y1=600-8=592, x2=300+8=308, y2=650+8=658
    # Expected height = 658 - 592 = 66
    # Expected width = 308 - 92 = 216
    crop1 = evidence["crops"][0]
    assert crop1.shape == (66, 216, 3)
    
    # Verify crop pixels match source frame exactly
    expected_crop1 = img[592:658, 92:308]
    assert np.array_equal(crop1, expected_crop1)
    
    # Verify trajectory metrics
    traj = evidence["trajectory"]
    assert traj["first_bbox"] == [100.0, 600.0, 300.0, 650.0]
    assert traj["last_bbox"] == [150.0, 600.0, 350.0, 650.0]
    assert traj["duration"] == 1.0  # 1000ms - 0ms
    assert traj["bottom_y_ratio"] == 650.0 / 720.0  # union_bbox y2=650, video_height=720
    
    # Centers:
    # First: [100, 600, 300, 650] -> center = (200, 625)
    # Last: [150, 600, 350, 650] -> center = (250, 625)
    # Displacement: sqrt((200-250)^2 + (625-625)^2) = 50.0
    assert traj["first_center"] == (200.0, 625.0)
    assert traj["last_center"] == (250.0, 625.0)
    assert abs(traj["displacement"] - 50.0) < 1e-6

def test_crop_extractor_no_side_effects_on_bundle():
    img = np.zeros((720, 1280, 3), dtype=np.uint8)
    bundle = ProposalBundle(video_path="video.mp4", fps=30.0, stride=2, video_width=1280, video_height=720)
    bundle.add_frame(
        frame_index=0,
        timestamp_sec=0.0,
        image=img,
        image_path=None,
        detections=[{
            "frame_index": 0,
            "timestamp_ms": 0.0,
            "candidate_id": "F00000_C01",
            "bbox_xyxy": [100.0, 600.0, 300.0, 650.0],
            "confidence": 0.9,
        }]
    )
    track = {
        "track_id": "T001",
        "groups": [
            {
                "frame_index": 0,
                "timestamp_ms": 0.0,
                "candidate_group_id": "G00000_01",
                "member_candidate_ids": ["F00000_C01"],
                "union_bbox_xyxy": [100.0, 600.0, 300.0, 650.0],
            }
        ]
    }
    extractor = TrackCropExtractor()
    evidence = extractor.extract_evidence(track, bundle, padding=8)
    
    # Mutate crop to check side effects
    crop = evidence["crops"][0]
    crop[0, 0] = [255, 255, 255]
    
    # Verify bundle image is unchanged
    original_frame_img = bundle.frame_for_index(0).image
    assert not np.array_equal(original_frame_img[592:658, 92:308], crop)
    assert np.all(original_frame_img == 0)  # should still be all zeros
