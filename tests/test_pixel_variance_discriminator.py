from __future__ import annotations

import numpy as np

from src.application.dto.proposal_bundle import ProposalBundle
from src.infrastructure.benchmark.hybrid_subtitle.pixel_variance_discriminator import (
    PixelVarianceDiscriminator,
)


def _bundle(first: np.ndarray, second: np.ndarray) -> ProposalBundle:
    bundle = ProposalBundle(
        "video.mp4",
        fps=30.0,
        stride=1,
        video_width=100,
        video_height=100,
    )
    for index, image in enumerate((first, second)):
        bundle.add_frame(
            frame_index=index,
            timestamp_sec=index / 30,
            image=image,
            image_path=None,
            detections=[],
        )
    return bundle


def _track() -> dict:
    return {
        "track_id": "T001",
        "groups": [
            {"frame_index": 0, "union_bbox_xyxy": [10, 10, 90, 90]},
            {"frame_index": 1, "union_bbox_xyxy": [10, 10, 90, 90]},
        ],
    }


def test_rejects_static_track_below_variance_threshold():
    image = np.zeros((100, 100, 3), dtype=np.uint8)

    evidence = PixelVarianceDiscriminator(10.0).evaluate(
        _track(), _bundle(image, image.copy())
    )

    assert evidence.rejected is True
    assert evidence.temporal_pixel_variance == 0.0
    assert evidence.reason == "low_temporal_pixel_variance"


def test_keeps_changing_track_above_variance_threshold():
    first = np.zeros((100, 100, 3), dtype=np.uint8)
    second = np.full((100, 100, 3), 255, dtype=np.uint8)

    evidence = PixelVarianceDiscriminator(10.0).evaluate(
        _track(), _bundle(first, second)
    )

    assert evidence.rejected is False
    assert evidence.temporal_pixel_variance is not None
    assert evidence.temporal_pixel_variance > 10.0
    assert evidence.reason == "passed"


def test_fails_closed_when_temporal_crops_are_insufficient():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    bundle = _bundle(image, image)
    track = {"track_id": "T001", "groups": [{"frame_index": 0, "union_bbox_xyxy": [10, 10, 90, 90]}]}

    evidence = PixelVarianceDiscriminator(10.0).evaluate(track, bundle)

    assert evidence.rejected is True
    assert evidence.temporal_pixel_variance is None
    assert evidence.reason == "insufficient_temporal_crops"
