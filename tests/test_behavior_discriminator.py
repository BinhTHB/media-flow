from src.domain.entities.segment import Segment
from src.infrastructure.benchmark.hybrid_subtitle.behavior_discriminator import (
    TrackBehaviorDiscriminator,
)
from src.application.dto.proposal_bundle import ProposalBundle
import numpy as np


def _track(start_ms=0.0, end_ms=4000.0, displacement=0.0):
    return {
        "track_id": "T001",
        "groups": [
            {
                "timestamp_ms": start_ms,
                "union_bbox_xyxy": [100.0, 100.0, 300.0, 150.0],
            },
            {
                "timestamp_ms": end_ms,
                "union_bbox_xyxy": [
                    100.0 + displacement,
                    100.0,
                    300.0 + displacement,
                    150.0,
                ],
            },
        ],
    }


def test_rejects_durable_static_track_without_speech():
    discriminator = TrackBehaviorDiscriminator(
        [Segment(id="speech_1", start=5.0, end=6.0, text="later", source="speech")],
        video_duration_seconds=20.0,
    )

    evidence = discriminator.evaluate(_track())

    assert evidence.rule_fired is True
    assert evidence.speech_overlap_ratio == 0.0
    assert evidence.coverage_ratio == 0.2


def test_keeps_dialogue_track_with_speech_overlap():
    discriminator = TrackBehaviorDiscriminator(
        [Segment(id="speech_1", start=1.0, end=3.0, text="dialogue", source="speech")],
        video_duration_seconds=20.0,
    )

    evidence = discriminator.evaluate(_track())

    assert evidence.rule_fired is False
    assert evidence.speech_overlap_ratio == 0.5


def test_missing_transcript_is_neutral():
    evidence = TrackBehaviorDiscriminator([], video_duration_seconds=20.0).evaluate(
        _track()
    )

    assert evidence.transcript_available is False
    assert evidence.rule_fired is False


def test_rejects_durable_static_text_even_with_continuous_speech():
    bundle = ProposalBundle("video.mp4", fps=30.0, stride=1, video_width=1280, video_height=720)
    image = np.zeros((720, 1280, 3), dtype=np.uint8)
    for index, timestamp in ((0, 0.0), (120, 4.0)):
        bundle.add_frame(
            frame_index=index,
            timestamp_sec=timestamp,
            image=image,
            image_path=None,
            detections=[],
        )
    track = _track()
    track["groups"][0]["frame_index"] = 0
    track["groups"][1]["frame_index"] = 120
    discriminator = TrackBehaviorDiscriminator(
        [Segment(id="speech", start=0.0, end=4.0, text="dialogue", source="speech")],
        video_duration_seconds=20.0,
        static_text_reader=lambda _crop: "OMEGA",
    )

    evidence = discriminator.evaluate_static_content(track, bundle)

    assert evidence.rule_fired is True
    assert evidence.rule_reason == "durable_static_text"
    assert evidence.static_text_similarity == 1.0
    assert evidence.ocr_text_samples == ("OMEGA", "OMEGA")


def test_static_content_rule_keeps_short_bottom_subtitle():
    bundle = ProposalBundle("video.mp4", fps=30.0, stride=1, video_width=1280, video_height=720)
    image = np.zeros((720, 1280, 3), dtype=np.uint8)
    bundle.add_frame(
        frame_index=0,
        timestamp_sec=0.0,
        image=image,
        image_path=None,
        detections=[],
    )
    bundle.add_frame(
        frame_index=21,
        timestamp_sec=0.7,
        image=image,
        image_path=None,
        detections=[],
    )
    track = _track(end_ms=700.0)
    track["groups"][0]["frame_index"] = 0
    track["groups"][1]["frame_index"] = 21

    evidence = TrackBehaviorDiscriminator(
        static_text_reader=lambda _crop: "subtitle"
    ).evaluate_static_content(track, bundle)

    assert evidence.rule_fired is False
    assert evidence.rule_reason == "neutral"
