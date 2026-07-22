from src.domain.entities.segment import Segment
from src.infrastructure.benchmark.hybrid_subtitle.behavior_discriminator import (
    TrackBehaviorDiscriminator,
)


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
