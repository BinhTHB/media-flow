from pathlib import Path

import pytest

from src.core.models.bbox import BBox
from src.core.models.speech_segment import SpeechSegment
from src.features.audio.models import AudioRequest
from src.features.audio.pipeline import AudioPipeline


def test_bbox_rejects_invalid_geometry_and_explicit_clamp_repairs_it():
    with pytest.raises(ValueError, match="x coordinates"):
        BBox(-10, 5, 120, 20, 100, 50)
    with pytest.raises(ValueError, match="x coordinates"):
        BBox(20, 5, 20, 20, 100, 50)

    assert BBox.clamped(-10, 5, 120, 20, 100, 50) == BBox(0, 5, 100, 20, 100, 50)
    assert BBox.clamped(100, 5, 100, 20, 100, 50) == BBox(99, 5, 100, 20, 100, 50)


def test_speech_segment_rejects_invalid_timing_and_explicit_factory_normalizes_it():
    with pytest.raises(ValueError, match="non-negative"):
        SpeechSegment("S1", -0.1, 1.0, "text")
    with pytest.raises(ValueError, match="greater than start_time"):
        SpeechSegment("S1", 1.0, 1.0, "text")

    segment = SpeechSegment.normalized_timing(
        segment_id="S1", start_time=-0.1, end_time=0.0, source_text="text"
    )
    assert segment.start_time == 0.0
    assert segment.end_time == pytest.approx(0.1)


def test_audio_pipeline_extracts_video_through_injected_port(tmp_path: Path):
    class Extractor:
        def __init__(self):
            self.calls = []

        def extract_audio(self, video_path, output_audio_path):
            self.calls.append((video_path, output_audio_path))
            return output_audio_path

    class ASR:
        def transcribe(self, audio_path, language="auto"):
            return []

    extractor = Extractor()
    result = AudioPipeline(asr_provider=ASR(), audio_extractor=extractor).process(
        AudioRequest(enabled=True, audio_path="INPUT.MP4"), str(tmp_path)
    )

    assert result.status == "SUCCESS"
    assert extractor.calls == [("INPUT.MP4", str(tmp_path / "audio" / "audio.wav"))]


def test_audio_pipeline_requires_extractor_for_video_input(tmp_path: Path):
    with pytest.raises(RuntimeError, match="Audio extraction provider"):
        AudioPipeline().process(
            AudioRequest(enabled=True, audio_path="input.mp4"), str(tmp_path)
        )
