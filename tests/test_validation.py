import pytest
from src.domain.entities.segment import Segment
from src.domain.entities.feature_config import FeatureConfig
from src.domain.entities.video_job import VideoJob
from src.shared.exceptions import ValidationError

def test_valid_segment():
    seg = Segment(id="s1", start=0.0, end=1.5, text="valid", source="speech", confidence=0.9)
    assert seg.id == "s1"
    assert seg.start == 0.0
    assert seg.end == 1.5
    assert seg.text == "valid"
    assert seg.source == "speech"
    assert seg.confidence == 0.9

def test_invalid_segment_start():
    with pytest.raises(ValidationError, match="Segment.start must be non-negative"):
        Segment(id="s1", start=-1.0, end=1.5, text="invalid", source="speech", confidence=0.9)

def test_invalid_segment_end():
    with pytest.raises(ValidationError, match="Segment.end must be greater than or equal to start"):
        Segment(id="s1", start=2.0, end=1.0, text="invalid", source="speech", confidence=0.9)

def test_invalid_segment_source():
    with pytest.raises(ValidationError, match="Segment.source must be either 'speech' or 'ocr'"):
        Segment(id="s1", start=0.0, end=1.0, text="invalid", source="invalid_src", confidence=0.9)

def test_invalid_segment_confidence():
    with pytest.raises(ValidationError, match="Segment.confidence must be between 0.0 and 1.0"):
        Segment(id="s1", start=0.0, end=1.0, text="invalid", source="speech", confidence=1.5)

def test_valid_video_job():
    job = VideoJob(job_id="j1", input_video="test.mp4")
    assert job.job_id == "j1"
    assert job.input_video == "test.mp4"
    assert job.source_language == "auto"
    assert job.speech_language == "auto"
    assert job.target_language == "vi"
    assert job.to_dict()["speech_language"] == "auto"

def test_invalid_video_job_empty_id():
    with pytest.raises(ValidationError, match="VideoJob.job_id cannot be empty"):
        VideoJob(job_id="", input_video="test.mp4")

def test_invalid_video_job_empty_video():
    with pytest.raises(ValidationError, match="VideoJob.input_video cannot be empty"):
        VideoJob(job_id="j1", input_video="")

def test_invalid_video_job_source_lang():
    with pytest.raises(ValidationError, match="VideoJob.source_language must be 'zh', 'en', or 'auto'"):
        VideoJob(job_id="j1", input_video="test.mp4", source_language="fr")

def test_invalid_video_job_speech_lang():
    with pytest.raises(ValidationError, match="VideoJob.speech_language must be"):
        VideoJob(job_id="j1", input_video="test.mp4", speech_language="fr")

def test_invalid_video_job_target_lang():
    with pytest.raises(ValidationError, match="VideoJob.target_language must be 'vi'"):
        VideoJob(job_id="j1", input_video="test.mp4", target_language="en")

def test_feature_config_validation():
    with pytest.raises(ValidationError, match="FeatureConfig.speech_to_text must be a boolean"):
        FeatureConfig(speech_to_text="yes")
