import os

import pytest
from src.domain.entities.segment import Segment
from src.domain.entities.feature_config import FeatureConfig
from src.domain.entities.video_job import VideoJob

os.environ.setdefault("GEMINI_API_KEY", "dummy_testing_api_key")

@pytest.fixture
def sample_video_job() -> VideoJob:
    return VideoJob(
        job_id="test_job_1",
        input_video="test.mp4",
        source_language="zh",
        target_language="vi",
        features=FeatureConfig(
            speech_to_text=True,
            ocr=False,
            translate=True,
            subtitle=False,
            render_video=False,
        ),
    )

@pytest.fixture
def mock_segments() -> list:
    return [
        Segment(id="s1", start=0.0, end=1.0, text="Hello", source="speech", confidence=0.9),
        Segment(id="s2", start=2.0, end=3.0, text="World", source="speech", confidence=0.85),
    ]
