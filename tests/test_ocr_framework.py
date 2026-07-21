import os
import sys
from types import ModuleType

import pytest

from src.application.use_cases.merge_segments import MergeSegmentsUseCase
from src.domain.entities.frame import Frame
from src.domain.entities.segment import Segment
from src.infrastructure.artifacts.artifact_manager import ArtifactManager
from src.infrastructure.factory.provider_factory import ProviderFactory
from src.infrastructure.providers.ocr.mock_ocr_provider import MockOCRProvider
from src.infrastructure.providers.ocr.paddle_ocr_provider import PaddleOCRProvider
from src.infrastructure.registry.provider_registry import ProviderRegistry
from src.infrastructure.providers.video.ffmpeg_frame_extraction_provider import (
    FFmpegFrameExtractionProvider,
)
from src.infrastructure.providers.video.mock_frame_extraction_provider import MockFrameExtractionProvider
from src.shared.exceptions import ProviderError


def test_ffmpeg_frame_validation_rejects_blank_frame(tmp_path):
    import cv2
    import numpy as np

    path = tmp_path / "blank.jpg"
    assert cv2.imwrite(str(path), np.full((1280, 720, 3), 128, dtype=np.uint8))

    with pytest.raises(ProviderError, match="blank"):
        FFmpegFrameExtractionProvider()._validate_frame_files(str(tmp_path), ["blank.jpg"], 720, 1280)


def test_ffmpeg_frame_validation_rejects_wrong_resolution(tmp_path):
    import cv2
    import numpy as np

    path = tmp_path / "wrong_size.jpg"
    image = np.zeros((1080, 1920, 3), dtype=np.uint8)
    image[:, ::2] = 255
    assert cv2.imwrite(str(path), image)

    with pytest.raises(ProviderError, match="resolution"):
        FFmpegFrameExtractionProvider()._validate_frame_files(str(tmp_path), ["wrong_size.jpg"], 720, 1280)


def test_mock_frame_extraction_provider(tmp_path):
    provider = MockFrameExtractionProvider()
    frames = provider.extract_frames("sample.mp4", 1.0, str(tmp_path))
    assert len(frames) == 2
    assert frames[0].timestamp == 0.0
    assert os.path.exists(frames[0].image_path)


def test_mock_ocr_provider_returns_segments(tmp_path):
    frame_path = tmp_path / "frame_0001.jpg"
    frame_path.write_text("mock", encoding="utf-8")
    provider = MockOCRProvider()
    segments = provider.detect_text([Frame(timestamp=1.0, image_path=str(frame_path))])
    assert len(segments) == 1
    assert segments[0].source == "ocr"
    assert segments[0].start == 1.0
    assert segments[0].confidence == 0.92


def test_merge_segments_use_case_stable_order():
    speech = [Segment(id="s1", start=2.0, end=3.0, text="speech", source="speech", confidence=0.9)]
    ocr = [Segment(id="o1", start=1.0, end=2.0, text="ocr", source="ocr", confidence=0.8)]
    merged = MergeSegmentsUseCase().execute(speech, ocr)
    assert [s.id for s in merged] == ["o1", "s1"]


def test_artifact_manager_frames_dir(tmp_path):
    manager = ArtifactManager(base_dir=str(tmp_path))
    frames_dir = manager.frames_dir("job_ocr")
    assert frames_dir.endswith(os.path.normpath("job_ocr/frames"))
    assert os.path.isdir(frames_dir)


def test_provider_factory_creates_frame_and_ocr_providers():
    config = {
        "defaults": {"ocr": "mock", "frame_extraction": "mock"},
        "providers": {"ocr": {"mock": {}}, "frame_extraction": {"mock": {}}},
    }
    factory = ProviderFactory(config=config)
    assert factory.create_ocr_provider().__class__.__name__ == "MockOCRProvider"
    assert factory.create_frame_extraction_provider().__class__.__name__ == "MockFrameExtractionProvider"


def test_paddle_ocr_provider_monkeypatched(monkeypatch, tmp_path):
    initialized_with = {}

    class FakePaddleOCR:
        def __init__(self, **kwargs):
            initialized_with.update(kwargs)

        def ocr(self, image_path, cls=True):
            box = [[0, 0], [1, 0], [1, 1], [0, 1]]
            text_conf = ("文字", 0.88)
            line = [box, text_conf]
            page = [line]
            return [page]

    module = ModuleType("paddleocr")
    module.PaddleOCR = FakePaddleOCR
    monkeypatch.setitem(sys.modules, "paddleocr", module)

    frame_path = tmp_path / "frame.jpg"
    frame_path.write_text("mock", encoding="utf-8")
    provider = PaddleOCRProvider(language="ch", use_gpu=False)
    segments = provider.detect_text([Frame(timestamp=2.0, image_path=str(frame_path))])
    assert len(segments) == 1
    assert segments[0].text == "文字"
    assert segments[0].source == "ocr"
    assert segments[0].confidence == 0.88
    assert initialized_with == {
        "use_angle_cls": True,
        "lang": "ch",
        "use_gpu": False,
        "show_log": False,
    }

def test_paddle_ocr_provider_parses_v3_result(monkeypatch, tmp_path):
    initialized_with = {}

    class Polygon:
        def tolist(self):
            return [[0, 0], [1, 0], [1, 1], [0, 1]]

    class FakePaddleOCR:
        def __init__(self, **kwargs):
            initialized_with.update(kwargs)

        def ocr(self, image_path):
            return [{
                "rec_polys": [Polygon()],
                "rec_texts": ["文字"],
                "rec_scores": [0.91],
            }]

    module = ModuleType("paddleocr")
    module.__version__ = "3.7.0"
    module.PaddleOCR = FakePaddleOCR
    monkeypatch.setitem(sys.modules, "paddleocr", module)

    frame_path = tmp_path / "frame.jpg"
    frame_path.write_text("mock", encoding="utf-8")
    segments = PaddleOCRProvider(use_gpu=False).detect_text(
        [Frame(timestamp=2.0, image_path=str(frame_path))]
    )

    assert initialized_with == {
        "lang": "ch",
        "device": "cpu",
        "enable_mkldnn": False,
        "engine_config": {"paddle_static": {"enable_new_ir": False}},
    }
    assert len(segments) == 1
    assert segments[0].text == "文字"
    assert segments[0].confidence == 0.91


def test_paddle_ocr_provider_wraps_missing_dependency(monkeypatch):
    import builtins

    original_import = builtins.__import__

    def missing_paddleocr(name, *args, **kwargs):
        if name == "paddleocr":
            raise ImportError("missing paddleocr")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing_paddleocr)

    with pytest.raises(ProviderError, match="paddleocr is not installed"):
        PaddleOCRProvider(use_gpu=False)._get_ocr()


def test_paddle_ocr_provider_wraps_initialization_error(monkeypatch):
    class FailingPaddleOCR:
        def __init__(self, **kwargs):
            raise RuntimeError("bad model")

    module = ModuleType("paddleocr")
    module.PaddleOCR = FailingPaddleOCR
    monkeypatch.setitem(sys.modules, "paddleocr", module)

    with pytest.raises(ProviderError, match="PaddleOCR initialization failed: bad model"):
        PaddleOCRProvider(use_gpu=False)._get_ocr()


def test_paddle_ocr_provider_wraps_runtime_error(monkeypatch, tmp_path):
    class FailingPaddleOCR:
        def __init__(self, **kwargs):
            pass

        def ocr(self, image_path, cls=True):
            raise RuntimeError("bad frame")

    module = ModuleType("paddleocr")
    module.PaddleOCR = FailingPaddleOCR
    monkeypatch.setitem(sys.modules, "paddleocr", module)
    image_path = tmp_path / "frame.jpg"
    image_path.write_text("mock", encoding="utf-8")

    with pytest.raises(ProviderError, match="PaddleOCR detection failed"):
        PaddleOCRProvider(use_gpu=False).detect_text([Frame(timestamp=0.0, image_path=str(image_path))])

def test_paddle_ocr_provider_wraps_malformed_result(monkeypatch, tmp_path):
    class MalformedPaddleOCR:
        def __init__(self, **kwargs):
            pass

        def ocr(self, image_path, cls=True):
            return [["not-a-detection"]]

    module = ModuleType("paddleocr")
    module.PaddleOCR = MalformedPaddleOCR
    monkeypatch.setitem(sys.modules, "paddleocr", module)
    image_path = tmp_path / "frame.jpg"
    image_path.write_text("mock", encoding="utf-8")

    with pytest.raises(ProviderError, match="PaddleOCR detection failed") as exc_info:
        PaddleOCRProvider(use_gpu=False).detect_text([Frame(timestamp=0.0, image_path=str(image_path))])

    assert isinstance(exc_info.value.__cause__, ValueError)

def test_registry_creates_paddle_ocr_provider():
    provider = ProviderRegistry().create(
        "ocr", "paddle", {"language": "en", "use_gpu": True}
    )

    assert isinstance(provider, PaddleOCRProvider)
    assert provider.language == "en"
    assert provider.use_gpu is True

def test_paddle_ocr_provider_checkpoints(monkeypatch, tmp_path):
    class FakePaddleOCR:
        def __init__(self, **kwargs):
            pass

        def ocr(self, image_path, cls=True):
            box = [[0, 0], [1, 0], [1, 1], [0, 1]]
            text_conf = ("??", 0.88)
            line = [box, text_conf]
            page = [line]
            return [page]

    module = ModuleType("paddleocr")
    module.PaddleOCR = FakePaddleOCR
    monkeypatch.setitem(sys.modules, "paddleocr", module)

    job_dir = tmp_path / "my_job"
    frames_dir = job_dir / "frames"
    os.makedirs(frames_dir)
    frame_path = frames_dir / "frame_0001.jpg"
    frame_path.write_text("mock", encoding="utf-8")

    provider = PaddleOCRProvider(language="ch", use_gpu=False)
    frame = Frame(timestamp=2.0, image_path=str(frame_path))
    
    segments = provider.detect_text([frame])
    assert len(segments) == 1
    assert segments[0].text == "??"

    checkpoint_path = job_dir / "debug" / "ocr_frames" / "frame_0001.json"
    assert os.path.exists(checkpoint_path)

    import json
    with open(checkpoint_path, "r", encoding="utf-8") as f:
        cp = json.load(f)
    assert cp["frame_id"] == "frame_0001"
    assert cp["frame_path"] == str(frame_path)
    assert cp["frame_timestamp"] == 2.0
    assert cp["provider_name"] == "paddle"
    assert cp["error"] is None
    assert len(cp["segments"]) == 1
    assert cp["segments"][0]["text"] == "??"

    cp["segments"][0]["text"] = "CachedText"
    with open(checkpoint_path, "w", encoding="utf-8") as f:
        json.dump(cp, f, indent=2, ensure_ascii=False)

    segments_cached = provider.detect_text([frame])
    assert len(segments_cached) == 1
    assert segments_cached[0].text == "CachedText"
