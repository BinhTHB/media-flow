import builtins
import sys
from types import ModuleType

import pytest
from PIL import Image

from src.domain.entities.frame import Frame
from src.infrastructure.factory.provider_factory import ProviderFactory
from src.infrastructure.providers.subtitle.paddle_subtitle_region_detector_provider import (
    PaddleOCRSubtitleRegionDetectorProvider,
    polygon_bbox,
    transform_polygon,
)
from src.infrastructure.registry.provider_registry import ProviderRegistry
from src.shared.exceptions import ProviderError


def _frame(tmp_path, timestamp=3.0):
    image_path = tmp_path / "frame.jpg"
    Image.new("RGB", (1000, 500)).save(image_path)
    return Frame(timestamp=timestamp, image_path=str(image_path))


def _install_paddle(monkeypatch, result=None, error=None, init_error=None):
    class FakePaddleOCR:
        def __init__(self, **kwargs):
            if init_error:
                raise init_error

        def ocr(self, image_path, cls=True):
            if error:
                raise error
            return result

    module = ModuleType("paddleocr")
    module.PaddleOCR = FakePaddleOCR
    monkeypatch.setitem(sys.modules, "paddleocr", module)


def _line(box, confidence=0.9, text="subtitle"):
    return [box, [text, confidence]]


def test_coordinate_transforms_preserve_original_frame_space():
    polygon = [[10, 20], [30, 20], [30, 40], [10, 40]]

    assert transform_polygon(polygon, "identity", 100, 200) == polygon
    assert transform_polygon(polygon, "vertical_flip", 100, 200) == [
        [10.0, 180.0], [30.0, 180.0], [30.0, 160.0], [10.0, 160.0],
    ]
    assert transform_polygon(polygon, "rotate180", 100, 200) == [
        [90.0, 180.0], [70.0, 180.0], [70.0, 160.0], [90.0, 160.0],
    ]
    assert transform_polygon(polygon, "identity", 100, 200, 200, 400) == [
        [20.0, 40.0], [60.0, 40.0], [60.0, 80.0], [20.0, 80.0],
    ]
    assert polygon_bbox([[90, 180], [70, 160]]) == {
        "left": 70, "top": 160, "right": 90, "bottom": 180,
    }


def test_wraps_missing_paddleocr_import(monkeypatch):
    original_import = builtins.__import__

    def missing_paddleocr(name, *args, **kwargs):
        if name == "paddleocr":
            raise ImportError("missing paddleocr")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing_paddleocr)

    with pytest.raises(ProviderError, match="paddleocr is not installed") as exc_info:
        PaddleOCRSubtitleRegionDetectorProvider(use_gpu=False)._get_ocr()

    assert isinstance(exc_info.value.__cause__, ImportError)


def test_wraps_paddleocr_initialization_error(monkeypatch):
    _install_paddle(monkeypatch, init_error=RuntimeError("bad model"))

    with pytest.raises(ProviderError, match="PaddleOCR initialization failed") as exc_info:
        PaddleOCRSubtitleRegionDetectorProvider(use_gpu=False)._get_ocr()

    assert isinstance(exc_info.value.__cause__, RuntimeError)


def test_wraps_paddleocr_runtime_error(monkeypatch, tmp_path):
    _install_paddle(monkeypatch, error=RuntimeError("bad frame"))

    with pytest.raises(ProviderError, match="PaddleOCR detection failed") as exc_info:
        PaddleOCRSubtitleRegionDetectorProvider(use_gpu=False).detect_regions([_frame(tmp_path)])

    assert isinstance(exc_info.value.__cause__, RuntimeError)


def test_wraps_malformed_paddleocr_output(monkeypatch, tmp_path):
    _install_paddle(monkeypatch, result=[["bad"]])

    with pytest.raises(ProviderError, match="invalid line shape"):
        PaddleOCRSubtitleRegionDetectorProvider(use_gpu=False).detect_regions([_frame(tmp_path)])


def test_normalizes_paddleocr_coordinates(monkeypatch, tmp_path):
    _install_paddle(
        monkeypatch,
        result=[[_line([[100, 350], [900, 350], [900, 450], [100, 450]])]],
    )
    frame = _frame(tmp_path)

    regions = PaddleOCRSubtitleRegionDetectorProvider(use_gpu=False).detect_regions([frame])

    assert len(regions) == 1
    region = regions[0]
    assert region.start_time == 3.0
    assert region.end_time == 5.0
    assert region.x == pytest.approx(0.1)
    assert region.y == pytest.approx(0.7)
    assert region.width == pytest.approx(0.8)
    assert region.height == pytest.approx(0.2)
    assert region.confidence == pytest.approx(0.9)


def test_filters_low_confidence_boxes(monkeypatch, tmp_path):
    _install_paddle(
        monkeypatch,
        result=[[_line([[100, 350], [900, 350], [900, 450], [100, 450]], 0.69)]],
    )

    assert PaddleOCRSubtitleRegionDetectorProvider(use_gpu=False).detect_regions([_frame(tmp_path)]) == []


def test_accepts_center_screen_text_and_filters_border_artifacts(monkeypatch, tmp_path):
    upper_box = [[100, 100], [900, 100], [900, 200], [100, 200]]
    off_center_box = [[0, 350], [100, 350], [100, 450], [0, 450]]
    _install_paddle(
        monkeypatch,
        result=[[_line(upper_box), _line(off_center_box)]],
    )

    regions = PaddleOCRSubtitleRegionDetectorProvider(use_gpu=False).detect_regions([_frame(tmp_path)])

    assert len(regions) == 1
    assert regions[0].y == pytest.approx(0.2)


def test_merges_valid_boxes_for_a_frame(monkeypatch, tmp_path):
    left = [[100, 350], [450, 350], [450, 450], [100, 450]]
    right = [[500, 360], [900, 360], [900, 460], [500, 460]]
    _install_paddle(
        monkeypatch,
        result=[[_line(left, 0.8), _line(right, 0.95)]],
    )

    regions = PaddleOCRSubtitleRegionDetectorProvider(use_gpu=False).detect_regions([_frame(tmp_path)])

    assert len(regions) == 1
    region = regions[0]
    assert region.x == pytest.approx(0.1)
    assert region.y == pytest.approx(0.7)
    assert region.width == pytest.approx(0.8)
    assert region.height == pytest.approx(0.22)
    assert region.confidence == pytest.approx(0.95)


def test_registry_and_factory_create_paddle_region_detector():
    registry_provider = ProviderRegistry().create(
        "subtitle_region_detector",
        "paddle",
        {"language": "en", "use_gpu": True, "min_confidence": 0.85},
    )
    factory_provider = ProviderFactory(
        config={
            "defaults": {"subtitle_region_detector": "paddle"},
            "providers": {
                "subtitle_region_detector": {
                    "paddle": {"language": "en", "use_gpu": True, "min_confidence": 0.85}
                }
            },
        }
    ).create_subtitle_region_detector_provider()

    assert isinstance(registry_provider, PaddleOCRSubtitleRegionDetectorProvider)
    assert isinstance(factory_provider, PaddleOCRSubtitleRegionDetectorProvider)
    assert registry_provider.language == factory_provider.language == "en"
    assert registry_provider.use_gpu is factory_provider.use_gpu is True
    assert registry_provider.min_confidence == factory_provider.min_confidence == 0.85
def _checkpoint_frame(tmp_path, name, timestamp):
    frames_dir = tmp_path / "job" / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    image_path = frames_dir / f"{name}.jpg"
    Image.new("RGB", (1000, 500)).save(image_path)
    return Frame(timestamp=timestamp, image_path=str(image_path))


def test_region_detector_writes_and_reuses_frame_checkpoint(monkeypatch, tmp_path):
    calls = []
    class FakePaddleOCR:
        def __init__(self, **kwargs):
            pass
        def ocr(self, image_path, cls=True):
            calls.append(image_path)
            return [[_line([[100, 350], [900, 350], [900, 450], [100, 450]])]]
    module = ModuleType("paddleocr")
    module.PaddleOCR = FakePaddleOCR
    monkeypatch.setitem(sys.modules, "paddleocr", module)
    frame = _checkpoint_frame(tmp_path, "frame_0001", 3.0)
    provider = PaddleOCRSubtitleRegionDetectorProvider(use_gpu=False)
    assert len(provider.detect_regions([frame])) == 1
    checkpoint = tmp_path / "job" / "debug" / "subtitle_region_frames" / "frame_0001.json"
    data = __import__("json").loads(checkpoint.read_text(encoding="utf-8"))
    assert set(("frame_id", "frame_path", "frame_timestamp", "provider_name", "processing_duration", "regions", "error")) <= set(data)
    assert data["regions"][0]["confidence"] == pytest.approx(0.9)
    assert {"x", "y", "width", "height"} <= set(data["regions"][0])
    provider.detect_regions([frame])
    assert len(calls) == 1


def test_region_detector_keeps_completed_checkpoint_when_later_frame_fails(monkeypatch, tmp_path):
    class FakePaddleOCR:
        def __init__(self, **kwargs):
            pass
        def ocr(self, image_path, cls=True):
            if image_path.endswith("frame_0002.jpg"):
                raise RuntimeError("bad frame")
            return [[_line([[100, 350], [900, 350], [900, 450], [100, 450]])]]
    module = ModuleType("paddleocr")
    module.PaddleOCR = FakePaddleOCR
    monkeypatch.setitem(sys.modules, "paddleocr", module)
    first = _checkpoint_frame(tmp_path, "frame_0001", 3.0)
    second = _checkpoint_frame(tmp_path, "frame_0002", 5.0)
    with pytest.raises(ProviderError):
        PaddleOCRSubtitleRegionDetectorProvider(use_gpu=False).detect_regions([first, second])
    base = tmp_path / "job" / "debug" / "subtitle_region_frames"
    assert (base / "frame_0001.json").exists()
    failed = __import__("json").loads((base / "frame_0002.json").read_text(encoding="utf-8"))
    assert failed["error"] == "bad frame"
