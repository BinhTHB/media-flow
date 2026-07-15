import json

import pytest
from PIL import Image

from src.application.ports.ocr_provider import OCRProvider
from src.domain.entities.frame import Frame
from src.infrastructure.factory.provider_factory import ProviderFactory
from src.infrastructure.gemini.runtime import GeminiRuntime, RuntimeConfig
from src.infrastructure.providers.ocr.gemini_vision_ocr_provider import GeminiVisionOCRProvider
from src.infrastructure.providers.translation.gemini_translation_provider import GeminiTranslationProvider
from src.shared.exceptions import ProviderError


def _response(payload):
    return {
        "candidates": [{"content": {"parts": [{"text": json.dumps(payload, ensure_ascii=False)}]}}],
        "modelStatus": {"modelStage": "STABLE"},
    }


class FakeTransport:
    def __init__(self, payloads):
        self.payloads = list(payloads)
        self.calls = []

    def send(self, method, url, *, headers, data=None, timeout=60.0):
        self.calls.append((method, json.loads(data.decode("utf-8")) if data else None))
        if method == "GET":
            return {"name": "models/test", "supportedGenerationMethods": ["generateContent"], "modelStage": "STABLE"}
        if not self.payloads:
            raise AssertionError("Unexpected REST request")
        return _response(self.payloads.pop(0))


def _frame(tmp_path, name="frame.png", size=(1920, 1080), timestamp=1.0):
    path = tmp_path / name
    Image.new("RGB", size, "white").save(path)
    return Frame(timestamp=timestamp, image_path=str(path))


def _provider(tmp_path, payloads, **kwargs):
    runtime = GeminiRuntime.create(RuntimeConfig(api_key="fake-api-key", cache_dir=str(tmp_path / "cache"), cache_enabled=True))
    provider = GeminiVisionOCRProvider(runtime=runtime, api_key="fake-api-key", **kwargs)
    transport = FakeTransport(payloads)
    runtime.get_client(provider.model, role="primary")._transport = transport
    runtime.get_client(provider.escalation_model, role="escalation")._transport = transport
    return provider, transport


def _detection(text="原文字幕", box=None, status="readable"):
    return {"text_verbatim": text, "language": "zh", "text_type": "subtitle", "box_2d": box or [100, 200, 300, 800], "transcription_status": status}


def test_provider_implements_port_and_uses_injected_runtime(tmp_path):
    runtime = GeminiRuntime.create()
    provider = GeminiVisionOCRProvider(runtime=runtime)
    assert isinstance(provider, OCRProvider)
    assert provider.runtime is runtime
    assert provider.runtime.scheduler is runtime.scheduler


def test_full_frame_schema_unicode_and_pixel_conversion(tmp_path):
    frame = _frame(tmp_path, size=(1920, 1080))
    payload = {"frames": [{"frame_id": "frame_0", "timestamp": 1.0, "detections": [_detection("原文字幕"), _detection("Xin chào", [400, 0, 1000, 1000])] }]}
    provider, transport = _provider(tmp_path, [payload])
    segments = provider.detect_text([frame])
    assert [segment.text for segment in segments] == ["原文字幕"]
    assert provider.observations[0]["pixel_box"] == [384.0, 108.0, 1536.0, 324.0]
    assert "inline_data" in transport.calls[0][1]["contents"][0]["parts"][2]
    assert "Do not translate" in provider.build_prompt()


@pytest.mark.parametrize(("size", "box", "expected"), [
    ((1080, 1920), [0, 0, 1000, 1000], [0.0, 0.0, 1080.0, 1920.0]),
    ((1234, 567), [100, 200, 900, 800], [246.8, 56.7, 987.2, 510.3]),
])
def test_bbox_conversion_for_portrait_and_odd_dimensions(tmp_path, size, box, expected):
    frame = _frame(tmp_path, size=size)
    payload = {"frames": [{"frame_id": "frame_0", "timestamp": 1.0, "detections": [_detection(box=box)]}]}
    provider, _ = _provider(tmp_path, [payload])
    provider.detect_text([frame])
    assert provider.observations[0]["pixel_box"] == pytest.approx(expected)


def test_empty_duplicates_batch_order_and_cache_hit(tmp_path):
    first = _frame(tmp_path, "a.png", timestamp=1.0)
    second = _frame(tmp_path, "b.png", timestamp=2.0)
    payload = {"frames": [
        {"frame_id": "frame_0", "timestamp": 1.0, "detections": [_detection(), _detection()]},
        {"frame_id": "frame_1", "timestamp": 2.0, "detections": []},
    ]}
    provider, transport = _provider(tmp_path, [payload], batch_size=2)
    assert [segment.text for segment in provider.detect_text([first, second])] == ["原文字幕"]
    assert "duplicate_dropped" in provider.observations[1]["review_flags"]
    provider.detect_text([first, second])
    assert len(transport.calls) == 1


@pytest.mark.parametrize("box", ([0, 0, 0, 1], [0, 0, 1, 1001], [10, 20, 5, 30]))
def test_invalid_bbox_is_rejected(tmp_path, box):
    frame = _frame(tmp_path)
    payload = {"frames": [{"frame_id": "frame_0", "timestamp": 1.0, "detections": [_detection(box=box)]}]}
    provider, _ = _provider(tmp_path, [payload])
    with pytest.raises(ProviderError):
        provider.detect_text([frame])


def test_wrong_frame_or_timestamp_is_rejected(tmp_path):
    frame = _frame(tmp_path)
    for invalid in (
        {"frames": [{"frame_id": "frame_1", "timestamp": 1.0, "detections": []}]},
        {"frames": [{"frame_id": "frame_0", "timestamp": 2.0, "detections": []}]},
    ):
        provider, _ = _provider(tmp_path, [invalid])
        with pytest.raises(ProviderError):
            provider.detect_text([frame])


def test_crop_retry_only_for_ambiguous_detection(tmp_path):
    frame = _frame(tmp_path)
    primary = {"frames": [{"frame_id": "frame_0", "timestamp": 1.0, "detections": [_detection(status="ambiguous")]}]}
    crop = {"text_verbatim": "原文字幕", "transcription_status": "readable"}
    provider, transport = _provider(tmp_path, [primary, crop])
    assert [segment.text for segment in provider.detect_text([frame])] == ["原文字幕"]
    assert len(transport.calls) == 2
    assert "crop_retry_resolved" in provider.observations[0]["review_flags"]


def test_escalation_budget_and_conflict_fail_closed(tmp_path):
    frame = _frame(tmp_path)
    primary = {"frames": [{"frame_id": "frame_0", "timestamp": 1.0, "detections": [_detection(status="ambiguous")]}]}
    ambiguous = {"text_verbatim": "", "transcription_status": "ambiguous"}
    conflict = {"text_verbatim": "不同", "transcription_status": "readable"}
    provider, transport = _provider(tmp_path, [primary, ambiguous, conflict], escalation_enabled=True, max_escalation_requests=1)
    assert provider.detect_text([frame]) == []
    assert len(transport.calls) == 3
    assert "REVIEW_REQUIRED" in provider.observations[0]["review_flags"]


def test_shadow_artifacts_and_registry_share_translation_runtime(tmp_path):
    config = {"defaults": {"ocr": "gemini_vision", "translation": "gemini"}, "providers": {"ocr": {"gemini_vision": {}}, "translation": {"gemini": {}}}}
    factory = ProviderFactory(config=config, gemini_api_key="test")
    ocr = factory.create_ocr_provider()
    translation = factory.create_translation_provider()
    assert isinstance(ocr, GeminiVisionOCRProvider)
    assert isinstance(translation, GeminiTranslationProvider)
    assert ocr.runtime is translation._runtime
    ocr.write_qualification_artifacts("job", [], [], output_dir=str(tmp_path))
    root = tmp_path / "job" / "qualification" / "gemini_ocr"
    assert {path.name for path in root.iterdir()} == {"observations.json", "provenance.json", "usage.json", "comparison.json"}
