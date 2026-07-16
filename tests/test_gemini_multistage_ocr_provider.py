import json

import pytest
from PIL import Image

from src.application.ports.ocr_provider import OCRProvider
from src.domain.entities.frame import Frame
from src.infrastructure.factory.provider_factory import ProviderFactory
from src.infrastructure.gemini.runtime import GeminiRuntime, RuntimeConfig
from src.infrastructure.providers.ocr.gemini_multistage_ocr_provider import (
    GeminiMultiStageOCRProvider,
)
from src.shared.exceptions import ProviderError


def _response(payload):
    return {
        "candidates": [{"content": {"parts": [{"text": json.dumps(payload)}]}}],
        "modelStatus": {"modelStage": "STABLE"},
    }


class FakeTransport:
    def __init__(self, payloads):
        self.payloads = list(payloads)
        self.calls = []

    def send(self, method, url, *, headers, data=None, timeout=60.0):
        self.calls.append((method, json.loads(data.decode("utf-8")) if data else None))
        if method == "GET":
            return {
                "name": "models/test",
                "supportedGenerationMethods": ["generateContent"],
                "modelStage": "STABLE",
            }
        return _response(self.payloads.pop(0))


def _frame(tmp_path, name, timestamp):
    path = tmp_path / name
    Image.new("RGB", (1000, 500), "white").save(path)
    return Frame(timestamp=timestamp, image_path=str(path))


def _provider(tmp_path, payloads, **kwargs):
    runtime = GeminiRuntime.create(
        RuntimeConfig(
            api_key="fake", cache_dir=str(tmp_path / "cache"), cache_enabled=False
        )
    )
    provider = GeminiMultiStageOCRProvider(runtime=runtime, api_key="fake", **kwargs)
    transport = FakeTransport(payloads)
    runtime.get_client(provider.model, api_key="fake")._transport = transport
    return provider, transport


def test_multistage_provider_tracks_regions_and_emits_evidence_gated_segments(tmp_path):
    first = _frame(tmp_path, "first.png", 1.0)
    second = _frame(tmp_path, "second.png", 2.0)
    provider, transport = _provider(
        tmp_path,
        [
            {"regions": [{"box_2d": [600, 100, 900, 900]}]},
            {"regions": [{"box_2d": [600, 100, 900, 900]}]},
            {"text_type": "subtitle"},
            {"text_verbatim": "原文字幕", "transcription_status": "readable"},
        ],
    )

    segments = provider.detect_text([first, second])

    assert isinstance(provider, OCRProvider)
    assert [segment.text for segment in segments] == ["原文字幕", "原文字幕"]
    assert len(transport.calls) == 4
    assert [entry["stage"] for entry in provider.provenance] == [
        "stage_a_region_detection",
        "stage_a_region_detection",
        "stage_b_classification",
        "stage_c_crop_transcription",
    ]
    assert segments[0].metadata["stage_provenance"] == ["A", "tracking", "B", "C"]
    assert segments[0].metadata["track_id"] == segments[1].metadata["track_id"]
    assert provider.rejected_regions == []
    assert provider.review_regions == []


def test_multistage_provider_records_rejected_and_review_artifacts(tmp_path):
    frame = _frame(tmp_path, "frame.png", 1.0)
    rejected, _ = _provider(
        tmp_path,
        [{"regions": [{"box_2d": [100, 100, 200, 200]}]}, {"text_type": "scene_text"}],
    )
    assert rejected.detect_text([frame]) == []
    assert rejected.rejected_regions[0]["status"] == "not_output_text_type"

    review, _ = _provider(
        tmp_path,
        [
            {"regions": [{"box_2d": [100, 100, 200, 200]}]},
            {"text_type": "subtitle"},
            {"text_verbatim": "", "transcription_status": "ambiguous"},
        ],
    )
    assert review.detect_text([frame]) == []
    assert review.review_regions[0]["status"] == "ambiguous_transcription"


def test_multistage_provider_verification_and_schema_validation(tmp_path):
    frame = _frame(tmp_path, "frame.png", 1.0)
    provider, _ = _provider(
        tmp_path,
        [
            {"regions": [{"box_2d": [100, 100, 200, 200]}]},
            {"text_type": "subtitle"},
            {"text_verbatim": "原文", "transcription_status": "readable"},
            {"text_verbatim": "khác", "transcription_status": "readable"},
        ],
        verification_enabled=True,
    )
    assert provider.detect_text([frame]) == []
    assert provider.review_regions[0]["status"] == "verification_failed"

    invalid, _ = _provider(tmp_path, [{"regions": [{"box_2d": [0, 0, 0, 1]}]}])
    with pytest.raises(ProviderError):
        invalid.detect_text([frame])


def test_factory_registers_multistage_provider_with_shared_runtime():
    config = {
        "defaults": {"ocr": "gemini_vision_multistage"},
        "providers": {"ocr": {"gemini_vision_multistage": {}}},
    }
    factory = ProviderFactory(config=config, gemini_api_key="test")
    provider = factory.create_ocr_provider()
    assert isinstance(provider, GeminiMultiStageOCRProvider)
    assert provider.runtime is factory.gemini_runtime
