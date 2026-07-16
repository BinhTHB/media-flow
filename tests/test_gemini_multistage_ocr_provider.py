import json

import pytest
from PIL import Image

from src.application.ports.ocr_provider import OCRProvider
from src.domain.entities.frame import Frame
from src.infrastructure.factory.provider_factory import ProviderFactory
from src.infrastructure.gemini.runtime import GeminiRuntime, RuntimeConfig
from src.infrastructure.providers.ocr.gemini_multistage_ocr_provider import GeminiMultiStageOCRProvider
from src.shared.exceptions import ProviderError


def _response(payload):
    return {"candidates": [{"content": {"parts": [{"text": json.dumps(payload)}]}}], "modelStatus": {"modelStage": "STABLE"}}


class FakeTransport:
    def __init__(self, payloads):
        self.payloads = list(payloads)
        self.calls = []

    def send(self, method, url, *, headers, data=None, timeout=60.0):
        self.calls.append((method, json.loads(data.decode("utf-8")) if data else None))
        if method == "GET":
            return {"name": "models/test", "supportedGenerationMethods": ["generateContent"], "modelStage": "STABLE"}
        if not self.payloads:
            raise AssertionError("Unexpected Gemini request")
        return _response(self.payloads.pop(0))


def _frame(tmp_path, name, timestamp):
    path = tmp_path / name
    Image.new("RGB", (1000, 500), "white").save(path)
    return Frame(timestamp=timestamp, image_path=str(path))


def _provider(tmp_path, payloads, **kwargs):
    runtime = GeminiRuntime.create(RuntimeConfig(api_key="fake", cache_dir=str(tmp_path / "cache"), cache_enabled=False))
    provider = GeminiMultiStageOCRProvider(runtime=runtime, api_key="fake", **kwargs)
    transport = FakeTransport(payloads)
    for model in set(provider.stage_models.values()):
        runtime.get_client(model, api_key="fake")._transport = transport
    return provider, transport


def _region(local_id, box):
    return {"local_id": local_id, "box_2d": box}


def _window(*frames):
    return {"frames": [{"frame_id": frame_id, "regions": regions} for frame_id, regions in frames]}


def _classification(text_type, evidence="position and contrast support the label"):
    return {"text_type": text_type, "evidence": evidence}


def _reading(text, status="readable"):
    return {"text_verbatim": text, "transcription_status": status}


def test_multistage_tracks_regions_into_one_evidence_gated_interval(tmp_path):
    first = _frame(tmp_path, "first.png", 1.0)
    second = _frame(tmp_path, "second.png", 2.0)
    provider, transport = _provider(tmp_path, [
        _window(("frame_0", [_region("local_0", [600, 100, 900, 900])]), ("frame_1", [_region("local_1", [600, 100, 900, 900])])),
        _classification("subtitle"), _reading("????"), _reading("????"),
    ])

    segments = provider.detect_text([first, second])

    assert isinstance(provider, OCRProvider)
    assert [segment.text for segment in segments] == ["????"]
    assert segments[0].start == 1.0
    assert segments[0].end == 3.0
    assert segments[0].metadata["stage_provenance"] == ["A", "tracking", "B", "C"]
    assert segments[0].metadata["validation_evidence"]["stage_b_evidence"]
    assert [entry["stage"] for entry in provider.provenance] == ["stage_a_region_detection", "stage_b_classification", "stage_c_crop_transcription", "stage_c_crop_transcription"]
    assert len(transport.calls) == 4
    assert provider.rejected_regions == []
    assert provider.review_regions == []


def test_multistage_records_rejected_and_review_artifacts(tmp_path):
    frame = _frame(tmp_path, "frame.png", 1.0)
    rejected, _ = _provider(tmp_path, [_window(("frame_0", [_region("local_0", [100, 100, 200, 200])])), _classification("scene_text")])
    assert rejected.detect_text([frame]) == []
    assert rejected.rejected_regions[0]["status"] == "not_output_text_type"
    assert rejected.rejected_regions[0]["text_type"] == "scene_text"

    review, _ = _provider(tmp_path, [_window(("frame_0", [_region("local_0", [100, 100, 200, 200])])), _classification("subtitle"), _reading("", "ambiguous")])
    assert review.detect_text([frame]) == []
    assert review.review_regions[0]["status"] == "ambiguous_transcription"


def test_multistage_uses_one_stage_d_request_on_disagreement(tmp_path):
    first = _frame(tmp_path, "first.png", 1.0)
    second = _frame(tmp_path, "second.png", 2.0)
    provider, transport = _provider(tmp_path, [
        _window(("frame_0", [_region("local_0", [100, 100, 200, 200])]), ("frame_1", [_region("local_1", [100, 100, 200, 200])])),
        _classification("subtitle"), _reading("??"), _reading("kh?c"), _reading("??"),
    ])
    segments = provider.detect_text([first, second])
    assert [segment.text for segment in segments] == ["??"]
    assert segments[0].metadata["validation_evidence"]["stage_d_verified"] is True
    assert segments[0].metadata["stage_provenance"][-1] == "D"
    assert [entry["stage"] for entry in provider.provenance].count("stage_d_verification") == 1
    assert len(transport.calls) == 5


def test_multistage_validates_stage_a_and_stage_b_schema(tmp_path):
    frame = _frame(tmp_path, "frame.png", 1.0)
    invalid, _ = _provider(tmp_path, [{"frames": [{"frame_id": "frame_1", "regions": []}]}])
    with pytest.raises(ProviderError):
        invalid.detect_text([frame])

    missing_evidence, _ = _provider(tmp_path, [_window(("frame_0", [_region("local_0", [100, 100, 200, 200])])), {"text_type": "subtitle"}])
    with pytest.raises(ProviderError):
        missing_evidence.detect_text([frame])


def test_multistage_deduplicates_overlapping_windows_and_uses_consensus_bbox(tmp_path):
    frames = [_frame(tmp_path, f"frame_{index}.png", float(index + 1)) for index in range(4)]
    provider, transport = _provider(tmp_path, [
        _window(("frame_0", [_region("local_0", [100, 100, 200, 200])]), ("frame_1", [_region("local_1", [120, 110, 220, 210])]), ("frame_2", [_region("local_2", [140, 120, 240, 220])])),
        _window(("frame_2", [_region("local_2", [140, 120, 240, 220])]), ("frame_3", [])),
        _classification("subtitle"), _reading("????"), _reading("????"), _reading("????"),
    ], window_size=3, window_overlap=1)
    segments = provider.detect_text(frames)
    assert len(segments) == 1
    assert segments[0].end > segments[0].start
    assert segments[0].bbox == {"left": 110.0, "top": 60.0, "right": 210.0, "bottom": 110.0}
    assert len(transport.calls) == 6


def test_multistage_stores_stage_configuration(tmp_path):
    provider, _ = _provider(tmp_path, [], stage_a_model="model-a", stage_b_model="model-b", stage_c_model="model-c", stage_d_model="model-d", stage_a_options={"temperature": 0.1}, stage_d_options={"temperature": 0.2})
    assert provider.stage_models == {"A": "model-a", "B": "model-b", "C": "model-c", "D": "model-d"}
    assert provider.stage_options["A"] == {"temperature": 0.1}
    assert provider.stage_options["D"] == {"temperature": 0.2}


def test_factory_registers_multistage_provider_with_shared_runtime():
    config = {"defaults": {"ocr": "gemini_vision_multistage"}, "providers": {"ocr": {"gemini_vision_multistage": {}}}}
    factory = ProviderFactory(config=config, gemini_api_key="test")
    provider = factory.create_ocr_provider()
    assert isinstance(provider, GeminiMultiStageOCRProvider)
    assert provider.runtime is factory.gemini_runtime
    assert set(provider.stage_models.values()) == {"gemini-3.1-flash-lite"}

def test_multistage_exposes_serializable_stage_artifacts(tmp_path):
    first = _frame(tmp_path, "first.png", 1.0)
    second = _frame(tmp_path, "second.png", 2.0)
    provider, _ = _provider(tmp_path, [
        _window(
            ("frame_0", [_region("local_0", [100, 100, 200, 200])]),
            ("frame_1", [_region("local_1", [100, 100, 200, 200])]),
        ),
        _classification("subtitle"),
        _reading("first"),
        _reading("second"),
        _reading("first"),
    ])

    accepted = provider.detect_text([first, second])
    artifacts = provider.debug_artifacts()

    assert [segment.text for segment in accepted] == ["first"]
    assert artifacts["region_windows"][0]["window_id"] == "window_0"
    assert len(artifacts["tracks"]) == 1
    assert artifacts["classifications"] == [{
        "track_id": "track_0",
        "text_type": "subtitle",
        "evidence": "position and contrast support the label",
    }]
    assert [result["text_verbatim"] for result in artifacts["crop_ocr_results"]] == [
        "first", "second"
    ]
    assert artifacts["verification_results"][0]["text_verbatim"] == "first"
    assert artifacts["accepted_intervals"][0]["text_verbatim"] == "first"
    assert artifacts["review_regions"] == []
    assert artifacts["rejected_regions"] == []
    json.dumps(artifacts)
