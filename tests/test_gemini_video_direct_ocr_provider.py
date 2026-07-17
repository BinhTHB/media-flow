from pathlib import Path
import pytest
import unicodedata
from src.infrastructure.gemini.client import GeminiClient
from src.infrastructure.gemini.contracts import GeminiRequest
from src.infrastructure.gemini.file_manager import FileManager
from src.infrastructure.providers.ocr.gemini_video_direct_provider import (
    GeminiDirectVideoOCRProvider,
)
from src.shared.exceptions import ProviderError


class _Response:
    text = '{"candidates": []}'
    cache_hit = False

    class usage:
        @staticmethod
        def to_dict():
            return {"prompt_token_count": 12}


def test_direct_video_request_serializes_video_metadata_fps(monkeypatch):
    provider = GeminiDirectVideoOCRProvider()
    captured = {}

    class Client:
        def count_tokens(self, request):
            captured["count_request"] = request
            return 12

        def generate_content(self, request):
            captured["generate_request"] = request
            return _Response()

    monkeypatch.setattr(provider.runtime, "get_client", lambda *args, **kwargs: Client())

    assert provider._request(
        "v1_discovery",
        "discovery only",
        provider._discovery_schema(),
        "https://example.invalid/files/video",
        "video/mp4",
        6,
        duration_seconds=10,
    ) == {"candidates": []}
    assert captured["count_request"] is captured["generate_request"]
    video_part = captured["count_request"].contents[0]["parts"][1]
    assert video_part["fileData"]["mimeType"] == "video/mp4"
    assert video_part["videoMetadata"]["fps"] == 6
    assert provider.provenance[-1]["input_type"] == "video"
    assert provider.provenance[-1]["image_part_count"] == 0
    assert provider.provenance[-1]["local_ocr_calls"] == 0
    assert provider.provenance[-1]["serialized_payload_sha256"]


def test_direct_video_media_resolution_uses_gemini_rest_enum():
    provider = GeminiDirectVideoOCRProvider(media_resolution="high")

    assert provider._media_resolution_enum() == "MEDIA_RESOLUTION_HIGH"


def test_direct_video_clamps_clip_offsets_and_preserves_source_clip_mapping(monkeypatch):
    provider = GeminiDirectVideoOCRProvider(clip_padding_seconds=0.4)
    captured = {}

    class Client:
        def count_tokens(self, request):
            captured["request"] = request
            return 9

        def generate_content(self, request):
            return _Response()

    monkeypatch.setattr(provider.runtime, "get_client", lambda *args, **kwargs: Client())
    provider._request(
        "v3_segmentation", "intervals only", provider._segmentation_schema(),
        "uri", "video/mp4", 12, start=0.1, end=9.9, duration_seconds=10,
    )
    metadata = captured["request"].contents[0]["parts"][1]["videoMetadata"]
    assert metadata["startOffset"] == "0.100s"
    assert metadata["endOffset"] == "9.900s"
    assert provider.provenance[-1]["clip_start"] == 0.1
    assert provider.provenance[-1]["clip_end"] == 9.9


def test_direct_video_rejects_invalid_and_oversized_boxes():
    provider = GeminiDirectVideoOCRProvider()

    with pytest.raises(ProviderError, match="invalid"):
        provider._validate_box([0, 0, 0, 100])
    with pytest.raises(ProviderError, match="cover safety"):
        provider._validate_box([0, 0, 1000, 1000])


def test_direct_video_rejects_incomplete_or_duplicate_classifications():
    provider = GeminiDirectVideoOCRProvider()
    candidates = [
        {"candidate_id": "a", "start_seconds": 0, "end_seconds": 1, "representative_box_2d": [1, 1, 2, 2]},
        {"candidate_id": "b", "start_seconds": 1, "end_seconds": 2, "representative_box_2d": [1, 1, 2, 2]},
    ]
    with pytest.raises(ProviderError, match="completeness"):
        provider._validate_classifications({"classifications": [
            {"candidate_id": "a", "classification": "dialogue_subtitle", "decision": "accepted"},
        ]}, candidates)
    with pytest.raises(ProviderError, match="duplicate"):
        provider._validate_classifications({"classifications": [
            {"candidate_id": "a", "classification": "other", "decision": "rejected"},
            {"candidate_id": "a", "classification": "other", "decision": "rejected"},
            {"candidate_id": "b", "classification": "other", "decision": "rejected"},
        ]}, candidates)


def test_direct_video_rejects_unreasoned_source_copy_and_cjk_translation():
    provider = GeminiDirectVideoOCRProvider()

    assert provider._validate_translation(
        {"event_id": "event", "translated_text": "原文", "is_source_copy": False},
        "event", "原文",
    ) is None
    assert provider._validate_translation(
        {"event_id": "event", "translated_text": "原文", "is_source_copy": True},
        "event", "khác",
    ) is None
    assert provider._validate_translation(
        {"event_id": "event", "translated_text": "Bản dịch tiếng Việt"},
        "event", "原文",
    ) == "Bản dịch tiếng Việt"


def test_file_manager_reuses_only_verified_matching_remote_file(tmp_path):
    source = tmp_path / "video.mp4"
    source.write_bytes(b"video")

    class Files:
        uploads = 0
        gets = 0

        def upload_file(self, path, *, mime_type):
            self.uploads += 1
            return {"file": {"name": "files/1", "uri": "https://redacted", "mimeType": mime_type, "sizeBytes": 5, "state": "PROCESSING"}}

        def get_file(self, name):
            self.gets += 1
            return {"file": {"name": name, "uri": "https://redacted", "mimeType": "video/mp4", "sizeBytes": 5, "state": "ACTIVE"}}

        def delete_file(self, name):
            pass

    files = Files()
    manager = FileManager(files, enabled=True, poll_interval_seconds=0)
    first = manager.upload_file(str(source), mime_type="video/mp4")
    second = manager.upload_file(str(source), mime_type="video/mp4")
    assert first is second
    assert files.uploads == 1
    assert files.gets >= 2
    assert manager.reuse_count == 1


def test_count_tokens_and_generate_use_identical_canonical_body(monkeypatch):
    payloads = []

    class Transport:
        def send(self, method, url, *, headers, data=None, timeout=60):
            payloads.append((url, data))
            if "countTokens" in url:
                return {"totalTokens": 7}
            return {"candidates": [{"content": {"parts": [{"text": "{}"}]}}], "usageMetadata": {}}

    client = GeminiClient(api_key="test", transport=Transport(), validate_models=False)
    request = GeminiRequest(
        model="gemini-3.1-flash-lite",
        contents=[{"parts": [{"text": "prompt"}]}],
        generation_config={"responseMimeType": "application/json"},
        system_instruction="system",
    )
    client.count_tokens(request)
    client.generate_content(request, use_cache=False)

    import json
    count_body = json.loads(payloads[0][1])
    generate_body = json.loads(payloads[1][1])
    counted_request = count_body["generateContentRequest"]
    assert counted_request.pop("model") == "models/gemini-3.1-flash-lite"
    assert counted_request == generate_body


def test_direct_video_port_rejects_missing_video_path():
    provider = GeminiDirectVideoOCRProvider()

    with pytest.raises(ProviderError, match="does not exist"):
        provider.extract_subtitles(Path("missing.mp4"), "zh")


def test_app_config_loader_reads_direct_video_config_correctly(monkeypatch):
    from src.interfaces.config.config_loader import ConfigLoader

    monkeypatch.setenv("GEMINI_OCR_STRATEGY", "gemini_video_direct")
    monkeypatch.setenv("GEMINI_DIRECT_VIDEO_DISCOVERY_FPS", "8.5")
    config = ConfigLoader().load()
    assert config.gemini.ocr_strategy == "gemini_video_direct"
    assert config.gemini.direct_video_discovery_fps == 8.5



def test_direct_video_mixed_script_rejections():
    provider = GeminiDirectVideoOCRProvider()

    # unicode control character in text
    assert provider._validate_transcription(
        {"event_id": "e", "text_verbatim": "hello\u0000world", "status": "readable"},
        {"event_id": "e"}
    ) is None

    # mixed script: Han script + Latin script in text
    assert provider._validate_transcription(
        {"event_id": "e", "text_verbatim": "字幕 hello", "status": "readable"},
        {"event_id": "e"}
    ) is None

    # clean Latin script only
    assert provider._validate_transcription(
        {"event_id": "e", "text_verbatim": "hello world", "status": "readable"},
        {"event_id": "e"}
    ) == {
        "text_verbatim": "hello world",
        "language": None,
        "script": None,
        "status": "readable",
    }


def test_direct_video_render_eligibility_build():
    provider = GeminiDirectVideoOCRProvider()
    candidates = [
        {"candidate_id": "c1", "start_seconds": 1.0, "end_seconds": 5.0, "representative_box_2d": [10, 10, 20, 20]}
    ]
    accepted_candidates = [
        {"candidate_id": "c1", "start_seconds": 1.0, "end_seconds": 5.0, "representative_box_2d": [10, 10, 20, 20]}
    ]
    events = [
        {"event_id": "e1", "candidate_id": "c1", "start_seconds_in_source": 1.5, "end_seconds_in_source": 4.5}
    ]
    geometry = {
        "e1": {"event_id": "e1", "representative_box_2d": [10, 10, 20, 20], "box_keyframes": []}
    }
    transcriptions = [
        {"event_id": "e1", "text_verbatim": "Hello", "status": "readable"}
    ]
    translations = [
        {"event_id": "e1", "translated_text": "Xin chào"}
    ]

    provider._build_render_eligibility(
        candidates, accepted_candidates, events, geometry, transcriptions, translations, duration=10.0
    )

    eligibility = provider.artifacts["render_eligibility"]["events"]
    assert len(eligibility) == 1
    assert eligibility[0]["render_eligible"] is True
    assert eligibility[0]["event_id"] == "e1"


def test_direct_video_empty_result_handling():
    provider = GeminiDirectVideoOCRProvider()
    provider._empty_artifacts()
    for name in ("classifications", "temporal_events", "geometry",
                  "transcriptions", "translations", "render_eligibility"):
        assert provider.artifacts[name] == {"events": []}
