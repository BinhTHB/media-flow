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


# ── New G.5 candidate-explosion fix tests ────────────────────────────


def test_deduplicate_merges_frame_like_candidates():
    provider = GeminiDirectVideoOCRProvider()
    candidates = [
        {"candidate_id": "c1", "start_seconds": 0.0, "end_seconds": 1.5, "representative_box_2d": [100, 100, 200, 200]},
        {"candidate_id": "c2", "start_seconds": 1.4, "end_seconds": 3.0, "representative_box_2d": [105, 105, 205, 205]},
        {"candidate_id": "c3", "start_seconds": 2.9, "end_seconds": 4.0, "representative_box_2d": [102, 102, 202, 202]},
    ]
    result, groups = provider._deduplicate_candidates(candidates, duration=10.0)
    assert len(result) == 1
    assert result[0]["start_seconds"] == 0.0
    assert result[0]["end_seconds"] == 4.0
    assert len(groups) == 1
    assert groups[0]["count"] == 3


def test_deduplicate_does_not_merge_different_spatial_regions():
    provider = GeminiDirectVideoOCRProvider()
    candidates = [
        {"candidate_id": "c1", "start_seconds": 0.0, "end_seconds": 2.0, "representative_box_2d": [100, 100, 200, 200]},
        {"candidate_id": "c2", "start_seconds": 0.0, "end_seconds": 2.0, "representative_box_2d": [800, 800, 900, 900]},
    ]
    result, groups = provider._deduplicate_candidates(candidates, duration=10.0)
    assert len(result) == 2
    assert len(groups) == 2


def test_deduplicate_does_not_merge_large_temporal_gap():
    provider = GeminiDirectVideoOCRProvider()
    candidates = [
        {"candidate_id": "c1", "start_seconds": 0.0, "end_seconds": 1.0, "representative_box_2d": [100, 100, 200, 200]},
        {"candidate_id": "c2", "start_seconds": 3.0, "end_seconds": 4.0, "representative_box_2d": [100, 100, 200, 200]},
    ]
    result, groups = provider._deduplicate_candidates(candidates, duration=10.0)
    assert len(result) == 2


def test_validate_candidates_skips_invalid_timestamps():
    provider = GeminiDirectVideoOCRProvider()
    payload = {
        "candidates": [
            {"candidate_id": "a", "start_seconds": -1, "end_seconds": 2, "representative_box_2d": [1, 1, 2, 2]},
            {"candidate_id": "b", "start_seconds": 1, "end_seconds": 0, "representative_box_2d": [1, 1, 2, 2]},
            {"candidate_id": "c", "start_seconds": 1, "end_seconds": 2, "representative_box_2d": [1, 1, 2, 2]},
            {"candidate_id": "d", "start_seconds": 1, "end_seconds": 12, "representative_box_2d": [1, 1, 2, 2]},
        ]
    }
    result = provider._validate_candidates(payload, duration=10.0)
    assert len(result) == 1
    assert result[0]["candidate_id"] == "c"


def test_compute_request_plan_conservative_estimates():
    provider = GeminiDirectVideoOCRProvider(max_events_per_track=20)
    accepted = [{"candidate_id": "a"}, {"candidate_id": "b"}]
    plan = provider._compute_request_plan(accepted, duration=10.0)
    assert plan["v3_requests"] == 2
    assert plan["v4_requests"] == 40
    assert plan["v5_requests"] == 40
    assert plan["translation_requests"] == 40
    assert plan["fixed_requests"] == 4
    assert plan["minimum_total_requests"] == 12
    assert plan["maximum_total_requests"] == 126


def test_candidate_explosion_fail_fast_raw():
    provider = GeminiDirectVideoOCRProvider(max_raw_candidates=2)
    payload = {
        "candidates": [
            {"candidate_id": "a", "start_seconds": 0, "end_seconds": 1, "representative_box_2d": [1, 1, 2, 2]},
            {"candidate_id": "b", "start_seconds": 1, "end_seconds": 2, "representative_box_2d": [1, 1, 2, 2]},
            {"candidate_id": "c", "start_seconds": 2, "end_seconds": 3, "representative_box_2d": [1, 1, 2, 2]},
        ]
    }
    candidates = provider._validate_candidates(payload, duration=10.0)
    with pytest.raises(ProviderError, match="CANDIDATE_EXPLOSION"):
        if len(candidates) > provider.max_raw_candidates:
            raise ProviderError(
                f"CANDIDATE_EXPLOSION: V1 raw={len(candidates)} > max={provider.max_raw_candidates}"
            )


def test_candidate_explosion_fail_fast_deduplicated():
    provider = GeminiDirectVideoOCRProvider(max_deduplicated_candidates=1)
    # These candidates have low IoU (different spatial bands) so they won't merge
    candidates = [
        {"candidate_id": "a", "start_seconds": 0, "end_seconds": 1, "representative_box_2d": [1, 1, 2, 2]},
        {"candidate_id": "b", "start_seconds": 1.2, "end_seconds": 2, "representative_box_2d": [800, 800, 900, 900]},
    ]
    deduped, _ = provider._deduplicate_candidates(candidates, duration=10.0)
    with pytest.raises(ProviderError, match="CANDIDATE_EXPLOSION"):
        if len(deduped) > provider.max_deduplicated_candidates:
            raise ProviderError(
                f"CANDIDATE_EXPLOSION: deduplicated={len(deduped)} > max={provider.max_deduplicated_candidates}"
            )
def test_candidate_explosion_fail_fast_accepted_tracks():
    provider = GeminiDirectVideoOCRProvider(max_accepted_subtitle_tracks=1)
    candidates = [
        {"candidate_id": "a", "start_seconds": 0, "end_seconds": 1, "representative_box_2d": [1, 1, 2, 2]},
        {"candidate_id": "b", "start_seconds": 1, "end_seconds": 2, "representative_box_2d": [1, 1, 2, 2]},
    ]
    classifications = {
        "classifications": [
            {"candidate_id": "a", "classification": "dialogue_subtitle", "decision": "accepted"},
            {"candidate_id": "b", "classification": "dialogue_subtitle", "decision": "accepted"},
        ]
    }
    accepted, _rejected, _review = provider._validate_classifications(classifications, candidates)
    with pytest.raises(ProviderError, match="CANDIDATE_EXPLOSION"):
        if len(accepted) > provider.max_accepted_subtitle_tracks:
            raise ProviderError(
                f"CANDIDATE_EXPLOSION: accepted={len(accepted)} > max={provider.max_accepted_subtitle_tracks}"
            )


def test_request_plan_fail_fast_max_total_requests():
    provider = GeminiDirectVideoOCRProvider(max_total_stage_requests=10)
    accepted = [{"candidate_id": "a"}, {"candidate_id": "b"}, {"candidate_id": "c"}]
    plan = provider._compute_request_plan(accepted, duration=10.0)
    with pytest.raises(ProviderError, match="CANDIDATE_EXPLOSION"):
        if plan["minimum_total_requests"] > provider.max_total_stage_requests:
            raise ProviderError(
                f"CANDIDATE_EXPLOSION: planned_requests={plan['minimum_total_requests']} > max={provider.max_total_stage_requests}"
            )


def test_token_budget_check_in_request():
    provider = GeminiDirectVideoOCRProvider(max_counted_input_tokens_per_job=5)
    captured = {}

    class Client:
        def count_tokens(self, request):
            captured["tokens"] = 100
            return 100

        def generate_content(self, request):
            return _Response()

    import pytest
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(provider.runtime, "get_client", lambda *args, **kwargs: Client())
    with pytest.raises(ProviderError, match="TOKEN_BUDGET_EXCEEDED"):
        provider._request(
            "v1_discovery", "test", provider._discovery_schema(),
            "uri", "video/mp4", 6, duration_seconds=10,
        )
    monkeypatch.undo()


def test_job_timeout_raises():
    provider = GeminiDirectVideoOCRProvider(job_timeout_seconds=0.01, no_progress_timeout_seconds=10)
    provider._job_start_time = 0.0
    with pytest.raises(ProviderError, match="JOB_TIMEOUT"):
        provider._check_job_timeout()


def test_no_progress_timeout_raises():
    provider = GeminiDirectVideoOCRProvider(job_timeout_seconds=1000, no_progress_timeout_seconds=0.01)
    now = __import__("time").monotonic()
    provider._job_start_time = now
    provider._last_progress_time = now - 1.0
    with pytest.raises(ProviderError, match="NO_PROGRESS_TIMEOUT"):
        provider._check_job_timeout()


def test_checkpoints_written_after_stages():
    provider = GeminiDirectVideoOCRProvider()
    provider._checkpoint("candidates_raw", {"candidates": []})
    assert "checkpoints" in provider.artifacts
    assert "candidates_raw" in provider.artifacts["checkpoints"]


def test_v2_returns_accepted_rejected_review():
    provider = GeminiDirectVideoOCRProvider()
    candidates = [
        {"candidate_id": "a", "start_seconds": 0, "end_seconds": 1, "representative_box_2d": [1, 1, 2, 2]},
        {"candidate_id": "b", "start_seconds": 1, "end_seconds": 2, "representative_box_2d": [1, 1, 2, 2]},
        {"candidate_id": "c", "start_seconds": 2, "end_seconds": 3, "representative_box_2d": [1, 1, 2, 2]},
    ]
    classifications = {
        "classifications": [
            {"candidate_id": "a", "classification": "dialogue_subtitle", "decision": "accepted"},
            {"candidate_id": "b", "classification": "watermark", "decision": "rejected"},
            {"candidate_id": "c", "classification": "review_required", "decision": "accepted"},
        ]
    }
    accepted, rejected, review = provider._validate_classifications(classifications, candidates)
    assert len(accepted) == 1
    assert accepted[0]["candidate_id"] == "a"
    assert len(rejected) == 1
    assert rejected[0]["candidate_id"] == "b"
    assert len(review) == 1
    assert review[0]["candidate_id"] == "c"


def test_v3_namespaces_duplicate_model_event_ids_by_candidate():
    provider = GeminiDirectVideoOCRProvider()
    first = provider._validate_events(
        {"candidate_id": "a", "events": [{"event_id": "0", "start_seconds_in_source": 0.2, "end_seconds_in_source": 0.8}]},
        {"candidate_id": "a", "representative_box_2d": [1, 1, 2, 2]},
        1.0, 3.0, 10.0,
    )
    second = provider._validate_events(
        {"candidate_id": "b", "events": [{"event_id": "0", "start_seconds_in_source": 0.2, "end_seconds_in_source": 0.8}]},
        {"candidate_id": "b", "representative_box_2d": [1, 1, 2, 2]},
        1.0, 3.0, 10.0,
    )

    assert first[0]["event_id"] == "a:0"
    assert second[0]["event_id"] == "b:0"
    assert first[0]["start_seconds_in_source"] == 1.2


def test_v3_preserves_source_absolute_times_outside_clip_relative_range():
    provider = GeminiDirectVideoOCRProvider()
    events = provider._validate_events(
        {"candidate_id": "a", "events": [{"event_id": "0", "start_seconds_in_source": 5.2, "end_seconds_in_source": 5.8}]},
        {"candidate_id": "a", "representative_box_2d": [1, 1, 2, 2]},
        5.0, 7.0, 10.0,
    )

    assert events[0]["start_seconds_in_source"] == 5.2
    assert events[0]["end_seconds_in_source"] == 5.8
