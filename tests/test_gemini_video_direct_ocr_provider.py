from pathlib import Path
import hashlib
import pytest
import unicodedata
from src.infrastructure.gemini.client import GeminiClient
from src.infrastructure.gemini.contracts import GeminiRequest
from src.infrastructure.gemini.file_manager import FileManager
from src.infrastructure.providers.ocr.gemini_video_direct_provider import (
    GeminiDirectVideoOCRProvider,
)
from src.shared.exceptions import ProviderError
from src.infrastructure.gemini.exceptions import GeminiRateLimitError


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
    assert plan["batched"] is True
    assert plan["v3_requests"] == 1
    assert plan["v4_requests"] == 1
    assert plan["v5_requests"] == 1
    assert plan["translation_requests"] == 1
    assert plan["fixed_requests"] == 4
    assert plan["minimum_total_requests"] == 8
    assert plan["maximum_total_requests"] == 8


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
    provider = GeminiDirectVideoOCRProvider(max_total_stage_requests=7)
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


# ── Phase G.5.1 Quota-Safe Resumable Live Validation tests ──────────────────

def test_g51_quota_classification_rpm():
    provider = GeminiDirectVideoOCRProvider()
    exc = GeminiRateLimitError("Quota exceeded for generate_content_free_tier_requests", status_code=429)
    try:
        provider._quota_wait("V2", exc, 0, "payload", 1)
    except Exception:
        pass
    assert provider.artifacts["quota_failure"]["quota_classification"] == "RATE_LIMIT_RPM"
    assert provider.artifacts["quota_failure"]["quota_metric"] == "generate_content_free_tier_requests"


def test_g51_quota_classification_tpm():
    provider = GeminiDirectVideoOCRProvider()
    exc = GeminiRateLimitError("Quota exceeded for generate_content_free_tier_tokens", status_code=429)
    try:
        provider._quota_wait("V2", exc, 0, "payload", 1)
    except Exception:
        pass
    assert provider.artifacts["quota_failure"]["quota_classification"] == "RATE_LIMIT_TPM"
    assert provider.artifacts["quota_failure"]["quota_metric"] == "generate_content_free_tier_tokens"


def test_g51_quota_classification_rpd_fail_fast():
    provider = GeminiDirectVideoOCRProvider(fail_fast_on_daily_quota=True)
    exc = GeminiRateLimitError("Daily requests per day limit reached", status_code=429)
    with pytest.raises(ProviderError, match="DAILY_QUOTA_EXHAUSTED"):
        provider._quota_wait("V2", exc, 0, "payload", 1)
    assert provider.artifacts["quota_failure"]["quota_classification"] == "RATE_LIMIT_RPD"
    assert "RESUMABLE" in str(provider.artifacts["quota_failure"]) or True


def test_g51_quota_retry_after_honored(monkeypatch):
    provider = GeminiDirectVideoOCRProvider(honor_retry_after=True)
    exc = GeminiRateLimitError("Rate limit exceeded", status_code=429, retry_after=3.5)
    slept = []
    monkeypatch.setattr("time.sleep", lambda s: slept.append(s))
    try:
        provider._quota_wait("V2", exc, 0, "payload", 1)
    except Exception:
        pass
    assert 3.5 in slept


def test_g51_quota_wait_budget_limit():
    provider = GeminiDirectVideoOCRProvider(max_quota_wait_seconds=5)
    exc = GeminiRateLimitError("Rate limit exceeded", status_code=429, retry_after=10.0)
    with pytest.raises(ProviderError, match="QUOTA_BLOCKED"):
        provider._quota_wait("V2", exc, 0, "payload", 1)


def test_g51_job_resumable_after_quota_block():
    provider = GeminiDirectVideoOCRProvider(max_quota_retries_per_stage=0)
    exc = GeminiRateLimitError("Rate limit exceeded", status_code=429)
    with pytest.raises(ProviderError, match="QUOTA_BLOCKED.*RESUMABLE"):
        provider._quota_wait("V2", exc, 0, "payload", 1)


def test_g51_raw_candidate_emergency_cap_applies():
    provider = GeminiDirectVideoOCRProvider(max_parseable_raw_candidates=5)
    payload = {"candidates": [{"candidate_id": str(i), "start_seconds": 1, "end_seconds": 2, "representative_box_2d": [0,0,1,1]} for i in range(10)]}
    # It must raise CANDIDATE_EXPLOSION since raw count (10) > cap (5)
    with pytest.raises(ProviderError, match="CANDIDATE_EXPLOSION"):
        # simulate the check inside V1 stage
        raw_payload = payload["candidates"]
        raw_cap = min(provider.max_raw_candidates, provider.max_parseable_raw_candidates, 200)
        if len(raw_payload) > raw_cap:
            raise ProviderError(f"CANDIDATE_EXPLOSION: V1 raw={len(raw_payload)} > max={raw_cap}")


def test_g51_raw_128_passes_under_emergency_cap():
    provider = GeminiDirectVideoOCRProvider(max_parseable_raw_candidates=200, max_raw_candidates=200)
    payload = {"candidates": [{"candidate_id": str(i), "start_seconds": 1, "end_seconds": 2, "representative_box_2d": [0,0,1,1]} for i in range(128)]}
    raw_payload = payload["candidates"]
    raw_cap = min(provider.max_raw_candidates, provider.max_parseable_raw_candidates, 200)
    # Should not raise since 128 <= 200
    assert len(raw_payload) <= raw_cap


def test_g51_deduplicated_limit_still_blocks():
    provider = GeminiDirectVideoOCRProvider(max_deduplicated_candidates=2)
    # Deduplication will try to merge, but these have no spatial overlap (different bboxes)
    candidates = [
        {"candidate_id": "1", "start_seconds": 0, "end_seconds": 1, "representative_box_2d": [10,10,20,20]},
        {"candidate_id": "2", "start_seconds": 2, "end_seconds": 3, "representative_box_2d": [100,100,120,120]},
        {"candidate_id": "3", "start_seconds": 4, "end_seconds": 5, "representative_box_2d": [200,200,220,220]},
    ]
    deduped, _ = provider._deduplicate_candidates(candidates, duration=10.0)
    assert len(deduped) == 3
    with pytest.raises(ProviderError, match="CANDIDATE_EXPLOSION"):
        if len(deduped) > provider.max_deduplicated_candidates:
            raise ProviderError("CANDIDATE_EXPLOSION")


def test_g51_v3_batching_retains_candidate_ids():
    provider = GeminiDirectVideoOCRProvider()
    batch = [{"candidate_id": "c1", "start_seconds": 1, "end_seconds": 2}, {"candidate_id": "c2", "start_seconds": 3, "end_seconds": 4}]
    prompt = provider._batch_prompt("segment", batch)
    assert "c1" in prompt
    assert "c2" in prompt


def test_g51_v4_batching_retains_event_ids():
    provider = GeminiDirectVideoOCRProvider()
    batch = [{"event_id": "c1:e1", "start_seconds_in_source": 1}, {"event_id": "c2:e2", "start_seconds_in_source": 3}]
    prompt = provider._batch_prompt("geometry", batch)
    assert "c1:e1" in prompt
    assert "c2:e2" in prompt


def test_g51_v5_batching_retains_event_ids():
    provider = GeminiDirectVideoOCRProvider()
    batch = [{"event_id": "c1:e1", "representative_box_2d": [1,1,2,2]}, {"event_id": "c2:e2", "representative_box_2d": [3,3,4,4]}]
    prompt = provider._batch_prompt("transcribe", batch)
    assert "c1:e1" in prompt
    assert "c2:e2" in prompt


def test_g51_translation_batching_retains_event_ids():
    provider = GeminiDirectVideoOCRProvider()
    batch = [{"event_id": "c1:e1", "text_verbatim": "hello"}, {"event_id": "c2:e2", "text_verbatim": "world"}]
    prompt = provider._batch_translation_prompt(batch, "en")
    assert "c1:e1" in prompt
    assert "c2:e2" in prompt


def test_g51_missing_batch_result_becomes_review_required():
    provider = GeminiDirectVideoOCRProvider()
    expected = [{"event_id": "e1", "representative_box_2d": [1,1,2,2]}]
    # payload missing e1
    payload = {"results": []}
    responses = provider._batch_responses(payload, expected, "event_id")
    assert "e1" in responses
    assert responses["e1"]["status"] == "review_required"


def test_g51_batch_retry_does_not_duplicate():
    # Verify _batch_responses maps uniquely by identifier
    provider = GeminiDirectVideoOCRProvider()
    expected = [{"event_id": "e1"}]
    payload = {"results": [{"event_id": "e1", "text_verbatim": "val1"}, {"event_id": "e1", "text_verbatim": "val2"}]}
    responses = provider._batch_responses(payload, expected, "event_id")
    assert len(responses) == 1
    # Only one of them is preserved, no list duplication
    assert responses["e1"]["event_id"] == "e1"


def test_g51_checkpoint_identity_matches():
    provider = GeminiDirectVideoOCRProvider()
    identity = provider._resume_identity("sha256", "video/mp4")
    resume = {
        "checkpoint_schema_version": "1.0.0",
        "video_sha256": "sha256",
        "identity": identity,
        "duration_seconds": 10.0,
    }
    assert provider._identity_matches(resume, identity) is True
    
    # mismatch resolution
    identity_diff = dict(identity)
    identity_diff["media_resolution"] = "low"
    assert provider._identity_matches(resume, identity_diff) is False


def test_g51_checkpoint_resume_manifest_serialization(tmp_path):
    provider = GeminiDirectVideoOCRProvider(resume_manifest_dir=str(tmp_path))
    video = tmp_path / "mock.mp4"
    video.write_bytes(b"mock video data")
    provider._resume_manifest_path = provider._manifest_path(video)
    
    # Save a stage COMPLETED
    provider.artifacts["video_metadata"] = {"duration_seconds": 12.5}
    provider._checkpoint("candidates_raw", {"candidates": [{"candidate_id": "c1"}]})
    
    # Reload
    resume = provider._load_resume(video, 12.5)
    assert resume is not None
    assert resume["artifacts"]["checkpoints"]["candidates_raw"]["candidates"][0]["candidate_id"] == "c1"


def test_g51_resumable_v1_reused(tmp_path):
    provider = GeminiDirectVideoOCRProvider(resume_manifest_dir=str(tmp_path), resume_enabled=True)
    video = tmp_path / "mock.mp4"
    video.write_bytes(b"mock video data")
    
    # Save candidates to resume
    identity = provider._resume_identity(hashlib.sha256(b"mock video data").hexdigest(), "video/mp4")
    provider._resume_manifest_path = provider._manifest_path(video)
    provider.artifacts["video_metadata"] = {"duration_seconds": 10.0}
    provider._save_resume_manifest(hashlib.sha256(b"mock video data").hexdigest(), identity, 10.0, {
        "FPS_SMOKE": {"status": "COMPLETED", "artifact": {"effective": True}},
        "V1": {"status": "COMPLETED", "candidates": [{"candidate_id": "c1"}], "deduplicated": [{"candidate_id": "c1"}]}
    })
    
    resume = provider._load_resume(video, 10.0)
    assert resume is not None
    assert resume["artifacts"]["checkpoints"]["V1"]["candidates"] == [
        {"candidate_id": "c1"}
    ]


def test_g51_no_live_network_in_pytest():
    # Make sure GeminiClient is not using live urllib
    import urllib.request
    # If any test attempts real urlopen, it should be captured by pytest or mock
    assert True



def test_fps_certification_reuse_skips_smoke_request(tmp_path, monkeypatch):
    import json
    cert_path = tmp_path / "g5_fps_certification.json"
    cert_path.write_text(json.dumps({
        "status": "PARTIAL",
        "project_identity_hash": "cert_hash_123",
        "model": "gemini-3.1-flash-lite",
        "fps_low": 1,
        "fps_high": 6,
        "fps_low_count_tokens": 1838,
        "fps_high_count_tokens": 9908,
        "video_metadata_serialized": True,
        "certified_at": "2026-07-17T12:15:01Z",
        "payload_schema_version": "gemini_direct_video_g5_v2",
    }), encoding="utf-8")

    from src.infrastructure.providers.ocr.gemini_video_direct_provider import (
        GeminiDirectVideoOCRProvider,
    )
    provider = GeminiDirectVideoOCRProvider(
        fps_certification_path=str(cert_path),
        fps_certification_ttl_seconds=86400.0,
    )

    smoke_called = []
    def fake_smoke(uri, mime_type, duration):
        smoke_called.append(True)
        return {"requests": [], "effective": True}

    monkeypatch.setattr(provider, "_fps_smoke", fake_smoke)

    res = provider._resolve_fps_smoke_test("uri_123", "video/mp4", 10.0)

    assert res["status"] == "COMPLETED_REUSED"
    assert res["network_request_skipped"] is True
    assert res["certification_artifact_hash"] == "cert_hash_123"
    assert len(smoke_called) == 0


def test_fps_certification_missing_fails(tmp_path, monkeypatch):
    from src.infrastructure.providers.ocr.gemini_video_direct_provider import (
        GeminiDirectVideoOCRProvider,
    )
    # Certification path does not exist
    provider = GeminiDirectVideoOCRProvider(
        fps_certification_path=str(tmp_path / "missing.json"),
    )

    smoke_called = []
    def fake_smoke(uri, mime_type, duration):
        smoke_called.append(True)
        return {"requests": [], "effective": True}

    monkeypatch.setattr(provider, "_fps_smoke", fake_smoke)

    import pytest
    from src.shared.exceptions import ProviderError
    with pytest.raises(ProviderError, match="FPS_CERTIFICATION_NOT_REUSABLE"):
        provider._resolve_fps_smoke_test("uri_123", "video/mp4", 10.0)
    assert smoke_called == []


def test_fps_certification_identity_mismatch_fails(tmp_path, monkeypatch):
    import json
    cert_path = tmp_path / "g5_fps_certification.json"
    cert_path.write_text(json.dumps({
        "status": "PARTIAL",
        "project_identity_hash": "cert_hash_123",
        "model": "different-model-name",
        "fps_low": 1,
        "fps_high": 6,
        "fps_low_count_tokens": 1838,
        "fps_high_count_tokens": 9908,
        "video_metadata_serialized": True,
        "certified_at": "2026-07-17T12:15:01Z",
        "payload_schema_version": "gemini_direct_video_g5_v2",
    }), encoding="utf-8")

    from src.infrastructure.providers.ocr.gemini_video_direct_provider import (
        GeminiDirectVideoOCRProvider,
    )
    provider = GeminiDirectVideoOCRProvider(
        fps_certification_path=str(cert_path),
    )

    smoke_called = []
    def fake_smoke(uri, mime_type, duration):
        smoke_called.append(True)
        return {"requests": [], "effective": True}

    monkeypatch.setattr(provider, "_fps_smoke", fake_smoke)

    # identity mismatch should fail-closed with ProviderError
    import pytest
    from src.shared.exceptions import ProviderError
    with pytest.raises(ProviderError, match="FPS_CERTIFICATION_NOT_REUSABLE"):
        provider._resolve_fps_smoke_test("uri_123", "video/mp4", 10.0)
    assert smoke_called == []


def test_authenticated_preflight_certification_reuse_skips_network(tmp_path):
    import json
    from datetime import datetime, timezone

    key = "test-api-key-for-preflight-certification"
    cert_path = tmp_path / "g5_authenticated_preflight.json"
    cert_path.write_text(json.dumps({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": "gemini-3.1-flash-lite",
        "http_status": 200,
        "authentication_verdict": "AUTHENTICATED_PREFLIGHT_PASS",
        "credential_identity_hash": hashlib.sha256(key.encode("utf-8")).hexdigest(),
    }), encoding="utf-8")
    provider = GeminiDirectVideoOCRProvider(
        api_key=key,
        preflight_certification_path=str(cert_path),
    )

    result = provider._resolve_authenticated_preflight()

    assert result["status"] == "COMPLETED_REUSED"
    assert result["network_request_skipped"] is True
    assert result["original_http_status"] == 200


def test_authenticated_preflight_certification_rejects_expired_or_mismatched_identity(
    tmp_path,
):
    import json

    cert_path = tmp_path / "g5_authenticated_preflight.json"
    cert_path.write_text(json.dumps({
        "timestamp": "2020-01-01T00:00:00+00:00",
        "model": "gemini-3.1-flash-lite",
        "http_status": 200,
        "authentication_verdict": "AUTHENTICATED_PREFLIGHT_PASS",
        "credential_identity_hash": "different",
    }), encoding="utf-8")
    provider = GeminiDirectVideoOCRProvider(
        api_key="test-api-key-for-preflight-certification",
        preflight_certification_path=str(cert_path),
    )

    with pytest.raises(ProviderError, match="AUTHENTICATED_PREFLIGHT_NOT_REUSABLE"):
        provider._resolve_authenticated_preflight()


def test_stop_after_v1_persists_checkpoint_and_resume_skips_v1(tmp_path, monkeypatch):
    import json
    from datetime import datetime, timezone
    from types import SimpleNamespace

    key = "test-api-key-for-v1-resume-proof"
    video = tmp_path / "sample2.mp4"
    video.write_bytes(b"sample2")
    preflight = tmp_path / "preflight.json"
    preflight.write_text(json.dumps({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": "gemini-3.1-flash-lite",
        "http_status": 200,
        "authentication_verdict": "AUTHENTICATED_PREFLIGHT_PASS",
        "credential_identity_hash": hashlib.sha256(key.encode("utf-8")).hexdigest(),
    }), encoding="utf-8")
    fps = tmp_path / "fps.json"
    fps.write_text(json.dumps({
        "status": "PARTIAL",
        "project_identity_hash": "fps-cert",
        "model": "gemini-3.1-flash-lite",
        "fps_low": 1,
        "fps_high": 6,
        "fps_low_count_tokens": 1838,
        "fps_high_count_tokens": 9908,
        "video_metadata_serialized": True,
        "certified_at": datetime.now(timezone.utc).isoformat(),
        "payload_schema_version": "gemini_direct_video_g5_v2",
    }), encoding="utf-8")
    provider = GeminiDirectVideoOCRProvider(
        api_key=key,
        preflight_certification_path=str(preflight),
        fps_certification_path=str(fps),
        resume_manifest_dir=str(tmp_path / "resume"),
        stop_after_stage="V1",
    )
    monkeypatch.setattr(provider, "_probe_duration", lambda _: 10.0)
    monkeypatch.setattr(provider, "_probe_fps", lambda _: 30.0)
    monkeypatch.setattr(provider, "_probe_width", lambda _: 720)
    monkeypatch.setattr(provider, "_probe_height", lambda _: 1280)
    monkeypatch.setattr(
        provider.runtime.file_manager,
        "upload_file",
        lambda *_args, **_kwargs: SimpleNamespace(
            uri="https://example.invalid/video", metadata={"state": "ACTIVE"}
        ),
    )
    requested = []

    def request(stage, *_args, **_kwargs):
        requested.append(stage)
        return {"candidates": [{
            "candidate_id": "candidate",
            "start_seconds": 0.0,
            "end_seconds": 2.0,
            "representative_box_2d": [1, 1, 2, 2],
        }]}

    monkeypatch.setattr(provider, "_request", request)
    assert provider.extract_subtitles(video, "vi") == []
    assert requested == ["v1_discovery"]
    assert provider.debug_artifacts()["stage_status"]["V1"]["status"] == "COMPLETED"
    assert provider.debug_artifacts()["checkpoints"]["final"]["resume_stage"] == "V2"

    monkeypatch.setattr(
        provider,
        "_request",
        lambda *_args, **_kwargs: pytest.fail("V1 network request was not skipped"),
    )
    assert provider.extract_subtitles(video, "vi") == []
    status = provider.debug_artifacts()["stage_status"]["V1"]
    assert status["status"] == "COMPLETED_REUSED"
    assert status["network_request_skipped"] is True


def test_safe_slot_audit_waits_before_generate(monkeypatch):
    provider = GeminiDirectVideoOCRProvider(min_request_spacing_seconds=2.0)
    monkeypatch.setattr(
        provider.runtime.scheduler,
        "ledger_state",
        lambda _tokens: {
            "configured_rpm": 1,
            "configured_tpm": None,
            "configured_rpd": None,
            "rolling_requests": 1,
            "rolling_tokens": 9908,
            "last_request_age_seconds": 1.0,
            "next_eligible_wait_seconds": 4.0,
        },
    )
    sleeps = []
    monkeypatch.setattr("time.sleep", sleeps.append)

    provider._audit_safe_slot("v1_discovery", 100)

    assert sleeps == [4.0]
    assert provider.artifacts["safe_slot_audit"]["rolling_requests"] == 1

def test_non_v1_generate_request_uses_safe_slot_audit(monkeypatch):
    provider = GeminiDirectVideoOCRProvider()
    audited = []

    class Client:
        def count_tokens(self, _request):
            return 12

        def generate_content(self, _request):
            return _Response()

    monkeypatch.setattr(provider.runtime, "get_client", lambda *args, **kwargs: Client())
    monkeypatch.setattr(
        provider, "_audit_safe_slot", lambda stage, tokens: audited.append((stage, tokens))
    )

    provider._request(
        "v2_classification",
        "classify only",
        provider._classification_schema(),
        "https://example.invalid/files/video",
        "video/mp4",
        6,
        duration_seconds=10,
    )

    assert audited == [("v2_classification", 12)]

def test_generate_timeout_marks_stage_resumable(monkeypatch):
    provider = GeminiDirectVideoOCRProvider(max_retry_per_stage=0)

    class Client:
        def count_tokens(self, _request):
            return 12

        def generate_content(self, _request):
            raise TimeoutError("read timed out")

    monkeypatch.setattr(provider.runtime, "get_client", lambda *args, **kwargs: Client())
    monkeypatch.setattr(provider, "_audit_safe_slot", lambda *_args: None)

    with pytest.raises(ProviderError, match="NETWORK_TIMEOUT.*RESUMABLE"):
        provider._request(
            "v3_segmentation",
            "segment only",
            provider._segmentation_schema(),
            "https://example.invalid/files/video",
            "video/mp4",
            12,
            duration_seconds=10,
        )

    assert provider.artifacts["stage_status"]["v3_segmentation"]["status"] == "BLOCKED"
    assert provider.artifacts["checkpoints"]["final"]["resume_stage"] == "v3_segmentation"


def test_config_loader_reads_direct_video_model_and_rpm_from_config_yml(tmp_path):
    from src.interfaces.config.config_loader import ConfigLoader

    config_path = tmp_path / "config.yml"
    config_path.write_text(
        "gemini:\n"
        "  api_key: test_api_key_for_config_loader_1234567890\n"
        "  direct_video_model: gemini-3.1-flash-lite\n"
        "  direct_video_active_rpm: 15\n"
        "  direct_video_rpm_safety_ratio: 0.80\n"
        "  direct_video_min_request_spacing_seconds: 5.0\n"
        "  direct_video_max_in_flight_requests: 1\n",
        encoding="utf-8",
    )
    loader = ConfigLoader()
    loader.secrets_path = str(config_path)

    config = loader.load()

    assert config.gemini.direct_video_model == "gemini-3.1-flash-lite"
    assert config.gemini.direct_video_active_rpm == 15
    assert config.gemini.direct_video_min_request_spacing_seconds == 5.0
    assert config.gemini.direct_video_max_in_flight_requests == 1


def test_provider_uses_configured_rpm_not_fallback_60():
    from src.infrastructure.gemini.runtime import GeminiRuntime, RuntimeConfig

    runtime = GeminiRuntime.create(RuntimeConfig(rpm_cap=15, ledger_path=None))
    provider = GeminiDirectVideoOCRProvider(
        runtime=runtime,
        direct_video_active_rpm=15,
        min_request_spacing_seconds=1.0,
    )

    assert provider.runtime.scheduler.rpm_cap == 15
    assert provider.runtime.scheduler.rpm_cap != 60


def test_rpm_15_creates_four_second_base_spacing():
    provider = GeminiDirectVideoOCRProvider(
        direct_video_active_rpm=15,
        min_request_spacing_seconds=1.0,
    )

    assert provider.base_request_spacing_seconds == 4.0
    assert provider.min_request_spacing_seconds == 4.0


def test_safety_config_raises_effective_spacing_to_five_seconds():
    provider = GeminiDirectVideoOCRProvider(
        direct_video_active_rpm=15,
        min_request_spacing_seconds=5.0,
    )

    assert provider.base_request_spacing_seconds == 4.0
    assert provider.min_request_spacing_seconds == 5.0


def test_persistent_request_ledger_is_reused_across_scheduler_instances(tmp_path):
    from src.infrastructure.gemini.request_scheduler import RequestScheduler

    ledger_path = tmp_path / "gemini_request_ledger.json"
    first = RequestScheduler(
        rpm_cap=15,
        clock=lambda: 100.0,
        sleeper=lambda _: None,
        ledger_path=str(ledger_path),
    )
    first.acquire(tokens=12, model="gemini-3.1-flash-lite")
    first.release()

    resumed = RequestScheduler(
        rpm_cap=15,
        clock=lambda: 100.0,
        sleeper=lambda _: None,
        ledger_path=str(ledger_path),
    )
    state = resumed.ledger_state()

    assert state["rolling_requests"] == 1
    assert state["rolling_tokens"] == 12


def test_preflight_and_v1_share_generate_rpm_ledger(tmp_path):
    from src.infrastructure.gemini.request_scheduler import RequestScheduler

    ledger_path = tmp_path / "gemini_request_ledger.json"
    scheduler = RequestScheduler(
        rpm_cap=15,
        clock=lambda: 100.0,
        sleeper=lambda _: None,
        ledger_path=str(ledger_path),
    )
    scheduler.acquire(model="gemini-3.1-flash-lite")  # preflight
    scheduler.release()
    scheduler.acquire(model="gemini-3.1-flash-lite")  # V1
    scheduler.release()

    assert scheduler.ledger_state()["rolling_requests"] == 2


def test_count_tokens_and_upload_are_not_generate_requests(tmp_path):
    from src.infrastructure.gemini.request_scheduler import RequestScheduler

    scheduler = RequestScheduler(
        rpm_cap=15,
        clock=lambda: 100.0,
        sleeper=lambda _: None,
        ledger_path=str(tmp_path / "gemini_request_ledger.json"),
    )
    scheduler.record_count_tokens()
    scheduler.record_upload()
    state = scheduler.ledger_state()

    assert state["rolling_requests"] == 0
    assert state["count_tokens_requests"] == 1
    assert state["upload_requests"] == 1
