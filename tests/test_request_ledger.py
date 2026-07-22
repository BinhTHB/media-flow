import json
import os
import tempfile
import pytest
from src.infrastructure.gemini.runtime import _create_ledger_callback
from src.infrastructure.gemini.usage_recorder import UsageRecorder, UsageRecord

def test_ledger_writes_cache_entry():
    with tempfile.TemporaryDirectory() as tmp:
        ledger = os.path.join(tmp, "ledger.jsonl")
        callback = _create_ledger_callback(ledger, "job-cache-001")
        record = UsageRecord(
            model="gemini-3.5-flash-lite",
            request_id="req-abc123",
            source_kind="benchmark_subtitle_verifier",
            cache_hit=True,
            total_tokens=2048,
            job_id="job-cache-001",
            success=True,
        )
        callback(record)

        assert os.path.exists(ledger)
        lines = open(ledger).readlines()
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["job_id"] == "job-cache-001"
        assert entry["purpose"] == "benchmark_subtitle_verifier"
        assert entry["model"] == "gemini-3.5-flash-lite"
        assert entry["live_or_cache"] == "cache"
        assert entry["status"] == "success"
        assert entry["tokens"] == 2048
        assert entry["request_id_or_hash"] == "req-abc123"
        assert entry["error_msg"] == ""

def test_ledger_writes_error_entry():
    with tempfile.TemporaryDirectory() as tmp:
        ledger = os.path.join(tmp, "ledger.jsonl")
        callback = _create_ledger_callback(ledger, "job-err-002")
        record = UsageRecord(
            model="gemini-2.5-flash-native-audio-preview-12-2025",
            request_id="req-xyz789",
            source_kind="benchmark_subtitle_verifier",
            cache_hit=False,
            total_tokens=0,
            job_id="job-err-002",
            success=False,
            error_msg="GeminiNetworkError: connection refused",
        )
        callback(record)

        assert os.path.exists(ledger)
        entry = json.loads(open(ledger).readlines()[0])
        assert entry["job_id"] == "job-err-002"
        assert entry["status"] == "error"
        assert entry["tokens"] == 0
        assert "connection refused" in entry["error_msg"]

def test_recorder_callback_wires():
    with tempfile.TemporaryDirectory() as tmp:
        ledger = os.path.join(tmp, "ledger.jsonl")
        callback = _create_ledger_callback(ledger, "job-test-42")
        recorder = UsageRecorder(usage_callback=callback)
        recorder.record(UsageRecord(
            model="gemini-3.5-flash-lite",
            source_kind="ocr_verifier",
            cache_hit=False,
            total_tokens=512,
            job_id="job-test-42",
            success=True,
            request_id="req-abc123"
        ))
        assert os.path.exists(ledger)
        entry = json.loads(open(ledger).readlines()[0])
        assert entry["job_id"] == "job-test-42"
        assert entry["purpose"] == "ocr_verifier"
        assert entry["status"] == "success"

def test_runtime_uses_configured_ledger_and_job_id(tmp_path):
    from src.infrastructure.gemini.runtime import GeminiRuntime, RuntimeConfig

    ledger = tmp_path / "ledger_job-round20.jsonl"
    runtime = GeminiRuntime.create(RuntimeConfig(api_key="fake", ledger_path=str(ledger)))
    runtime.set_job_id("job-round20")
    runtime.usage_recorder.record(UsageRecord(
        model="gemini-3.5-flash-lite",
        source_kind="translation",
        job_id="job-round20",
        success=True,
        request_id="req-12345"
    ))

    entry = json.loads(ledger.read_text(encoding="utf-8").strip())
    assert entry["job_id"] == "job-round20"
    assert entry["purpose"] == "translation"
    assert runtime.get_client()._job_id == "job-round20"

def test_ledger_writes_image_purpose_with_media_sha256():
    with tempfile.TemporaryDirectory() as tmp:
        ledger = os.path.join(tmp, "ledger.jsonl")
        callback = _create_ledger_callback(ledger, "job-image-999")
        record = UsageRecord(
            model="gemini-3.5-flash-lite",
            request_id=None,
            media_sha256="abcde12345hashvalue",
            source_kind="image",
            cache_hit=True,
            total_tokens=1024,
            job_id="job-image-999",
            success=True,
        )
        callback(record)

        assert os.path.exists(ledger)
        lines = open(ledger).readlines()
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["job_id"] == "job-image-999"
        assert entry["purpose"] == "image"
        assert entry["request_id_or_hash"] == "abcde12345hashvalue"