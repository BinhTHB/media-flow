import json
import subprocess
import sys
from pathlib import Path

import pytest

from src.infrastructure.benchmark.gemini_phase_e import (
    BenchmarkValidationError,
    aggregate_asr_metrics,
    aggregate_ocr_metrics,
    bbox_iou,
    character_error_rate,
    classify_failure,
    evaluate_safety_regression,
    sanitize_artifact,
    sha256_file,
    validate_manifest,
    word_error_rate,
)


def _manifest(tmp_path: Path, *, reference=True) -> Path:
    frame, audio = tmp_path / "frame.bin", tmp_path / "audio.wav"
    frame.write_bytes(b"frame")
    audio.write_bytes(b"audio")
    payload = {
        "samples": [{
            "id": "sample1",
            "frames": [{"path": frame.name, "sha256": sha256_file(frame), "timestamp": 0.0}],
            "audio": {"path": audio.name, "sha256": sha256_file(audio)},
            "references": {"ocr": {"detections": []}, "asr": {"segments": []}} if reference else {},
        }]
    }
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    return manifest


def test_exact_cer_iou_and_ocr_false_positive_aggregate():
    assert character_error_rate("abc", "adc") == pytest.approx(1 / 3)
    assert word_error_rate("one two", "one three") == pytest.approx(1 / 2)
    assert bbox_iou([0, 0, 10, 10], [5, 5, 15, 15]) == pytest.approx(25 / 175)
    metrics = aggregate_ocr_metrics(
        {"detections": [{"text": "字幕", "box_2d": [0, 0, 10, 10]}]},
        {"detections": [{"text": "字幕", "box_2d": [0, 0, 10, 10]}, {"text": "noise", "box_2d": [20, 20, 30, 30]}]},
    )
    assert metrics["false_positives"] == 1
    assert metrics["f1"] == pytest.approx(2 / 3)


def test_asr_wer_timestamp_mae_and_key_phrases():
    metrics = aggregate_asr_metrics(
        {"segments": [{"start": 0, "end": 2, "text": "因果 英国 test"}]},
        {"segments": [{"start": 0.5, "end": 2.5, "text": "因果 英国 best"}]},
    )
    assert metrics["wer"] == pytest.approx(1 / 3)
    assert metrics["timestamp_mae"] == pytest.approx(0.5)
    assert metrics["exact_key_phrase_counts"] == {"因果": 1, "英国": 1}


def test_manifest_relative_path_hash_mismatch_and_missing_reference(tmp_path):
    manifest = _manifest(tmp_path, reference=False)
    validated = validate_manifest(manifest)
    assert Path(validated["samples"][0]["frames"][0]["path"]).is_absolute()
    assert validated["samples"][0]["reference_status"] == "REFERENCE_MISSING"
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["samples"][0]["audio"]["sha256"] = "0" * 64
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(BenchmarkValidationError, match="hash mismatch"):
        validate_manifest(manifest)


def test_failure_sanitization_and_safety_regression():
    assert classify_failure("HTTP 429") == "QUOTA_EXHAUSTED"
    assert classify_failure("invalid JSON schema") == "STRUCTURED_OUTPUT_INVALID"
    sanitized = sanitize_artifact({
        "Authorization": "Bearer secret", "api_key": "secret",
        "inline_data": {"data": "payload"}, "data": "A" * 129, "text": "safe",
    })
    assert sanitized["Authorization"] == sanitized["api_key"] == "[REDACTED]"
    assert sanitized["inline_data"] == sanitized["data"] == "[REDACTED]"
    assert sanitized["text"] == "safe"
    assert not evaluate_safety_regression({"samples": [{"id": "sample5", "metadata": {"intervals": {"interval_6": {
        "status": "SKIPPED", "reason": "true_person_overlap_risk"}}}}]})
    assert evaluate_safety_regression({"samples": [{"id": "sample5", "metadata": {"intervals": {"interval_6": {"status": "FAILED"}}}}]})


def test_script_dry_run_writes_sanitized_artifacts(tmp_path):
    manifest, output = _manifest(tmp_path), tmp_path / "benchmark"
    result = subprocess.run(
        [sys.executable, "scripts/benchmark_gemini_phase_e.py", "--manifest", str(manifest), "--output-dir", str(output)],
        cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert {path.name for path in output.iterdir()} == {
        "manifest.json", "environment.json", "failures.json", "comparison.json", "summary.md",
    }
    assert json.loads((output / "environment.json").read_text())["execution"] == "dry_run"


def test_live_requires_key_and_marks_incomplete(tmp_path, monkeypatch):
    manifest, output = _manifest(tmp_path, reference=False), tmp_path / "benchmark"
    monkeypatch.setenv("GEMINI_API_KEY", "test-only")
    result = subprocess.run(
        [sys.executable, "scripts/benchmark_gemini_phase_e.py", "--manifest", str(manifest), "--output-dir", str(output), "--live"],
        cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, check=False,
        env={**__import__("os").environ, "GEMINI_API_KEY": "test-only"},
    )
    assert result.returncode == 0
    codes = {failure["code"] for failure in json.loads((output / "failures.json").read_text())}
    assert {"REFERENCE_MISSING", "BENCHMARK_INCOMPLETE"} <= codes
