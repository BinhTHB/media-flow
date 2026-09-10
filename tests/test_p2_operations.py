from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.application.result import ProcessingResult
from src.interfaces.jobs.runner import JobError, JobOptions, JobRunner, event_as_json
from src.interfaces.jobs.store import JobStore


class _FakeUseCase:
    def __init__(self, options: JobOptions) -> None:
        self.options = options

    def execute(self, request) -> ProcessingResult:
        Path(self.options.output_path).write_bytes(b"output")
        return ProcessingResult(
            job_id=str(request.job_id),
            status="SUCCESS",
            input_video_path=self.options.input_path,
            output_video_path=self.options.output_path,
            detections_summary={"subtitle": 2},
            tracks_summary={"subtitle": 1},
        )


def _options(tmp_path: Path, job_id: str = "p2-job") -> JobOptions:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"fixture")
    return JobOptions(
        input_path=str(source),
        output_path=str(tmp_path / "output.mp4"),
        preset="preserve-all",
        device="cpu",
        job_id=job_id,
        overwrite=True,
        no_visual_cache=True,
    )


def test_runner_persists_safe_provenance_and_metrics(tmp_path, monkeypatch) -> None:
    options = _options(tmp_path)
    store = JobStore(tmp_path / "jobs")
    monkeypatch.setattr(JobRunner, "_preflight", lambda self, value: "cpu")
    runner = JobRunner(
        job_store=store,
        use_case_factory=lambda opts, request: (_FakeUseCase(opts), {}),
    )

    events = []
    runner.run(options, events.append)
    state = store.get(options.job_id)

    assert state["provenance"] == {
        "profile": "balanced",
        "device": "cpu",
        "providers": {"asr": "mock", "translation": "mock", "tts": "auto"},
    }
    assert state["metrics"]["detections"] == {"subtitle": 2}
    assert state["metrics"]["tracks"] == {"subtitle": 1}
    assert state["metrics"]["duration_seconds"] >= 0
    assert json.loads(event_as_json(events[-1]))["type"] == "completed"


def test_persisted_cancel_is_observed_at_checkpoint(tmp_path, monkeypatch) -> None:
    options = _options(tmp_path, "cancel-me")
    store = JobStore(tmp_path / "jobs")
    store.request_cancel(options.job_id)
    monkeypatch.setattr(JobRunner, "_preflight", lambda self, value: "cpu")
    runner = JobRunner(job_store=store, use_case_factory=lambda opts, request: (_FakeUseCase(opts), {}))

    with pytest.raises(JobError) as raised:
        runner.run(options)

    assert raised.value.code == "cancelled"
    assert store.get(options.job_id)["status"] == "BLOCKED"


def test_claim_prevents_duplicate_job_and_releases_after_run(tmp_path, monkeypatch) -> None:
    options = _options(tmp_path, "same-job")
    store = JobStore(tmp_path / "jobs")
    assert store.claim(options.job_id)
    monkeypatch.setattr(JobRunner, "_preflight", lambda self, value: "cpu")
    runner = JobRunner(job_store=store, use_case_factory=lambda opts, request: (_FakeUseCase(opts), {}))

    with pytest.raises(JobError) as raised:
        runner.run(options)
    assert raised.value.code == "concurrency_limit"
    store.release(options.job_id)
    assert not store.is_claimed(options.job_id)


def test_store_bounds_events_and_cleanup_skips_running_or_claimed(tmp_path) -> None:
    store = JobStore(tmp_path / "jobs", max_events=2)
    for index in range(3):
        store.append_event("old", {"index": index, "authorization": "secret"})
    store.save_state("old", {"id": "old", "status": "SUCCESS", "updated_at": 1})
    store.save_state("running", {"id": "running", "status": "RUNNING", "updated_at": 2})
    store.save_state("claimed", {"id": "claimed", "status": "SUCCESS", "updated_at": 3})
    assert store.claim("claimed")
    old = store.get("old")
    assert [event["index"] for event in old["events"]] == [1, 2]
    assert all(event["authorization"] == "[REDACTED]" for event in old["events"])

    removed = store.cleanup(max_age_seconds=10, now=100)

    assert removed == ["old"]
    assert store.get("running") is not None
    assert store.get("claimed") is not None
    store.release("claimed")


def test_benchmark_manifest_has_fixture_and_thresholds() -> None:
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / "benchmarks/manifest.json").read_text(encoding="utf-8"))
    assert manifest["version"] == 1
    assert manifest["thresholds"]["max_secret_findings"] == 0
    for case in manifest["cases"]:
        assert (root / "benchmarks" / case["fixture"]).is_file()
