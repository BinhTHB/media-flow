from __future__ import annotations
import json
import pytest
from src.application.process_video import ProcessVideoUseCase
from src.application.request import ProcessingRequest
from src.core.errors import ConfigurationError, ValidationError
from src.features.audio.models import AudioResult
from src.features.audio.pipeline import AudioRequest
from src.features.visual.pipeline import VisualRequest, VisualResult
from src.interfaces.composition.config import ProviderConfig, load_provider_config
from src.interfaces.jobs.store import JobStore


def test_provider_precedence_runtime_env_config_default():
    config = ProviderConfig(defaults={"translation": "gemini"})
    assert config.resolve("translation", "openrouter", {"MEDIAFLOW_TRANSLATION_PROVIDER": "env"}) == "openrouter"
    assert config.resolve("translation", "auto", {"MEDIAFLOW_TRANSLATION_PROVIDER": "env"}) == "env"
    assert config.resolve("translation", "auto", {}) == "gemini"
    assert ProviderConfig().resolve("translation", "auto", {}) == "auto"


def test_provider_config_rejects_duplicate_keys(tmp_path):
    path = tmp_path / "providers.yaml"
    path.write_text("defaults:\n  speech: auto\n  speech: gemini\n", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="Duplicate"): load_provider_config(path)


def test_job_store_recovery_and_secret_redaction(tmp_path):
    store = JobStore(tmp_path / "jobs")
    store.save_state("job-1", {"id": "job-1", "status": "RUNNING", "api_key": "secret"})
    store.append_event("job-1", {"type": "started", "token": "secret"})
    store.save_result("job-1", {"status": "SUCCESS"})
    restored = JobStore(tmp_path / "jobs").get("job-1")
    assert restored["api_key"] == restored["events"][0]["token"] == "[REDACTED]"
    assert restored["result"]["status"] == "SUCCESS"
    assert JobStore(tmp_path / "jobs").list()[0]["id"] == "job-1"
    assert not list((tmp_path / "jobs").rglob("*.tmp"))


def test_use_case_preflight_creates_no_artifacts(tmp_path):
    request = ProcessingRequest(job_id="invalid", input_video=str(tmp_path / "missing.mp4"), output_video=str(tmp_path / "new/out.mp4"), audio=AudioRequest(enabled=False), visual=VisualRequest(enabled=False))
    with pytest.raises(ValidationError): ProcessVideoUseCase(workspace_base_dir=str(tmp_path / "jobs")).execute(request)
    assert not (tmp_path / "jobs").exists()
    assert not (tmp_path / "new").exists()


class _Audio:
    def __init__(self, status): self.status = status
    def process(self, request, job_workspace_dir): return AudioResult(enabled=True, status=self.status, warnings=["audio warning"])
class _Visual:
    def __init__(self, status): self.status = status
    def process(self, request, job_workspace_dir, output_video_path): return VisualResult(enabled=True, status=self.status, rendered_video_path="visual.mp4", warnings=["visual warning"])
class _Video:
    def mux_audio_video(self, video_no_audio_path, audio_source_path, output_path): return output_path
    def probe_video(self, path): return {"duration": 1.0, "fps": 30.0, "width": 10, "height": 10}


@pytest.mark.parametrize(("audio", "visual", "expected"), [("SUCCESS","SUCCESS","SUCCESS"),("PARTIAL","SUCCESS","PARTIAL"),("SUCCESS","PARTIAL","PARTIAL"),("FAILED","SUCCESS","FAILED"),("SUCCESS","BLOCKED","BLOCKED")])
def test_status_aggregation(tmp_path, audio, visual, expected):
    source = tmp_path / "source.mp4"; source.write_bytes(b"x")
    request = ProcessingRequest(job_id=f"status-{audio}-{visual}", input_video=str(source), output_video=str(tmp_path / "out.mp4"), audio=AudioRequest(enabled=True), visual=VisualRequest(enabled=True), overwrite=True)
    result = ProcessVideoUseCase(_Audio(audio), _Visual(visual), _Video(), str(tmp_path / "jobs")).execute(request)
    assert result.status == expected
    assert result.warnings == ["audio warning", "visual warning"]
    assert json.loads((tmp_path / "jobs" / request.job_id / "result.json").read_text())["status"] == expected
