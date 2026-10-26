from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import pytest

import scripts.run_e2e as cli
from src.application.request import ProcessingRequest
from src.application.result import ProcessingResult
from src.core.errors import ConfigurationError, ValidationError
from src.features.audio.models import AudioRequest
from src.features.visual.pipeline import VisualRequest
from src.interfaces.jobs.runner import JobError, JobOptions, JobRunner
from src.interfaces.web.server import build_job_options


class FakeUseCase:
    def __init__(self, result: ProcessingResult | None = None, error: Exception | None = None) -> None:
        self.result = result
        self.error = error

    def execute(self, request):
        if self.error:
            raise self.error
        assert self.result is not None
        return self.result


def _options(tmp_path: Path, profile: str = "balanced") -> JobOptions:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"fixture")
    return JobOptions(
        input_path=str(source),
        output_path=str(tmp_path / "output.mp4"),
        preset="preserve-all",
        device="cpu",
        overwrite=True,
        profile=profile,
        no_visual_cache=True,
    )


def _result(options: JobOptions) -> ProcessingResult:
    Path(options.output_path).write_bytes(b"output")
    return ProcessingResult(
        job_id="offline",
        status="SUCCESS",
        input_video_path=options.input_path,
        output_video_path=options.output_path,
        detections_summary={"subtitle": 2},
    )


def test_runner_emits_structured_progress_offline(tmp_path, monkeypatch) -> None:
    options = _options(tmp_path)
    monkeypatch.setattr(JobRunner, "_preflight", lambda self, value: "cpu")
    runner = JobRunner(use_case_factory=lambda opts, request: (FakeUseCase(_result(opts)), {}))
    events = []

    result = runner.run(options, events.append)

    assert result.status == "SUCCESS"
    assert [event.type for event in events] == ["started", "progress", "progress", "progress", "completed"]
    assert [event.progress for event in events] == [5, 25, 40, 92, 100]
    assert events[-2].details == {"detections": {"subtitle": 2}}
    assert events[-1].to_dict()["details"]["result"]["status"] == "SUCCESS"


def test_runner_maps_failure_and_emits_failed_event(tmp_path, monkeypatch) -> None:
    options = _options(tmp_path)
    monkeypatch.setattr(JobRunner, "_preflight", lambda self, value: "cpu")
    runner = JobRunner(
        use_case_factory=lambda opts, request: (FakeUseCase(error=ValidationError("bad media")), {})
    )
    events = []

    with pytest.raises(JobError) as raised:
        runner.run(options, events.append)

    assert raised.value.code == "validation_error"
    assert events[-1].type == "failed"
    assert events[-1].details == {"error": {"code": "validation_error", "message": "bad media"}}


def test_cli_and_web_resolve_same_runner_options(tmp_path) -> None:
    options = _options(tmp_path, profile="fast")
    web = build_job_options({
        "id": "same-job",
        "input_path": options.input_path,
        "output_path": options.output_path,
        "preset": options.preset,
        "profile": options.profile,
        "device": options.device,
        "overwrite": options.overwrite,
    })
    captured = {}

    class CapturingRunner:
        def run(self, received, callback):
            captured["options"] = received
            return _result(received)

    args = Namespace(
        input=options.input_path, output=options.output_path, preset=options.preset,
        device=options.device, job_id="same-job", overwrite=True, profile="fast",
        source_language="auto", target_language="vi", voice="elevenlabs:env",
        asr_provider="auto", asr_correction="auto", translation_provider="auto",
        duration_rewrite="auto", max_tts_speedup=1.30, tts_provider="auto",
        visual_cache_dir=options.visual_cache_dir, no_visual_cache=True,
        yolo_weights=options.yolo_weights, yolo_imgsz=1024, yolo_conf=0.05,
        yolo_iou=0.70, yolo_subtitle_rescue_conf=0.01,
        yolo_subtitle_rescue_max_gap_seconds=3.0,
        yolo_subtitle_rescue_min_support_ratio=0.25, subtitle_layout="temporal",
    )

    cli.run_job(args, runner=CapturingRunner())

    resolved = captured["options"]
    assert type(resolved) is type(web) is JobOptions
    assert (resolved.input_path, resolved.output_path, resolved.preset, resolved.profile) == (
        web.input_path, web.output_path, web.preset, web.profile,
    )


def test_runner_prepares_generic_preset(tmp_path, monkeypatch) -> None:
    options = _options(tmp_path)
    options.preset = "translate-subtitle-generic"
    monkeypatch.setattr(JobRunner, "_preflight", lambda self, value: "cpu")
    runner = JobRunner()

    normalized, request = runner.prepare(options)

    # The default burn-in weights are swapped for the generic text model.
    assert Path(normalized.yolo_weights) == Path(normalized.yolo_weights).resolve()
    assert Path(normalized.yolo_weights).as_posix().endswith("models/yolo/text/best.pt")
    # Visual semantics match the base preset (translate-subtitle).
    assert request.visual.enabled is True
    assert request.visual.subtitle_mode == "translate_replace"
    assert request.audio.enabled is False


def test_generic_preset_missing_weights_names_huggingface(tmp_path, monkeypatch) -> None:
    options = _options(tmp_path)
    options.preset = "translate-subtitle-generic"
    options.yolo_weights = str(tmp_path / "missing.pt")
    monkeypatch.setattr(JobRunner, "_require_dependency", lambda self, mod, pkg: object())
    monkeypatch.setattr("src.interfaces.jobs.runner.shutil.which", lambda name: name)

    with pytest.raises(ConfigurationError, match="huggingface.co/RoyRud1902/yolo11n-text"):
        JobRunner().prepare(options)


def test_compose_wraps_generic_detector(monkeypatch, tmp_path) -> None:
    import src.interfaces.jobs.runner as runner_mod
    from src.features.visual.text_classifier import TextClassifyingDetector

    captured: dict = {}

    class _StubDetector:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(runner_mod, "YoloDetectorAdapter", _StubDetector)
    monkeypatch.setattr(runner_mod, "build_translation_service", lambda *args, **kwargs: object())

    options = _options(tmp_path)
    options.preset = "translate-subtitle-generic"
    generic_weights = str(tmp_path / "text-best.pt")
    options.yolo_weights = generic_weights
    request = ProcessingRequest(
        job_id="generic-compose",
        input_video=options.input_path,
        output_video=options.output_path,
        audio=AudioRequest(enabled=False),
        visual=VisualRequest(enabled=True, subtitle_enabled=True, subtitle_mode="translate_replace"),
        device="cpu",
        overwrite=True,
    )

    use_case, _ = runner_mod.JobRunner._compose(options, request)

    # The generic operating point: predictor-level conf, rescue disabled.
    assert captured["conf"] == 0.20
    assert captured["subtitle_rescue_conf"] is None
    assert captured["weights_path"] == generic_weights
    assert isinstance(use_case.visual_pipeline.detector, TextClassifyingDetector)


def test_runner_prepares_flow_preset_with_explicit_features(tmp_path, monkeypatch) -> None:
    # The new contract: preset selects the detection flow, feature fields say
    # what happens. yolo-text + translate must reproduce the old generic
    # subtitle-preset behavior.
    options = _options(tmp_path)
    options.preset = "yolo-text"
    options.subtitle_action = "translate"
    options.dub_audio = False
    options.extra_text_action = "preserve"
    monkeypatch.setattr(JobRunner, "_preflight", lambda self, value: "cpu")
    runner = JobRunner()

    normalized, request = runner.prepare(options)

    assert normalized.preset == "yolo-text"
    assert Path(normalized.yolo_weights).as_posix().endswith("models/yolo/text/best.pt")
    assert request.visual.enabled is True
    assert request.visual.subtitle_mode == "translate_replace"
    assert request.visual.other_text_action == "preserve"
    assert request.audio.enabled is False


def test_runner_disables_visual_for_preserve_only_job(tmp_path, monkeypatch) -> None:
    # preserve subtitles + preserve extra text = nothing touches pixels:
    # detection is skipped entirely (mirrors the old dub-audio preset).
    options = _options(tmp_path)
    options.preset = "yolo-4class"
    options.subtitle_action = "preserve"
    options.dub_audio = True
    options.extra_text_action = "preserve"
    monkeypatch.setattr(JobRunner, "_preflight", lambda self, value: "cpu")
    runner = JobRunner()

    _, request = runner.prepare(options)

    assert request.visual.enabled is False
    assert request.audio.enabled is True
    assert request.audio.dubbing_enabled is True


def test_runner_rejects_unknown_preset(tmp_path) -> None:
    options = _options(tmp_path)
    options.preset = "nonsense"
    with pytest.raises(Exception, match="Unsupported preset"):
        JobRunner().prepare(options)
