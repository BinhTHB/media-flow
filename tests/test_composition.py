import os
import argparse
import pytest
from src.interfaces.composition.application_builder import ApplicationBuilder
from src.interfaces.composition.application_context import ApplicationContext
from src.interfaces.application.media_flow_application import MediaFlowApplication
from src.domain.entities.video_job import VideoJob
from src.domain.entities.feature_config import FeatureConfig
from src.domain.entities.job_status import JobStatus
from src.shared.exceptions import ValidationError

def test_application_builder_builds_full_graph():
    builder = ApplicationBuilder(
        output_dir="test_output",
        video_provider="mock",
        speech_provider="mock",
        translation_provider="mock",
        subtitle_provider="mock",
        ocr_provider="mock",
        frame_provider="mock",
    )
    app = builder.build()
    
    assert isinstance(app, MediaFlowApplication)
    assert isinstance(app.context, ApplicationContext)
    assert app.context.artifact_manager.base_dir == "test_output"
    assert app.orchestrator is not None

def test_media_flow_application_run():
    builder = ApplicationBuilder(
        output_dir="test_output",
        video_provider="mock",
        speech_provider="mock",
        translation_provider="mock",
        subtitle_provider="mock",
        ocr_provider="mock",
        frame_provider="mock",
    )
    app = builder.build()
    
    job = VideoJob(
        job_id="test_comp_job",
        input_video="test.mp4",
        source_language="zh",
        target_language="vi",
        features=FeatureConfig(
            speech_to_text=True,
            ocr=False,
            translate=False,
            subtitle=False,
            render_video=False,
        ),
    )
    
    result = app.run(job)
    assert result.status == JobStatus.COMPLETED
    assert len(result.segments) == 2

def test_application_builder_from_cli_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="cli_out")
    parser.add_argument("--video-provider", default="mock")
    parser.add_argument("--speech-provider", default="mock")
    parser.add_argument("--translation-provider", default="mock")
    parser.add_argument("--subtitle-provider", default="mock")
    parser.add_argument("--ocr-provider", default="mock")
    parser.add_argument("--frame-provider", default="mock")
    parser.add_argument("--whisper-model", default="base")
    parser.add_argument("--target-lang", default="vi")
    parser.add_argument("--frame-interval", type=float, default=1.0)
    parser.add_argument("--keep-audio", action="store_true")
    parser.add_argument("--subtitle-format", default="srt")
    parser.add_argument("--subtitle-style", default="default")
    parser.add_argument("--font", default=None)
    parser.add_argument("--font-size", type=int, default=None)
    parser.add_argument("--outline", type=float, default=None)
    parser.add_argument("--shadow", type=float, default=None)
    parser.add_argument("--margin-v", type=int, default=None)
    parser.add_argument("--max-chars-per-line", type=int, default=42)
    parser.add_argument("--max-lines-per-subtitle", type=int, default=2)
    parser.add_argument("--min-duration", type=float, default=0.8)
    parser.add_argument("--max-duration", type=float, default=6.0)
    parser.add_argument("--no-deduplicate", action="store_true")
    parser.add_argument("--disable-segmentation", action="store_true")
    parser.add_argument("--prefer-ocr-over-speech", action="store_true")
    
    args = parser.parse_args([])
    builder = ApplicationBuilder.from_cli_args(args)
    app = builder.build()
    
    assert isinstance(app, MediaFlowApplication)
    assert app.context.artifact_manager.base_dir == "cli_out"

def test_cli_does_not_manually_wire_dependencies():
    with open("src/interfaces/cli/demo.py", "r", encoding="utf-8") as f:
        source = f.read()

    assert "ProviderFactory" not in source
    assert "ProviderRegistry" not in source
    assert "JobOrchestrator" not in source
    assert "ExtractAudioUseCase" not in source
    assert "ApplicationBuilder.from_cli_args" in source


def test_cli_does_not_instantiate_directly():
    demo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../src/interfaces/cli/demo.py"))
    with open(demo_path, "r", encoding="utf-8") as f:
        source = f.read()
    forbidden_imports = [
        "ProviderFactory",
        "ProviderRegistry",
        "JobOrchestrator",
        "ExtractAudioUseCase",
        "TranscribeSpeechUseCase",
        "DetectOnscreenTextUseCase",
        "TranslateSegmentsUseCase",
        "GenerateSubtitleUseCase",
        "RenderVideoUseCase",
        "ExtractFramesUseCase",
        "MergeSegmentsUseCase",
        "NormalizeSegmentsUseCase",
        "SubtitleSegmentationUseCase",
        "ArtifactManager",
    ]
    for imp in forbidden_imports:
        assert f"from src.infrastructure" not in source or f"{imp}" not in source,             f"CLI demo.py must not import {imp} directly"
    assert "ApplicationBuilder.from_cli_args" in source,         "CLI must use ApplicationBuilder.from_cli_args"
    assert "app = builder.build()" in source,         "CLI must build app via builder"
    assert "app.run_cli(args)" in source,         "CLI must call app.run_cli(args)"

def test_config_loader_loads_defaults():
    from src.interfaces.config.config_loader import ConfigLoader

    app_config = ConfigLoader().load()
    assert app_config.defaults.get("video") == "mock"
    assert app_config.providers.get("speech", {}).get("faster-whisper", {}).get("model_size") == "base"

def test_config_loader_applies_gemini_env_overrides_without_storing_api_key(
    tmp_path, monkeypatch
):
    from src.interfaces.config.config_loader import ConfigLoader

    providers_path = tmp_path / "providers.yaml"
    styles_path = tmp_path / "styles.yaml"
    providers_path.write_text(
        "providers:\n"
        "  translation:\n"
        "    gemini:\n"
        "      model: configured-model\n"
        "      max_retries: 2\n",
        encoding="utf-8",
    )
    styles_path.write_text("styles: {}\n", encoding="utf-8")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("GEMINI_MODEL", "env-model")
    monkeypatch.setenv("GEMINI_MAX_RETRIES", "4")
    monkeypatch.setenv("GEMINI_CACHE_ENABLED", "false")

    loader = ConfigLoader(str(providers_path), str(styles_path))
    config = loader.load()

    assert config.gemini.model == "env-model"
    assert config.gemini.max_retries == 4
    assert config.gemini.cache_enabled is False
    assert config.providers["translation"]["gemini"].get("api_key") is None
    assert loader.gemini_api_key == "test-key"

def test_config_loader_missing_sections(tmp_path):
    from src.interfaces.config.config_loader import ConfigLoader

    loader = ConfigLoader(
        providers_path=str(tmp_path / "missing.yaml"),
        subtitle_styles_path=str(tmp_path / "missing_styles.yaml"),
    )
    app_config = loader.load()
    assert app_config.defaults == {}
    assert app_config.styles == {}
    assert app_config.subtitle_quality.deduplicate is True

def test_app_config_get_provider_config():
    from src.interfaces.config.app_config import AppConfig

    config = AppConfig(defaults={"video": "mock"}, providers={"video": {"mock": {}}})
    provider_config = config.get_provider_config("video", overrides={"output_dir": "tmp"})
    assert provider_config.name == "mock"
    assert provider_config.options == {"output_dir": "tmp"}

def test_provider_registry_create():
    from src.infrastructure.registry.provider_registry import ProviderRegistry

    registry = ProviderRegistry()
    assert registry.create("video", "mock", {}).__class__.__name__ == "MockVideoProvider"
    assert registry.create("speech", "mock", {}).__class__.__name__ == "MockSpeechProvider"
    assert registry.create("video", "ffmpeg", {"output_dir": "custom"}).__class__.__name__ == "FFmpegVideoProvider"
    assert registry.create("translation", "gemini", {}).model == "gemini-3.1-flash-lite"

def test_provider_factory_uses_registry():
    import pytest
    from src.infrastructure.factory.provider_factory import ProviderFactory

    factory = ProviderFactory(config={"defaults": {"video": "mock"}, "providers": {"video": {"mock": {}}}})
    assert factory.create_video_provider().__class__.__name__ == "MockVideoProvider"
    with pytest.raises(KeyError):
        factory.create_video_provider(name="nonexistent")

def test_application_builder_context_uses_app_config():
    from src.interfaces.config.app_config import AppConfig

    app = ApplicationBuilder(video_provider="mock", speech_provider="mock").build()
    assert isinstance(app.context.config, AppConfig)


def test_gemini_model_is_selected_from_provider_config():
    from src.infrastructure.factory.provider_factory import ProviderFactory

    factory = ProviderFactory(
        config={
            "defaults": {"translation": "gemini"},
            "providers": {
                "translation": {
                    "gemini": {
                        "model": "gemini-3.1-flash-lite",
                        "max_retries": 5,
                    }
                }
            },
        }
    )

    provider = factory.create_translation_provider()

    assert provider.model == "gemini-3.1-flash-lite"
    assert provider.max_retries == 5


def test_config_loader_rejects_quality_keys_in_style_presets(tmp_path):
    from src.interfaces.config.config_loader import ConfigLoader

    providers_path = tmp_path / "providers.yaml"
    styles_path = tmp_path / "styles.yaml"
    providers_path.write_text("subtitle_quality: {}\n", encoding="utf-8")
    styles_path.write_text(
        "styles:\n  invalid:\n    font_family: Arial\n    max_chars_per_line: 42\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="cannot contain quality keys"):
        ConfigLoader(str(providers_path), str(styles_path)).load()


def test_config_loader_rejects_visual_keys_in_subtitle_quality(tmp_path):
    from src.interfaces.config.config_loader import ConfigLoader

    providers_path = tmp_path / "providers.yaml"
    styles_path = tmp_path / "styles.yaml"
    providers_path.write_text(
        "subtitle_quality:\n  font_size: 28\n",
        encoding="utf-8",
    )
    styles_path.write_text("styles: {}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="cannot contain visual style keys"):
        ConfigLoader(str(providers_path), str(styles_path)).load()

def test_application_builder_preserves_quality_config_and_applies_overrides(tmp_path):
    providers_path = tmp_path / "providers.yaml"
    styles_path = tmp_path / "styles.yaml"
    providers_path.write_text(
        "subtitle_quality:\n"
        "  max_chars_per_line: 30\n"
        "  max_lines_per_subtitle: 1\n"
        "  min_duration: 1.2\n"
        "  max_duration: 4.0\n",
        encoding="utf-8",
    )
    styles_path.write_text("styles: {}\n", encoding="utf-8")

    configured = ApplicationBuilder(
        config_path=str(providers_path),
        subtitle_styles_path=str(styles_path),
    ).build()
    overridden = ApplicationBuilder(
        config_path=str(providers_path),
        subtitle_styles_path=str(styles_path),
        max_chars_per_line=24,
    ).build()

    assert configured.context.config.subtitle_quality.max_chars_per_line == 30
    assert configured.context.config.subtitle_quality.max_lines_per_subtitle == 1
    assert configured.context.config.subtitle_quality.min_duration == 1.2
    assert configured.context.config.subtitle_quality.max_duration == 4.0
    assert overridden.context.config.subtitle_quality.max_chars_per_line == 24
