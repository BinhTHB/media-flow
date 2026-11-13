"""Smoke tests for the CLI entry: the command the web UI preview shows must
parse into the same JobOptions contract the web server builds."""

from src.interfaces.cli import build_job_options


def test_web_ui_preview_command_parses_into_job_options():
    # Exact flag sequence emitted by frontend/studio.js buildCliPreview().
    argv = [
        "--input", "video.mp4",
        "--preset", "yolo-4class",
        "--subtitle-action", "translate",
        "--dub-audio",
        "--extra-text-action", "blur",
        "--source-language", "zh",
        "--target-language", "vi",
        "--subtitle-layout", "in-place",
        "--voice-id", "elevenlabs:EXAVITQu4vr4xnSDxMaL",
        "--hide-source-subtitle",
        "--overwrite",
    ]
    options = build_job_options(argv)

    assert options.input_path == "video.mp4"
    assert options.preset == "yolo-4class"
    assert options.subtitle_action == "translate"
    assert options.dub_audio is True
    assert options.extra_text_action == "blur"
    assert options.source_language == "zh"
    assert options.target_language == "vi"
    assert options.subtitle_layout == "in-place"
    assert options.voice == "elevenlabs:EXAVITQu4vr4xnSDxMaL"
    assert options.hide_source_subtitle is True
    assert options.overwrite is True


def test_defaults_and_derived_output_path():
    options = build_job_options(["--input", "clip.mp4"])

    assert options.preset == "yolo-4class"
    assert options.subtitle_action == "blur"
    assert options.dub_audio is False
    assert options.hide_source_subtitle is True
    assert options.output_path.replace("\\", "/").endswith("outputs/cli/clip-yolo-4class.mp4")


def test_reup_defaults_to_picture_speech_and_can_turn_speech_off():
    options = build_job_options(["--input", "clip.mp4", "--preset", "reup"])

    assert options.subtitle_action == "translate"
    assert options.extra_text_action == "blur"
    assert options.dub_audio is True

    quiet = build_job_options(["--input", "clip.mp4", "--preset", "reup", "--no-dub-audio"])
    assert quiet.dub_audio is False
    kept = build_job_options([
        "--input", "clip.mp4", "--preset", "reup", "--subtitle-action", "preserve",
    ])
    assert kept.subtitle_action == "preserve"
    assert kept.dub_audio is True
    assert options.create_missing_subtitles is True
    off = build_job_options(["--input", "clip.mp4", "--preset", "reup", "--no-speech-captions"])
    assert off.create_missing_subtitles is False


def test_show_source_subtitle_overrides_default():
    options = build_job_options(["--input", "v.mp4", "--show-source-subtitle"])
    assert options.hide_source_subtitle is False
