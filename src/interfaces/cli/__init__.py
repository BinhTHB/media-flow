"""Command-line entry point for the MediaFlow pipeline.

Thin composition over :class:`JobRunner` (the same structured runner the
local web server uses). Flags mirror the CLI preview shown by the web UI:

    python -m src.interfaces.cli --input video.mp4 \
        --preset yolo-4class --subtitle-action translate --dub-audio \
        --extra-text-action blur
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from src.interfaces.jobs.runner import (
    PROJECT_ROOT,
    JobOptions,
    JobRunner,
    format_event,
)

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "cli"


def _flag_present(tokens: list[str], name: str) -> bool:
    return any(token == name or token.startswith(f"{name}=") for token in tokens)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.interfaces.cli",
        description="MediaFlow video translator (same runner as the web UI).",
    )
    parser.add_argument("--input", required=True, help="Input video path")
    parser.add_argument(
        "--output",
        default=None,
        help="Output video path (default: outputs/cli/<input-stem>-<preset>.mp4)",
    )
    parser.add_argument(
        "--preset",
        default="yolo-4class",
        help="Detection flow: yolo-4class | yolo-text | reup. Reup runs the visual and audio options you pass.",
    )
    parser.add_argument(
        "--subtitle-action",
        dest="subtitle_action",
        default="blur",
        help="Burned-in subtitle handling: translate | blur | preserve",
    )
    parser.add_argument(
        "--dub-audio",
        dest="dub_audio",
        action="store_true",
        help="Dub the audio pipeline over the video. Preset reup dubs unless --no-dub-audio is set.",
    )
    parser.add_argument(
        "--no-dub-audio",
        dest="no_dub_audio",
        action="store_true",
        help="Keep the original speech bed. Turns off the reup dub default.",
    )
    parser.add_argument(
        "--extra-text-action",
        dest="extra_text_action",
        default="blur",
        help="Logo/watermark/other text handling: blur | preserve",
    )
    parser.add_argument("--source-language", dest="source_language", default="auto")
    parser.add_argument("--target-language", dest="target_language", default="vi")
    parser.add_argument(
        "--subtitle-layout",
        dest="subtitle_layout",
        default="temporal",
        help="in-place | temporal | bottom",
    )
    parser.add_argument(
        "--voice-id",
        dest="voice",
        default="elevenlabs:env",
        help="TTS voice, e.g. elevenlabs:env or fish:<id>",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--profile", default="balanced")
    parser.add_argument(
        "--variation",
        default="balanced",
        choices=["off", "light", "balanced", "aggressive"],
        help="Bake speed, crop, mirror, and pitch into the finished file (default: balanced)",
    )
    parser.add_argument(
        "--hide-source-subtitle",
        dest="hide_source_subtitle",
        action="store_true",
        default=True,
        help="Hide the original burned-in subtitles (default)",
    )
    parser.add_argument(
        "--show-source-subtitle",
        dest="hide_source_subtitle",
        action="store_false",
        help="Keep the original burned-in subtitles visible",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output file if it exists",
    )
    parser.add_argument(
        "--no-speech-captions",
        action="store_true",
        help="Reup: do not create captions from speech. Only translate text the detector read.",
    )
    parser.add_argument(
        "--no-visual-cache",
        dest="no_visual_cache",
        action="store_true",
        help="Disable the visual stage cache",
    )
    return parser


def build_job_options(argv: Sequence[str] | None = None) -> JobOptions:
    args = build_parser().parse_args(argv)
    tokens = list(argv if argv is not None else sys.argv[1:])
    subtitle_action = args.subtitle_action
    extra_text_action = args.extra_text_action
    dub_audio = bool(args.dub_audio)
    if args.preset.removesuffix("-generic") == "reup":
        if not _flag_present(tokens, "--subtitle-action"):
            subtitle_action = "translate"
        if not _flag_present(tokens, "--extra-text-action"):
            extra_text_action = "blur"
        dub_audio = False if args.no_dub_audio else True
    output_path = args.output
    if output_path is None:
        stem = Path(args.input).stem or "video"
        output_path = str(DEFAULT_OUTPUT_DIR / f"{stem}-{args.preset}.mp4")
    return JobOptions(
        input_path=args.input,
        output_path=output_path,
        preset=args.preset,
        subtitle_action=subtitle_action,
        dub_audio=dub_audio,
        extra_text_action=extra_text_action,
        device=args.device,
        overwrite=args.overwrite,
        profile=args.profile,
        source_language=args.source_language,
        target_language=args.target_language,
        voice=args.voice,
        subtitle_layout=args.subtitle_layout,
        hide_source_subtitle=args.hide_source_subtitle,
        no_visual_cache=args.no_visual_cache,
        variation_level=args.variation,
        create_missing_subtitles=not args.no_speech_captions,
    )


def main(argv: Sequence[str] | None = None) -> int:
    options = build_job_options(argv)
    runner = JobRunner()
    runner.run(options, on_event=lambda event: print(format_event(event), flush=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
