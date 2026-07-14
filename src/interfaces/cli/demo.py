import argparse
import json
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from src.interfaces.composition.application_builder import ApplicationBuilder

def main():
    parser = argparse.ArgumentParser(description="AI Video Translator CLI Demo")
    parser.add_argument("video_path", type=str, help="Path to input video file")
    parser.add_argument("--source-lang", type=str, default="zh", help="Source language (zh|en|auto)")
    parser.add_argument(
        "--speech-lang",
        type=str,
        default="auto",
        help="Speech language (zh|en|vi|auto), independent from OCR source language",
    )
    parser.add_argument("--target-lang", type=str, default="vi", help="Target language (vi)")
    parser.add_argument("--stt", action="store_true", help="Enable speech-to-text")
    parser.add_argument("--ocr", action="store_true", help="Enable OCR text detection")
    parser.add_argument("--skip-ocr", action="store_true")
    parser.add_argument("--reuse-ocr-cache", action="store_true")
    parser.add_argument("--skip-subtitle-region-detection", action="store_true")
    parser.add_argument("--reuse-subtitle-region-cache", action="store_true")
    parser.add_argument("--rerender-only", action="store_true")
    parser.add_argument("--debug-from-artifacts", type=str, default="")
    parser.add_argument("--ocr-device", choices=["cpu", "gpu", "auto"], default="gpu")
    parser.add_argument("--resource-profile", choices=["low", "balanced", "quality"], default="balanced")
    parser.add_argument("--ffmpeg-encoder", choices=["auto", "cpu", "nvenc"], default="auto")
    parser.add_argument("--ffmpeg-gpu-decode", action="store_true")
    parser.add_argument("--ffmpeg-nvenc-cq", type=int, default=23)
    parser.add_argument("--ffmpeg-nvenc-preset", default="p5")
    parser.add_argument("--translate", action="store_true", help="Enable translation")
    parser.add_argument("--subtitle", action="store_true", help="Enable subtitle generation")
    parser.add_argument("--subtitle-placement", action="store_true", help="Enable automatic subtitle placement")
    parser.add_argument(
        "--placement-mode",
        choices=["avoid_source_subtitle", "replace_source_subtitle", "fixed_safe_zone"],
        help="Subtitle placement policy",
    )
    parser.add_argument(
        "--debug-placement",
        "--subtitle-placement-debug",
        dest="subtitle_placement_debug",
        action="store_true",
        help="Generate subtitle placement debug artifacts",
    )
    parser.add_argument("--render", action="store_true", help="Enable video rendering with subtitles")
    parser.add_argument("--dubbing", action="store_true", help="Enable Vietnamese dubbing")
    parser.add_argument("--tts-voice", default="vi-VN-HoaiMyNeural")
    parser.add_argument("--tts-rate", default="+0%")
    parser.add_argument(
        "--original-audio-mode",
        choices=["keep", "lower", "replace"],
        default="lower",
    )
    parser.add_argument("--original-audio-volume", type=float, default=0.2)
    parser.add_argument(
        "--preserve-original-audio",
        action="store_true",
        help="Mix the original audio at full volume",
    )
    parser.add_argument("--render-output", type=str, help="Rendered video output path")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing rendered video")

    parser.add_argument("--video-provider", type=str, help="Video provider (mock|ffmpeg)")
    parser.add_argument("--speech-provider", type=str, help="Speech provider (mock|faster-whisper)")
    parser.add_argument("--translation-provider", type=str, help="Translation provider (mock|openai)")
    parser.add_argument("--subtitle-provider", type=str, help="Subtitle provider (mock|ass)")
    parser.add_argument("--ocr-provider", type=str, help="OCR provider (mock|paddle)")
    parser.add_argument("--subtitle-region-detector", type=str, help="Region detector provider (mock|paddle)")
    parser.add_argument("--frame-provider", type=str, help="Frame extraction provider (mock|ffmpeg)")
    parser.add_argument("--frame-interval", type=float, default=1.0, help="Interval (seconds) for frame extraction")
    parser.add_argument("--max-chars-per-line", type=int, default=42, help="Max chars per subtitle line")
    parser.add_argument("--max-lines-per-subtitle", type=int, default=2, help="Max lines per subtitle")
    parser.add_argument("--no-deduplicate", action="store_true", help="Disable duplicate text removal")
    parser.add_argument("--disable-segmentation", action="store_true", help="Disable subtitle segmentation")
    parser.add_argument("--resegment", action="store_true", help="Rerun local subtitle segmentation from cached normalized text")
    parser.add_argument(
        "--enable-vietnamese-text-correction",
        action="store_true",
        help="Apply only high-confidence Vietnamese subtitle corrections",
    )
    parser.add_argument("--max-duration", type=float, default=6.0, help="Maximum subtitle duration")
    parser.add_argument("--min-duration", type=float, default=0.8, help="Minimum subtitle duration")
    parser.add_argument("--prefer-ocr-over-speech", action="store_true", help="Prefer OCR text over speech when overlapping")
    parser.add_argument("--whisper-model", type=str, help="Whisper model (tiny|base|small|medium)")
    parser.add_argument("--output-dir", type=str, default="output", help="Output directory")
    parser.add_argument("--job-id", type=str, default="job_demo_123", help="Unique Job ID")
    parser.add_argument("--keep-audio", action="store_true", help="Keep extracted audio.wav")
    parser.add_argument("--subtitle-format", type=str, choices=["srt", "ass"], default="srt", help="Subtitle format (srt|ass)")
    parser.add_argument("--subtitle-style", type=str, default="default", help="Subtitle style preset")
    parser.add_argument("--font", type=str, help="Font family override")
    parser.add_argument("--font-size", type=int, help="Font size override")
    parser.add_argument("--outline", type=float, help="Outline width override")
    parser.add_argument("--shadow", type=float, help="Shadow offset override")
    parser.add_argument("--margin-v", type=int, help="Vertical margin override")

    args = parser.parse_args()

    builder = ApplicationBuilder.from_cli_args(args)
    app = builder.build()

    try:
        result_job = app.run_cli(args)
        print(json.dumps(result_job.to_dict(), indent=2, ensure_ascii=False))
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
