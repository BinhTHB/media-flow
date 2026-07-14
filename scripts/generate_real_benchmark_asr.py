"""Generate Faster-Whisper ASR artifacts for real benchmark videos."""

import argparse
import json
import subprocess
from pathlib import Path

from src.infrastructure.providers.speech.faster_whisper_speech_provider import FasterWhisperSpeechProvider


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-root", type=Path, default=Path("test_videos/real_mini_benchmark"))
    parser.add_argument("--output-root", type=Path, default=Path("output/real_video_benchmark_results"))
    parser.add_argument("--samples", default="sample1,sample4")
    args = parser.parse_args()
    provider = FasterWhisperSpeechProvider("base", "cuda", "float16")
    for sample in args.samples.split(","):
        video = args.benchmark_root / f"{sample}.mp4"
        destination = args.output_root / sample
        audio = destination / "audio" / "asr.wav"
        audio.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(video), "-vn", "-ar", "16000", "-ac", "1", str(audio)],
            check=True, capture_output=True,
        )
        segments = provider.transcribe(str(audio), "auto")
        payload = [{
            **segment.to_dict(),
            "language": "auto",
            "provider": {"name": "faster-whisper", "model": "base", "device": "cuda"},
        } for segment in segments]
        transcripts = destination / "transcripts"
        transcripts.mkdir(exist_ok=True)
        (transcripts / "asr_segments.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
