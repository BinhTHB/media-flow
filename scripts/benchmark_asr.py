"""Benchmark SpeechProvider implementations without changing the production pipeline."""

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from src.infrastructure.providers.speech.faster_whisper_speech_provider import (
    FasterWhisperSpeechProvider,
)


DEFAULT_SUSPICIOUS_SUBSTITUTIONS = ("因果:英国",)


def normalized_text(text: str) -> str:
    return " ".join(text.strip().split())


def levenshtein(left: list[str], right: list[str]) -> int:
    """Return edit distance using a memory-efficient dynamic-programming row."""
    if len(left) < len(right):
        left, right = right, left
    row = list(range(len(right) + 1))
    for left_index, left_value in enumerate(left, 1):
        next_row = [left_index]
        for right_index, right_value in enumerate(right, 1):
            next_row.append(
                min(
                    next_row[-1] + 1,
                    row[right_index] + 1,
                    row[right_index - 1] + (left_value != right_value),
                )
            )
        row = next_row
    return row[-1]


def error_rates(transcript: str, reference: str | None) -> dict[str, float] | None:
    if reference is None:
        return None
    reference_chars = list(re.sub(r"\s+", "", reference))
    transcript_chars = list(re.sub(r"\s+", "", transcript))
    reference_words = normalized_text(reference).split()
    transcript_words = normalized_text(transcript).split()
    return {
        "cer": round(
            levenshtein(transcript_chars, reference_chars) / max(len(reference_chars), 1),
            4,
        ),
        "wer": round(
            levenshtein(transcript_words, reference_words) / max(len(reference_words), 1),
            4,
        ),
    }


def segment_metrics(segments: list[dict[str, Any]]) -> dict[str, Any]:
    durations = [max(segment["end"] - segment["start"], 0.0) for segment in segments]
    texts = [normalized_text(segment.get("text", "")) for segment in segments]
    nonempty_confidences = [
        segment["confidence"]
        for segment in segments
        if isinstance(segment.get("confidence"), (float, int))
        and segment["confidence"] > 0
    ]
    transcript = " ".join(texts)
    duplicate_count = sum(
        count - 1 for text, count in Counter(texts).items() if text and count > 1
    )
    return {
        "segment_count": len(segments),
        "average_segment_duration": round(sum(durations) / len(durations), 3)
        if durations
        else 0.0,
        "shortest_segment_duration": round(min(durations), 3) if durations else 0.0,
        "longest_segment_duration": round(max(durations), 3) if durations else 0.0,
        "character_count": len(re.sub(r"\s+", "", transcript)),
        "word_count": len(transcript.split()),
        "empty_segment_count": sum(not text for text in texts),
        "duplicate_segment_count": duplicate_count,
        "average_confidence": round(
            sum(nonempty_confidences) / len(nonempty_confidences), 4
        )
        if nonempty_confidences
        else None,
    }


def parse_substitutions(values: list[str]) -> list[tuple[str, str]]:
    pairs = []
    for value in values:
        expected, separator, observed = value.partition(":")
        if not separator or not expected or not observed:
            raise ValueError(f"Invalid substitution '{value}'; use expected:observed.")
        pairs.append((expected, observed))
    return pairs


def homophone_analysis(
    segments: list[dict[str, Any]], substitutions: list[tuple[str, str]]
) -> list[dict[str, Any]]:
    findings = []
    for expected, observed in substitutions:
        affected = [
            {
                "id": segment["id"],
                "start": segment["start"],
                "end": segment["end"],
                "text": segment["text"],
            }
            for segment in segments
            if observed in segment.get("text", "")
        ]
        findings.append(
            {
                "expected": expected,
                "observed": observed,
                "frequency": len(affected),
                "affected_timestamps": affected,
            }
        )
    return findings


def extract_audio(video: Path, audio_path: Path) -> float:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required to benchmark a video input.")
    started = time.perf_counter()
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(video),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            str(audio_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return round(time.perf_counter() - started, 3)


def build_provider(name: str, model: str):
    if name == "faster-whisper":
        return FasterWhisperSpeechProvider(model_size=model)
    raise ValueError(
        f"Unsupported provider '{name}'. Register a benchmark adapter in "
        "build_provider() without modifying the production pipeline."
    )


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def benchmark_model(
    provider_name: str,
    model: str,
    audio_path: Path,
    source_language: str,
    output_dir: Path,
    reference: str | None,
    substitutions: list[tuple[str, str]],
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    provider = build_provider(provider_name, model)
    started = time.perf_counter()
    segments = provider.transcribe(str(audio_path), source_language)
    runtime_seconds = round(time.perf_counter() - started, 3)
    serialized = [segment.to_dict() for segment in segments]
    transcript = " ".join(segment["text"] for segment in serialized)
    metrics = segment_metrics(serialized)
    metrics["error_rates"] = error_rates(transcript, reference)
    metrics["homophone_findings"] = homophone_analysis(serialized, substitutions)
    runtime = {"stt_seconds": runtime_seconds}

    write_json(output_dir / "transcript.json", serialized)
    write_json(output_dir / "metrics.json", metrics)
    write_json(output_dir / "runtime.json", runtime)
    return {"model": model, "metrics": metrics, "runtime": runtime}


def recommendation(models: list[str]) -> dict[str, str]:
    available = set(models)
    return {
        "smoke_test": "tiny" if "tiny" in available else models[0],
        "development": "base" if "base" in available else models[0],
        "production": "small" if "small" in available else "base" if "base" in available else models[0],
        "chinese_short_videos": "small" if "small" in available else "base" if "base" in available else models[0],
        "english_short_videos": "base" if "base" in available else models[0],
    }


def comparison_markdown(
    provider: str, results: list[dict[str, Any]], recommendations: dict[str, str]
) -> str:
    lines = [
        "# ASR Benchmark Comparison",
        "",
        f"Provider: `{provider}`",
        "",
        "| Model | Runtime | Segments | Avg duration | Avg confidence | CER | WER | Suspicious `英国` |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for result in results:
        metrics = result["metrics"]
        rates = metrics["error_rates"] or {}
        suspicious = sum(
            item["frequency"] for item in metrics["homophone_findings"] if item["observed"] == "英国"
        )
        lines.append(
            "| {model} | {runtime:.3f}s | {segments} | {duration:.3f}s | {confidence} | {cer} | {wer} | {suspicious} |".format(
                model=result["model"],
                runtime=result["runtime"]["stt_seconds"],
                segments=metrics["segment_count"],
                duration=metrics["average_segment_duration"],
                confidence=metrics["average_confidence"]
                if metrics["average_confidence"] is not None
                else "n/a",
                cer=rates.get("cer", "n/a"),
                wer=rates.get("wer", "n/a"),
                suspicious=suspicious,
            )
        )
    lines.extend(
        [
            "",
            "## Recommendations",
            "",
            *[
                f"- **{purpose.replace('_', ' ')}:** `{model}`"
                for purpose, model in recommendations.items()
            ],
            "",
            "Recommendations are trade-off defaults, not production configuration changes.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--provider", required=True)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--source-language", default="zh")
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("benchmarks"))
    parser.add_argument(
        "--suspicious-substitution",
        action="append",
        default=DEFAULT_SUSPICIOUS_SUBSTITUTIONS,
        metavar="EXPECTED:OBSERVED",
    )
    args = parser.parse_args()

    if not args.video.is_file():
        parser.error(f"video does not exist: {args.video}")
    reference = (
        args.reference.read_text(encoding="utf-8") if args.reference and args.reference.is_file() else None
    )
    if args.reference and reference is None:
        print(f"Warning: reference transcript not found; skipping CER/WER: {args.reference}")

    run_dir = args.output_dir / datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True)
    audio_path = run_dir / "audio.wav"
    audio_seconds = extract_audio(args.video, audio_path)
    substitutions = parse_substitutions(args.suspicious_substitution)

    results = []
    for model in args.models:
        print(f"Benchmarking {args.provider}/{model}...")
        result = benchmark_model(
            args.provider,
            model,
            audio_path,
            args.source_language,
            run_dir / model,
            reference,
            substitutions,
        )
        result["runtime"]["audio_extraction_seconds"] = audio_seconds
        write_json(run_dir / model / "runtime.json", result["runtime"])
        results.append(result)

    recommendations = recommendation(args.models)
    (run_dir / "comparison.md").write_text(
        comparison_markdown(args.provider, results, recommendations), encoding="utf-8"
    )
    print(f"Benchmark saved to: {run_dir}")


if __name__ == "__main__":
    main()
