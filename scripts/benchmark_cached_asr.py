"""Benchmark FasterWhisper configurations against one cached audio artifact."""

import argparse
import json
import time
from pathlib import Path

from src.infrastructure.runtime.cuda_dll_bootstrap import register_cuda_dll_directories


def warnings_for(segments: list[dict], audio_duration: float) -> list[str]:
    text = "".join(segment["text"].strip() for segment in segments)
    speech_duration = sum(segment["end"] - segment["start"] for segment in segments)
    warnings = []
    if len(text) <= 4:
        warnings.append("low_information_asr_output")
    if not text:
        warnings.append("empty_or_near_empty_transcript")
    if speech_duration and len(text) / speech_duration < 0.5:
        warnings.append("very_low_chars_per_second")
    if (
        len(segments) == 1
        and segments[0]["end"] - segments[0]["start"] >= audio_duration * 0.8
        and len(text) <= 10
    ):
        warnings.extend(
            ["single_long_segment_with_tiny_text", "duration_text_inconsistency"]
        )
    if any(segment["avg_logprob"] < -1.0 for segment in segments):
        warnings.append("low_avg_logprob")
    if any(segment["no_speech_prob"] > 0.6 for segment in segments):
        warnings.append("high_no_speech_probability")
    texts = [segment["text"].strip() for segment in segments if segment["text"].strip()]
    if len(texts) != len(set(texts)):
        warnings.append("repeated_hallucinated_text")
    return warnings


def benchmark(model, audio: Path, language: str, vad: bool, audio_duration: float) -> dict:
    started = time.perf_counter()
    raw_segments, info = model.transcribe(
        str(audio),
        language=None if language == "auto" else language,
        vad_filter=vad,
    )
    segments = [
        {
            "start": segment.start,
            "end": segment.end,
            "text": segment.text.strip(),
            "avg_logprob": segment.avg_logprob,
            "no_speech_prob": segment.no_speech_prob,
            "compression_ratio": segment.compression_ratio,
        }
        for segment in raw_segments
    ]
    runtime = time.perf_counter() - started
    text = "".join(segment["text"] for segment in segments)
    total_duration = sum(segment["end"] - segment["start"] for segment in segments)
    return {
        "language_mode": language,
        "detected_language": info.language,
        "language_probability": info.language_probability,
        "vad": vad,
        "runtime_seconds": runtime,
        "segment_count": len(segments),
        "total_speech_duration": total_duration,
        "average_segment_duration": total_duration / len(segments) if segments else 0.0,
        "max_segment_duration": max(
            (segment["end"] - segment["start"] for segment in segments),
            default=0.0,
        ),
        "raw_text": " ".join(segment["text"] for segment in segments),
        "text_length": len(text),
        "chars_per_second": len(text) / total_duration if total_duration else 0.0,
        "average_logprob": (
            sum(segment["avg_logprob"] for segment in segments) / len(segments)
            if segments else None
        ),
        "average_no_speech_prob": (
            sum(segment["no_speech_prob"] for segment in segments) / len(segments)
            if segments else None
        ),
        "average_compression_ratio": (
            sum(segment["compression_ratio"] for segment in segments) / len(segments)
            if segments else None
        ),
        "low_confidence_count": sum(
            segment["avg_logprob"] < -1.0 for segment in segments
        ),
        "hallucination_warnings": warnings_for(segments, audio_duration),
        "segments": segments,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--speech-report", type=Path)
    parser.add_argument("--models", nargs="+", default=["tiny", "base", "small"])
    args = parser.parse_args()

    import wave

    with wave.open(str(args.audio), "rb") as audio_file:
        audio_duration = audio_file.getnframes() / audio_file.getframerate()

    register_cuda_dll_directories()
    from faster_whisper import WhisperModel

    results = []
    for model_name in args.models:
        model = WhisperModel(model_name, device="cuda", compute_type="float16")
        configs = [("zh", False)] if model_name == "tiny" else [
            ("zh", False), ("zh", True), ("auto", False), ("auto", True)
        ]
        for language, vad in configs:
            result = benchmark(model, args.audio, language, vad, audio_duration)
            result["model"] = model_name
            results.append(result)
            print(
                f"{model_name}/{language}/vad={vad}: "
                f"{result['segment_count']} segments, {result['text_length']} chars"
            )
        del model

    report = {"audio": str(args.audio), "audio_duration": audio_duration, "results": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if args.speech_report and args.speech_report.is_file():
        speech_report = json.loads(args.speech_report.read_text(encoding="utf-8"))
        speech_report["asr_benchmark"] = report
        args.speech_report.write_text(
            json.dumps(speech_report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
