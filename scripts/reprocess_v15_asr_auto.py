"""Reprocess cached v15 audio and OCR using auto-detected speech language."""

import json
import math
import shutil
import time
from pathlib import Path

from src.application.services.content_quality_auditor import ContentQualityAuditor
from src.application.services.source_text_leakage_detector import SourceTextLeakageDetector
from src.application.services.speech_quality_auditor import SpeechQualityAuditor
from src.application.services.subtitle_segmentation_service import SubtitleSegmentationService
from src.application.services.vietnamese_text_quality_auditor import (
    VietnameseTextQualityAuditor,
)
from src.application.use_cases.group_segments import GroupSegmentsUseCase
from src.application.use_cases.merge_segments import MergeSegmentsUseCase
from src.application.use_cases.normalize_segments import NormalizeSegmentsUseCase
from src.application.use_cases.subtitle_segmentation import SubtitleSegmentationUseCase
from src.domain.entities.segment import Segment
from src.infrastructure.providers.subtitle.ass_subtitle_provider import ASSSubtitleProvider
from src.infrastructure.runtime.cuda_dll_bootstrap import register_cuda_dll_directories


SOURCE_JOB = Path("output/paddle311_phase3_validation_20260712_v15")
OUTPUT_JOB = Path("output/paddle311_phase3_validation_20260712_v15_asr_auto_1e")
VIDEO_DURATION = 22.616236


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")

def load_segments(path: Path) -> list[Segment]:
    return [
        Segment(
            id=item["id"],
            start=item["start"],
            end=item["end"],
            text=item["text"],
            translated_text=item.get("translated_text", ""),
            source=item["source"],
            confidence=item["confidence"],
        )
        for item in json.loads(path.read_text(encoding="utf-8"))
    ]


def load_cached_ocr() -> list[Segment]:
    filtered = json.loads(
        (SOURCE_JOB / "debug/filtered_ocr.json").read_text(encoding="utf-8")
    )["segments"]
    cached_translations = {
        item["id"]: item["translated_text"]
        for item in json.loads(
            (SOURCE_JOB / "transcripts/translated.json").read_text(encoding="utf-8")
        )
        if item["source"] == "ocr"
    }
    return [
        Segment(
            id=item["id"],
            start=item["start"],
            end=item["end"],
            text=item["text"],
            translated_text=cached_translations.get(item["id"], item["text"]),
            source="ocr",
            confidence=item["confidence"],
        )
        for item in filtered
    ]


def transcribe() -> tuple[list[Segment], dict]:
    register_cuda_dll_directories()
    from faster_whisper import WhisperModel

    model = WhisperModel("small", device="cuda", compute_type="float16")
    started = time.perf_counter()
    raw_segments, info = model.transcribe(
        str(SOURCE_JOB / "audio/audio.wav"),
        language=None,
        vad_filter=False,
    )
    raw_segments = list(raw_segments)
    runtime = time.perf_counter() - started
    segments = [
        Segment(
            id=f"speech_{index}",
            start=segment.start,
            end=segment.end,
            text=segment.text.strip(),
            translated_text=segment.text.strip(),
            source="speech",
            confidence=max(0.0, min(math.exp(segment.avg_logprob), 1.0)),
        )
        for index, segment in enumerate(raw_segments)
    ]
    metrics = {
        "model": "small",
        "language_mode": "auto",
        "detected_language": info.language,
        "language_probability": info.language_probability,
        "vad": False,
        "runtime_seconds": runtime,
        "average_logprob": (
            sum(segment.avg_logprob for segment in raw_segments) / len(raw_segments)
            if raw_segments else None
        ),
        "average_no_speech_prob": (
            sum(segment.no_speech_prob for segment in raw_segments) / len(raw_segments)
            if raw_segments else None
        ),
        "average_compression_ratio": (
            sum(segment.compression_ratio for segment in raw_segments) / len(raw_segments)
            if raw_segments else None
        ),
    }
    return segments, metrics


def main() -> None:
    transcript_path = OUTPUT_JOB / "transcripts/transcript.json"
    speech_report_path = OUTPUT_JOB / "debug/speech_quality_report.json"
    if transcript_path.is_file() and speech_report_path.is_file():
        speech = load_segments(transcript_path)
        previous_speech_report = json.loads(
            speech_report_path.read_text(encoding="utf-8")
        )
        asr_metrics = previous_speech_report["asr_metrics"]
    else:
        speech, asr_metrics = transcribe()
    ocr = load_cached_ocr()
    for directory in ("transcripts", "subtitles", "debug", "audio"):
        (OUTPUT_JOB / directory).mkdir(parents=True, exist_ok=True)
    shutil.copy2(SOURCE_JOB / "audio/audio.wav", OUTPUT_JOB / "audio/audio.wav")
    write_json(transcript_path, [s.to_dict() for s in speech])

    merged = MergeSegmentsUseCase().execute(speech, ocr)
    grouped = GroupSegmentsUseCase().execute(merged, "zh")
    for segment in grouped:
        if segment.source == "speech":
            segment.translated_text = segment.text
    write_json(OUTPUT_JOB / "transcripts/translated.json", [s.to_dict() for s in grouped])

    speech_report = SpeechQualityAuditor.report(
        speech, speech, speech, "auto", detected_language=asr_metrics["detected_language"]
    )
    speech_report["asr_metrics"] = asr_metrics
    speech_report["translation_policy"] = "Vietnamese ASR passed through unchanged"
    write_json(speech_report_path, speech_report)

    normalizer = NormalizeSegmentsUseCase()
    normalized = normalizer.execute(grouped, is_translated=True, video_duration=VIDEO_DURATION)
    write_json(
        OUTPUT_JOB / "transcripts/normalized.json",
        [segment.to_dict() for segment in normalized],
    )
    final = SubtitleSegmentationUseCase(SubtitleSegmentationService()).execute(
        normalized, is_translated=True
    )
    vietnamese_report = []
    corrections_applied = 0
    for index, segment in enumerate(final, 1):
        audit = VietnameseTextQualityAuditor.audit(segment.translated_text)
        if audit["corrections"]:
            segment.translated_text = audit["suggested_text"]
            audit["actual_action"] = "correct"
            corrections_applied += len(audit["corrections"])
        vietnamese_report.append({
            "caption_index": index,
            "timestamp": f"{segment.start:.3f} --> {segment.end:.3f}",
            **audit,
        })
    write_json(
        OUTPUT_JOB / "transcripts/segmented.json",
        [segment.to_dict() for segment in final],
    )
    write_json(
        OUTPUT_JOB / "debug/vietnamese_text_quality_report.json",
        {
            "captions": vietnamese_report,
            "warnings_count": sum(
                bool(item["corrections"]) for item in vietnamese_report
            ),
            "corrections_applied": corrections_applied,
            "mode": "correction-enabled",
        },
    )

    content_report = []
    for index, segment in enumerate(final, 1):
        score, findings = ContentQualityAuditor.audit(
            segment.translated_text, segment.source
        )
        content_report.append({
            "caption_index": index,
            "timestamp": f"{segment.start:.3f} --> {segment.end:.3f}",
            "original_text": segment.translated_text,
            "score": score,
            "findings": findings,
            "recommended_action": "drop" if findings else "keep",
            "actual_action": "keep",
            "reason": "report-only audit",
        })
    write_json(
        OUTPUT_JOB / "debug/content_quality_report.json",
        {"captions": content_report},
    )
    srt = ASSSubtitleProvider().generate_srt(final)
    (OUTPUT_JOB / "subtitles/subtitle.srt").write_text(srt, encoding="utf-8")

    overlaps = sum(left.end > right.start for left, right in zip(final, final[1:]))
    leakage = SourceTextLeakageDetector.detect(final, "zh")
    metrics = {
        **normalizer.metrics,
        **{key: value for key, value in speech_report.items() if not isinstance(value, list)},
        "final_caption_count": len(final),
        "overlap_count": overlaps,
        "max_simultaneous_captions": 1 if overlaps == 0 else 2,
        "captions_beyond_duration_count": sum(
            segment.end > VIDEO_DURATION for segment in final
        ),
        "source_leakage_count": len(leakage),
        "content_quality_findings_count": sum(
            bool(item["findings"]) for item in content_report
        ),
        "vietnamese_text_warnings_count": sum(
            bool(item["corrections"]) for item in vietnamese_report
        ),
    }
    write_json(OUTPUT_JOB / "debug/debug_metadata.json", {"quality_metrics": metrics})
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
