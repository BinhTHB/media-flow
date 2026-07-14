import json
from pathlib import Path
from src.domain.entities.segment import Segment
from src.application.use_cases.normalize_segments import NormalizeSegmentsUseCase
from src.application.use_cases.subtitle_segmentation import SubtitleSegmentationUseCase
from src.application.services.subtitle_segmentation_service import SubtitleSegmentationService
from src.application.services.speech_quality_auditor import SpeechQualityAuditor

def load_segments(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [
        Segment(
            id=item["id"],
            start=item["start"],
            end=item["end"],
            text=item["text"],
            translated_text=item["translated_text"],
            source=item["source"],
            confidence=item["confidence"],
        )
        for item in data
    ]

def main():
    translated_path = Path("output/paddle311_phase3_validation_20260712_v15/transcripts/translated.json")
    if not translated_path.exists():
        print("Error: translated.json not found")
        return
        
    segments = load_segments(translated_path)
    translated_segments = load_segments(translated_path)
    asr_segments = load_segments(translated_path.parent / "transcript.json")
        
    print(f"Loaded {len(segments)} segments before normalization.")
    for s in segments:
        print(f"  - {s.id} ({s.source}): {s.start:.3f} -> {s.end:.3f} | {s.translated_text}")

    normalizer = NormalizeSegmentsUseCase()
    normalized = normalizer.execute(segments, is_translated=True, video_duration=22.616236)
    job_dir = translated_path.parent.parent
    with open(job_dir / "transcripts" / "normalized.json", "w", encoding="utf-8") as f:
        json.dump([segment.to_dict() for segment in normalized], f, ensure_ascii=False, indent=2)
    
    print(f"\nAfter Normalization (kept {len(normalized)} segments):")
    for s in normalized:
        print(f"  - {s.id} ({s.source}): {s.start:.3f} -> {s.end:.3f} | {s.translated_text}")
        
    print("\nNormalization Metrics:")
    print(json.dumps(normalizer.metrics, indent=2))
    
    segmenter = SubtitleSegmentationUseCase(SubtitleSegmentationService())
    final = segmenter.execute(normalized, is_translated=True)
    with open(job_dir / "transcripts" / "segmented.json", "w", encoding="utf-8") as f:
        json.dump([segment.to_dict() for segment in final], f, ensure_ascii=False, indent=2)

    debug_path = job_dir / "debug" / "debug_metadata.json"
    with open(debug_path, "r", encoding="utf-8") as f:
        debug_metadata = json.load(f)
    debug_metadata.setdefault("quality_metrics", {}).update(normalizer.metrics)
    speech_report = SpeechQualityAuditor.report(
        asr_segments,
        [segment for segment in translated_segments if segment.source == "speech"],
        translated_segments,
        "zh",
    )
    debug_metadata["quality_metrics"].update({
        key: value
        for key, value in speech_report.items()
        if key not in ("asr_segments", "translated_speech_segments", "warnings")
    })
    debug_metadata["quality_metrics"]["speech_quality_warnings"] = speech_report["warnings"]
    with open(debug_path, "w", encoding="utf-8") as f:
        json.dump(debug_metadata, f, ensure_ascii=False, indent=2)
    with open(debug_path.parent / "speech_quality_report.json", "w", encoding="utf-8") as f:
        json.dump(speech_report, f, ensure_ascii=False, indent=2)
    
    print(f"\nFinal SRT captions ({len(final)}):")
    for idx, s in enumerate(final, 1):
        print(f"{idx}\n{s.start:.3f} --> {s.end:.3f}\n{s.translated_text}\n")

    overlap_count = sum(
        left.end > right.start for left, right in zip(final, final[1:])
    )
    print(json.dumps({
        "overlap_count": overlap_count,
        "max_simultaneous_captions": 1 if overlap_count == 0 else 2,
        "captions_beyond_duration_count": sum(
            segment.end > 22.616236 for segment in final
        ),
    }, indent=2))

if __name__ == "__main__":
    main()