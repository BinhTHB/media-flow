import argparse
import json
from pathlib import Path
from statistics import mean

import yaml


def load_manifest(path: Path) -> list[dict]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    samples = data.get("samples", [])
    if not isinstance(samples, list):
        raise ValueError("Manifest samples must be a list.")
    required = {"id", "path", "source_language", "target_language", "frame_interval"}
    for sample in samples:
        missing = required - sample.keys()
        if missing:
            raise ValueError(f"Sample {sample.get('id', '<unknown>')} missing: {sorted(missing)}")
    return samples


def count_overlaps(segments: list[dict]) -> tuple[int, int]:
    ordered = sorted(segments, key=lambda segment: segment["start"])
    overlaps = sum(1 for left, right in zip(ordered, ordered[1:]) if right["start"] < left["end"])
    events = sorted(
        [(segment["start"], 1) for segment in ordered]
        + [(segment["end"], -1) for segment in ordered],
        key=lambda event: (event[0], event[1]),
    )
    active = maximum = 0
    for _, delta in events:
        active += delta
        maximum = max(maximum, active)
    return overlaps, maximum


def extract_metrics(job_dir: Path) -> dict:
    debug = job_dir / "debug"
    transcripts = job_dir / "transcripts"
    metadata_path = debug / "debug_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    quality = metadata.get("quality_metrics", {})
    segments_path = transcripts / "segmented.json"
    segments = json.loads(segments_path.read_text(encoding="utf-8")) if segments_path.exists() else []
    overlaps, simultaneous = count_overlaps(segments)
    report_path = debug / "content_quality_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8")).get("captions", []) if report_path.exists() else []
    scores = [caption["score"] for caption in report]
    return {
        "output_exists": (job_dir / "videos" / "output.mp4").exists(),
        "subtitle_exists": (job_dir / "subtitles" / "subtitle.srt").exists(),
        "caption_count": len(segments),
        "content_quality_scores": scores,
        "placeholder_count": sum("placeholder_only" in item["findings"] for item in report),
        "source_leakage_count": quality.get("source_text_leakage_count"),
        "overlap_count": overlaps,
        "max_simultaneous": simultaneous,
        "captions_beyond_duration": sum(segment["end"] > quality.get("video_duration", float("inf")) for segment in segments),
        "ocr_checkpoints": len(list((debug / "ocr_frames").glob("*.json"))),
        "subtitle_region_checkpoints": len(list((debug / "subtitle_region_frames").glob("*.json"))),
        "quality_gate_status": quality.get("quality_gate_status"),
    }


def classify(sample: dict, metrics: dict) -> str:
    if not metrics["output_exists"] or not metrics["subtitle_exists"]:
        return "fail"
    checks = {
        "caption_count": (metrics["caption_count"], sample.get("expected_min_caption_count", 0), "min"),
        "source_leakage_count": (metrics["source_leakage_count"], sample.get("expected_max_source_leakage", 0), "max"),
        "overlap_count": (metrics["overlap_count"], sample.get("expected_max_overlap", 0), "max"),
        "max_simultaneous": (metrics["max_simultaneous"], sample.get("expected_max_simultaneous", 1), "max"),
        "captions_beyond_duration": (metrics["captions_beyond_duration"], sample.get("expected_max_captions_beyond_duration", 0), "max"),
    }
    for _, (actual, expected, direction) in checks.items():
        if actual is None or (direction == "min" and actual < expected) or (direction == "max" and actual > expected):
            return "conditional"
    return "pass" if metrics["quality_gate_status"] == "PASS" else "conditional"


def aggregate(results: list[dict]) -> dict:
    statuses = [result["status"] for result in results]
    scores = [score for result in results for score in result["metrics"].get("content_quality_scores", [])]
    captions = [result["metrics"]["caption_count"] for result in results]
    return {
        "samples_passed": statuses.count("pass"),
        "samples_failed": statuses.count("fail"),
        "samples_conditional": statuses.count("conditional"),
        "average_caption_count": mean(captions) if captions else 0.0,
        "average_content_quality_score": mean(scores) if scores else 0.0,
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="benchmarks/quality_samples.yaml")
    parser.add_argument("--job-prefix", default="quality_benchmark")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--collect-job-id")
    args = parser.parse_args()
    samples = load_manifest(Path(args.manifest))
    if args.dry_run:
        print(json.dumps({"samples": samples, "mode": "dry-run"}, ensure_ascii=False, indent=2))
        return
    if not args.collect_job_id:
        raise SystemExit("Only --dry-run and --collect-job-id are supported; execution is intentionally explicit.")
    sample = samples[0]
    metrics = extract_metrics(Path("output") / args.collect_job_id)
    result = {"sample_id": sample["id"], "job_id": args.collect_job_id, "metrics": metrics, "status": classify(sample, metrics)}
    report = aggregate([result])
    Path("output/quality_benchmark_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()