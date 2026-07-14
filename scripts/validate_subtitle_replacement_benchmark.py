"""Validate adaptive subtitle replacement artifacts against benchmark manifests."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


CATEGORIES = {
    "bottom_stable_subtitle", "center_subtitle", "top_subtitle", "moving_subtitle",
    "two_line_subtitle", "three_line_subtitle", "white_subtitle", "yellow_subtitle",
    "subtitle_with_shadow", "sticker_noise", "punctuation_noise", "ui_overlay",
    "watermark", "scene_text", "portrait_video", "landscape_video",
    "rotated_metadata_video", "bright_background", "complex_background",
}


def iou(left: dict, right: dict) -> float:
    overlap_width = max(0.0, min(left["right"], right["right"]) - max(left["left"], right["left"]))
    overlap_height = max(0.0, min(left["bottom"], right["bottom"]) - max(left["top"], right["top"]))
    overlap = overlap_width * overlap_height
    union = (
        (left["right"] - left["left"]) * (left["bottom"] - left["top"])
        + (right["right"] - right["left"]) * (right["bottom"] - right["top"])
        - overlap
    )
    return overlap / union if union else 0.0


def load_manifest(root: Path) -> list[dict]:
    path = root / "benchmark_manifest.json"
    if not path.exists():
        path = root / "manifest.json"
    if not path.exists():
        return []
    samples = json.loads(path.read_text(encoding="utf-8")).get("samples", [])
    if not isinstance(samples, list):
        raise ValueError("manifest.json samples must be a list.")
    for sample in samples:
        if not {"id", "path", "categories"} <= sample.keys():
            raise ValueError(f"Invalid benchmark sample: {sample}")
        unknown = set(sample["categories"]) - CATEGORIES
        if unknown:
            raise ValueError(f"Unknown benchmark categories: {sorted(unknown)}")
    return samples


def _ground_truth(root: Path, sample: dict) -> dict:
    path = root / sample.get("ground_truth", f"{sample['id']}/ground_truth.json")
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def compute_metrics(report: dict, ground_truth: dict) -> dict:
    captions = report.get("captions", [])
    mask_ratios = [caption.get("mask_area_ratio", 0.0) for caption in captions]
    fallbacks = [caption for caption in captions if caption.get("strategy") == "full_width_fallback"]
    forbidden = ground_truth.get("forbidden_regions", [])
    expected = ground_truth.get("expected_source_subtitle_bboxes", {})
    false_positive = 0
    leakage = 0
    clipping = 0
    track_jumps = 0
    mapping_failures = 0
    statuses = []
    for caption in captions:
        mask = caption.get("mask_bbox")
        source = caption.get("source_bbox")
        if caption.get("source_leakage_detected") is True:
            leakage += 1
        if caption.get("target_text_clipped") is True:
            clipping += 1
        if caption.get("coordinate_mapping") != "PASS":
            mapping_failures += 1
        if caption.get("fallback_reason") == "track_jump_interpolated":
            track_jumps += 1
        if mask and any(iou(mask, region["bbox"]) > 0.05 for region in forbidden):
            false_positive += 1
        expected_bbox = expected.get(caption.get("caption_id"))
        if expected_bbox and (not source or iou(source, expected_bbox) < 0.10):
            leakage += 1
        statuses.append(caption.get("production_style_replacement") == "PASS")
    return {
        "source_leakage_count": leakage,
        "false_positive_mask_count": false_positive,
        "fallback_count": len(fallbacks),
        "full_width_fallback_rate": len(fallbacks) / len(captions) if captions else 0.0,
        "mask_area_ratio_avg": sum(mask_ratios) / len(mask_ratios) if mask_ratios else 0.0,
        "mask_area_ratio_max": max(mask_ratios, default=0.0),
        "track_jump_count": track_jumps,
        "target_text_clipping_count": clipping,
        "coordinate_mapping_fail_count": mapping_failures,
        "production_style_pass_rate": sum(statuses) / len(statuses) if statuses else 0.0,
    }


def classify(metrics: dict) -> str:
    if any(metrics[key] for key in (
        "source_leakage_count", "false_positive_mask_count",
        "target_text_clipping_count", "coordinate_mapping_fail_count",
    )):
        return "FAIL"
    if metrics["fallback_count"] or metrics["production_style_pass_rate"] < 1.0:
        return "CONDITIONAL"
    return "PASS"


def summarize(results: list[dict]) -> dict:
    category_results = defaultdict(list)
    for result in results:
        for category in result["categories"]:
            category_results[category].append(result["status"])
    category_rates = {
        category: round(statuses.count("PASS") / len(statuses), 4)
        for category, statuses in category_results.items()
    }
    statuses = [result["status"] for result in results]
    return {
        "per_video": results,
        "per_category_pass_rate": category_rates,
        "global_robustness_verdict": (
            "FAIL" if "FAIL" in statuses else "CONDITIONAL" if "CONDITIONAL" in statuses else "PASS"
        ) if statuses else "CONDITIONAL",
        "failure_examples": [result["id"] for result in results if result["status"] == "FAIL"],
        "recommended_fixes": (
            ["Add representative videos and ground_truth.json before claiming cross-video robustness."]
            if not results else []
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-root", type=Path, default=Path("test_videos/subtitle_replacement_benchmark"))
    parser.add_argument("--output-root", type=Path, default=Path("output/subtitle_replacement_benchmark_results"))
    args = parser.parse_args()
    samples = load_manifest(args.benchmark_root)
    results = []
    for sample in samples:
        report_path = args.output_root / sample["id"] / "debug" / "adaptive_render_report.json"
        if not report_path.exists():
            results.append({"id": sample["id"], "categories": sample["categories"], "status": "CONDITIONAL", "metrics": {}, "reason": "missing_render_report"})
            continue
        report = json.loads(report_path.read_text(encoding="utf-8"))
        metrics = compute_metrics(report, _ground_truth(args.benchmark_root, sample))
        results.append({"id": sample["id"], "categories": sample["categories"], "status": classify(metrics), "metrics": metrics})
    summary = summarize(results)
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
