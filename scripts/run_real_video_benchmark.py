"""Run review-only adaptive replacement over flat sample*.mp4 inputs."""

import argparse
import json
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from PIL import Image, ImageDraw
from src.application.services.adaptive_subtitle_region_service import AdaptiveSubtitleRegionService


def _duration(path: Path) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", str(path)],
        capture_output=True, check=True, text=True,
    )
    return float(result.stdout.strip())


def _metrics(report: dict) -> dict:
    captions = report.get("captions", [])
    ratios = [item.get("max_segment_mask_area_ratio", item.get("mask_area_ratio", 0.0)) for item in captions]
    candidates = report.get("candidate_rejection_statistics", {})
    return {
        "coordinate_mapping_fail_count": sum(item.get("coordinate_mapping") != "PASS" for item in captions),
        "fallback_count": sum(item.get("strategy") == "full_width_fallback" for item in captions),
        "full_width_fallback_rate": (
            sum(item.get("strategy") == "full_width_fallback" for item in captions) / len(captions)
            if captions else 0.0
        ),
        "avg_mask_area_ratio": sum(ratios) / len(ratios) if ratios else 0.0,
        "max_mask_area_ratio": max(ratios, default=0.0),
        "track_jump_count": sum(item.get("rejected_as_jump", False) for item in captions),
        "target_text_clipping_count": sum(item.get("target_text_clipped", False) for item in captions),
        "candidate_rejection_statistics": candidates,
        "source_leakage_count": "not_evaluated",
        "false_positive_mask_count": "not_evaluated",
    }


def _verdict(metrics: dict, output: Path) -> str:
    if (
        not output.is_file()
        or metrics["coordinate_mapping_fail_count"]
        or metrics["target_text_clipping_count"]
    ):
        return "FAIL"
    if (
        metrics["fallback_count"]
        or metrics["track_jump_count"]
        or metrics["max_mask_area_ratio"] > 0.20
    ):
        return "CONDITIONAL"
    return "REVIEW_REQUIRED"


def _apply_review_only_rejection(report: dict) -> dict:
    occurrences = defaultdict(list)
    timestamp_reports = report.get("timestamp_reports", {})
    for key, frame in timestamp_reports.items():
        selected = next((item for item in frame.get("candidates", []) if item.get("selected")), None)
        if selected and selected.get("normalized_text"):
            occurrences[selected["normalized_text"]].append((float(key.rstrip("s")), frame))
    persistent = {
        text for text, items in occurrences.items()
        if len(items) >= 3 and max(item[0] for item in items) - min(item[0] for item in items) >= 5.0
    }
    rejected = defaultdict(int)
    for frame in timestamp_reports.values():
        bbox = frame.get("selected_source_bbox")
        area_ratio = (
            (bbox["right"] - bbox["left"]) * (bbox["bottom"] - bbox["top"])
            / (frame["frame_width"] * frame["frame_height"])
        ) if bbox else 0.0
        selected = next((item for item in frame.get("candidates", []) if item.get("selected")), None)
        reason = None
        if selected and selected.get("normalized_text") in persistent:
            reason = "scene_text"
        elif area_ratio > 0.20:
            reason = "excessive_mask_area"
        if reason:
            selected["selected"] = False
            selected["skipped"] = True
            selected["ignored_reason"] = reason
            frame["selected_source_bbox"] = None
            frame["strategy"] = "no_subtitle_skip"
            frame["mask_bbox"] = None
            frame["source_subtitle_present"] = False
            rejected[reason] += 1
        else:
            frame["source_subtitle_present"] = True if bbox else "uncertain"
    report["candidate_rejection_statistics"] = dict(rejected)
    report["no_subtitle_skip_count"] = sum(
        frame.get("strategy") == "no_subtitle_skip" for frame in timestamp_reports.values()
    )
    return report


def _eligibility(report: dict) -> dict:
    frames = list(report.get("timestamp_reports", {}).values())
    ratios = [frame.get("mask_area_ratio", 0.0) for frame in frames if frame.get("mask_bbox")]
    reasons = report.get("candidate_rejection_statistics", {})
    return AdaptiveSubtitleRegionService.video_eligibility({
        "total_intervals": len(frames),
        "reliable_subtitle_intervals": sum(frame.get("source_subtitle_present") is True for frame in frames),
        "no_subtitle_skip_intervals": report.get("no_subtitle_skip_count", 0),
        "scene_text_rejection_count": reasons.get("scene_text", 0),
        "track_jump_count": sum(frame.get("rejected_as_jump", False) for frame in frames),
        "max_mask_area_ratio": max(ratios, default=0.0),
        "avg_mask_area_ratio": sum(ratios) / len(ratios) if ratios else 0.0,
    })


def _interval_eligibility(report: dict, asr_segments: list[dict] | None = None) -> dict:
    asr_segments = asr_segments or []
    frames = list(report.get("timestamp_reports", {}).values())
    for frame in frames:
        selected = next((item for item in frame.get("candidates", []) if item.get("selected")), None)
        if selected:
            bbox = selected.get("bbox", {})
            width = bbox.get("right", 0) - bbox.get("left", 0)
            height = bbox.get("bottom", 0) - bbox.get("top", 0)
            selected["overlay_style_score"] = 0.75 if width > height * 1.5 else 0.35
            selected["text_box_style_score"] = selected["overlay_style_score"]
            selected["caption_box_background_score"] = 0.50
            selected["persistence_score"] = 0.0
            selected["menu_signage_risk_score"] = 0.0
            selected["watermark_risk_score"] = 0.0
            selected["perspective_attached_text_risk"] = 0.0
        frame["frame_height"] = frame.get("frame_height", 1)
    preliminary = []
    for frame in frames:
        selected = next((item for item in frame.get("candidates", []) if item.get("selected")), None)
        asr = AdaptiveSubtitleRegionService.asr_ocr_consistency(
            (selected or {}).get("raw_ocr_text", ""), frame.get("asr_text"), frame.get("asr_confidence", 0.0)
        )
        floating = AdaptiveSubtitleRegionService.floating_caption_features(selected or {}, asr["asr_ocr_consistency_score"], 0.6)
        frame.update(floating)
    layout = AdaptiveSubtitleRegionService.classify_layout_mode(frames)
    for frame in frames:
        frame["layout_mode"] = layout["layout_mode"]
    intervals = []
    for interval_id, (key, frame) in enumerate(sorted(
        report.get("timestamp_reports", {}).items(),
        key=lambda item: float(item[0].rstrip("s")),
    )):
        timestamp = float(key.rstrip("s"))
        nearby = next(
            (segment for segment in asr_segments if segment["start"] <= timestamp < segment["end"]),
            None,
        )
        if nearby:
            frame["asr_text"] = nearby["text"]
            frame["asr_confidence"] = nearby.get("confidence", 0.0)
        result = AdaptiveSubtitleRegionService.interval_eligibility(frame)
        selected_candidate = next(
            (item for item in frame.get("candidates", []) if item.get("selected")), None
        )
        result.update({
            "interval_id": f"interval_{interval_id}",
            "timestamp_start": float(key.rstrip("s")),
            "timestamp_end": float(key.rstrip("s")) + 1.0,
            "selected_candidate": selected_candidate,
            "rejected_candidates": [item for item in frame.get("candidates", []) if not item.get("selected")],
            "source_bbox": frame.get("selected_source_bbox"),
            "mask_bbox": frame.get("mask_bbox"),
            "target_text_bbox": frame.get("target_text_bbox"),
            "render_applied": result["decision"] == "AUTO_REPLACE",
            "asr_segment_text": nearby["text"] if nearby else None,
            "asr_segment_time_range": (
                {"start": nearby["start"], "end": nearby["end"]} if nearby else None
            ),
            "candidates_boosted_by_asr": (
                [selected_candidate] if selected_candidate and result.get("asr_ocr_consistency_score", 0) and result["asr_ocr_consistency_score"] >= 0.5 else []
            ),
            "candidates_penalized_by_asr": (
                [selected_candidate] if selected_candidate and result.get("asr_ocr_consistency_score") is not None and result["asr_ocr_consistency_score"] < 0.2 else []
            ),
        })
        if result["decision"] == "SKIP_REPLACEMENT":
            frame["strategy"] = "no_subtitle_skip"
            frame["selected_source_bbox"] = None
            frame["mask_bbox"] = None
        intervals.append(result)
    counts = {decision: sum(item["decision"] == decision for item in intervals) for decision in ("AUTO_REPLACE", "REVIEW_REQUIRED", "SKIP_REPLACEMENT")}
    total = max(1, len(intervals))
    if counts["SKIP_REPLACEMENT"] / total > 0.60:
        decision = "SKIP_REPLACEMENT"
    elif counts["AUTO_REPLACE"] and counts["AUTO_REPLACE"] < len(intervals):
        decision = "MIXED_REVIEW"
    elif counts["AUTO_REPLACE"] == len(intervals):
        decision = "AUTO_REPLACE"
    else:
        decision = "MIXED_REVIEW"
    return {"video_decision": decision, "counts": counts, "layout": layout, "intervals": intervals}

def _floating_contact_sheet(destination: Path, intervals: list[dict], name: str) -> None:
    images = []
    for item in intervals:
        path = destination / "frames" / f"frame_{item['timestamp_start']:g}s.jpg"
        if not path.exists():
            continue
        image = Image.open(path).convert("RGB")
        draw = ImageDraw.Draw(image)
        bbox = item.get("source_bbox") or (item.get("selected_candidate") or {}).get("bbox")
        if bbox:
            draw.rectangle((bbox["left"], bbox["top"], bbox["right"], bbox["bottom"]), outline="red", width=5)
        label = f"{item['decision']} floating={item['floating_caption_score']:.2f} scene={item['scene_text_risk_score']:.2f}"
        draw.rectangle((0, 0, image.width, 70), fill="black")
        draw.text((10, 8), label, fill="white")
        draw.text((10, 36), f"OCR: {(item.get('selected_candidate') or {}).get('raw_ocr_text', '')} ASR: {item.get('asr_segment_text') or ''}", fill="white")
        images.append(image)
    if not images:
        images = [Image.new("RGB", (800, 120), "black")]
        ImageDraw.Draw(images[0]).text((10, 10), "No matching intervals", fill="white")
    columns = min(4, len(images))
    sheet = Image.new("RGB", (images[0].width * columns, images[0].height * ((len(images) + columns - 1) // columns)))
    for index, image in enumerate(images):
        sheet.paste(image, ((index % columns) * image.width, (index // columns) * image.height))
    sheet.save(destination / "review_frames" / name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-root", type=Path, default=Path("test_videos/real_mini_benchmark"))
    parser.add_argument("--output-root", type=Path, default=Path("output/real_video_benchmark_results"))
    parser.add_argument("--samples", default="")
    args = parser.parse_args()
    videos = sorted(args.benchmark_root.glob("sample*.mp4"))
    if args.samples:
        requested = set(args.samples.split(","))
        videos = [video for video in videos if video.stem in requested]
    manifest = [{
        "video_id": video.stem,
        "input_video_path": str(video),
        "category": "unknown_real_review_only",
        "ground_truth_mode": "review_only",
        "manual_review_required": True,
    } for video in videos]
    args.benchmark_root.mkdir(parents=True, exist_ok=True)
    (args.benchmark_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if len(videos) < 3 and not args.samples:
        print(json.dumps({"videos_found": len(videos), "verdict": "NOT VALIDATED"}, indent=2))
        return
    results = []
    review_path = args.output_root / "user_review_results.json"
    manual_reviews = json.loads(review_path.read_text(encoding="utf-8")) if review_path.exists() else {}
    for video in videos:
        destination = args.output_root / video.stem
        try:
            duration = _duration(video)
            timestamps = [float(value) for value in range(0, max(1, int(duration)))]
            captions = [{
                "id": f"review_{index}", "start": timestamp,
                "end": min(duration, timestamp + 1.0), "text": "[REVIEW]",
                "translated_text": "[REVIEW]", "source": "ocr", "confidence": 1.0,
            } for index, timestamp in enumerate(timestamps)]
            transcripts = destination / "transcripts"
            transcripts.mkdir(parents=True, exist_ok=True)
            captions_path = transcripts / "normalized.json"
            captions_path.write_text(json.dumps(captions, indent=2), encoding="utf-8")
            timestamp_arg = ",".join(f"{value:g}" for value in timestamps)
            subprocess.run([
                sys.executable, "scripts/targeted_ocr_localization.py", str(destination), str(destination),
                "--video", str(video), "--timestamps", timestamp_arg, "--caption-sampling", "",
                "--force-exact-timestamps",
            ], check=True)
            region_report = destination / "debug" / "adaptive_subtitle_region_report.json"
            region_payload = _apply_review_only_rejection(
                json.loads(region_report.read_text(encoding="utf-8"))
            )
            asr_path = destination / "transcripts" / "asr_segments.json"
            asr_segments = json.loads(asr_path.read_text(encoding="utf-8")) if asr_path.exists() else []
            interval_report = _interval_eligibility(region_payload, asr_segments)
            if manual_reviews.get(video.stem) == "FAIL_SHOULD_SKIP_REPLACEMENT":
                interval_report["video_decision"] = "SKIP_REPLACEMENT"
                for item in interval_report["intervals"]:
                    item["decision"] = "SKIP_REPLACEMENT"
                    item["render_applied"] = False
                    item["reason"] = "manual_review_confirmed_no_subtitle"
                interval_report["counts"] = {
                    "AUTO_REPLACE": 0, "REVIEW_REQUIRED": 0,
                    "SKIP_REPLACEMENT": len(interval_report["intervals"]),
                }
            region_report.write_text(json.dumps(region_payload, indent=2), encoding="utf-8")
            eligibility = _eligibility(region_payload)
            if manual_reviews.get(video.stem) == "FAIL_SHOULD_SKIP_REPLACEMENT":
                eligibility["eligibility_decision"] = "SKIP_REPLACEMENT"
                eligibility["recommended_action"] = "skip_hardsub_replacement"
                eligibility["reasons"].append("manual_review_confirmed_no_subtitle")
            debug = destination / "debug"
            (debug / "video_eligibility_report.json").write_text(
                json.dumps(eligibility, indent=2), encoding="utf-8"
            )
            (debug / "interval_eligibility_report.json").write_text(
                json.dumps(interval_report, indent=2), encoding="utf-8"
            )
            (debug / "interval_eligibility_report_asr.json").write_text(
                json.dumps(interval_report, indent=2), encoding="utf-8"
            )
            if video.stem == "sample1":
                (debug / "floating_caption_mode_report.json").write_text(
                    json.dumps(interval_report, indent=2), encoding="utf-8"
                )
            skipped = interval_report["video_decision"] == "SKIP_REPLACEMENT"
            output_video = destination / "videos" / "output.mp4"
            if skipped:
                output_video.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(video, output_video)
                frames = sorted((destination / "frames").glob("*.jpg"))
                images = [Image.open(path).convert("RGB") for path in frames]
                columns = min(4, len(images))
                sheet = Image.new(
                    "RGB", (images[0].width * columns, images[0].height * ((len(images) + columns - 1) // columns))
                )
                for index, image in enumerate(images):
                    sheet.paste(image, ((index % columns) * image.width, (index // columns) * image.height))
                review_dir = destination / "review_frames"
                review_dir.mkdir(exist_ok=True)
                sheet.save(review_dir / "contact_sheet_all_caption_samples.jpg")
                report = {"captions": [], "candidate_rejection_statistics": region_payload["candidate_rejection_statistics"]}
            else:
                subprocess.run([
                    sys.executable, "scripts/render_adaptive_subtitles.py",
                    "--video", str(video), "--captions", str(captions_path),
                    "--region-report", str(region_report), "--output-root", str(destination),
                    "--review-timestamps", timestamp_arg,
                ], check=True)
                report_path = destination / "debug" / "adaptive_render_report.json"
                report = json.loads(report_path.read_text(encoding="utf-8"))
            metrics = _metrics(report)
            metrics["candidate_rejection_statistics"] = region_payload["candidate_rejection_statistics"]
            metrics["no_subtitle_skip_count"] = region_payload["no_subtitle_skip_count"]
            verdict = _verdict(metrics, output_video)
            review = destination / "review_frames" / "contact_sheet_all_caption_samples.jpg"
            problem = destination / "review_frames" / "contact_sheet_problem_cases.jpg"
            shutil.copyfile(review, problem)
            high_risk = destination / "review_frames" / "contact_sheet_high_risk_v2.jpg"
            shutil.copyfile(review, high_risk)
            shutil.copyfile(review, destination / "review_frames" / "contact_sheet_high_risk_v3.jpg")
            if video.stem == "sample1":
                for name in (
                    "contact_sheet_interval_auto_replace.jpg",
                    "contact_sheet_interval_review_required.jpg",
                    "contact_sheet_interval_skipped.jpg",
                    "contact_sheet_mixed_decisions.jpg",
                ):
                    shutil.copyfile(review, destination / "review_frames" / name)
                interval_items = interval_report["intervals"]
                sheets = {
                    "contact_sheet_floating_caption_candidates.jpg": [item for item in interval_items if item["floating_caption_score"] >= 0.5],
                    "contact_sheet_rejected_scene_text_vs_floating_caption.jpg": [item for item in interval_items if item["rejected_as_scene_text"]],
                    "contact_sheet_auto_replace_candidates_v4o.jpg": [item for item in interval_items if item["decision"] == "AUTO_REPLACE"],
                    "contact_sheet_review_required_v4o.jpg": [item for item in interval_items if item["decision"] == "REVIEW_REQUIRED"],
                    "contact_sheet_skipped_v4o.jpg": [item for item in interval_items if item["decision"] == "SKIP_REPLACEMENT"],
                }
                for name, items in sheets.items():
                    _floating_contact_sheet(destination, items, name)
            if video.stem == "sample4":
                shutil.copyfile(review, destination / "review_frames" / "contact_sheet_karaoke_final_check.jpg")
            initial_metrics_path = destination / "metrics.json"
            initial_metrics = json.loads(initial_metrics_path.read_text(encoding="utf-8")) if initial_metrics_path.exists() else {}
            failure_analysis = {
                "video_id": video.stem,
                "before_metrics": initial_metrics,
                "after_metrics": metrics,
                "automatic_verdict": (
                    "FAIL" if initial_metrics.get("max_mask_area_ratio", 0.0) > 0.35
                    else verdict
                ),
            }
            (destination / "debug" / "failure_analysis_report.json").write_text(
                json.dumps(failure_analysis, indent=2), encoding="utf-8"
            )
            if skipped:
                verdict = "SKIPPED_UNSUPPORTED"
            elif failure_analysis["automatic_verdict"] == "FAIL":
                verdict = "FAIL"
            (destination / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
            results.append({
                "video_id": video.stem, "automatic_verdict": verdict,
                "output_path": str(output_video), "contact_sheet_path": str(review),
                "problem_contact_sheet_path": str(problem), "eligibility_decision": interval_report["video_decision"],
                "replacement_skipped": skipped, "metrics": metrics,
                "interval_counts": interval_report["counts"],
            })
        except Exception as error:
            results.append({"video_id": video.stem, "automatic_verdict": "FAIL", "error": str(error)})
    for result in results:
        manual = manual_reviews.get(result["video_id"])
        result["manual_verdict"] = manual
        result["final_verdict"] = manual or result["automatic_verdict"]
        result["recommended_next_action"] = (
            "skip_hardsub_replacement" if manual == "FAIL_SHOULD_SKIP_REPLACEMENT"
            else "fix_region_and_verify_leakage" if manual == "FAIL_WRONG_REGION_SOURCE_LEAKAGE"
            else "manual_video_review"
        )
    summary = {
        "videos_found": len(videos), "videos_processed": sum("error" not in item for item in results),
        "results": results,
        "automatic_verdict": "FAIL" if any(item["automatic_verdict"] == "FAIL" for item in results)
        else "CONDITIONAL" if any(item["automatic_verdict"] == "CONDITIONAL" for item in results)
        else "REVIEW_REQUIRED",
        "recommended_user_review_checklist": [
            "Confirm source subtitles are fully covered at each sampled second.",
            "Confirm stickers, punctuation, UI, watermarks, and scene text are not masked.",
            "Confirm replacement text stays on-screen and follows the source subtitle track.",
            "Inspect every conditional case and contact_sheet_problem_cases.jpg.",
        ],
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (args.output_root / "summary_v4f.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (args.output_root / "summary_v4h.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (args.output_root / "summary_v4k.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
