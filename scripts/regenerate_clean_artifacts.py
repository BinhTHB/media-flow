"""Generate Static Subtitle Cover MVP artifacts from source videos only."""

import hashlib
import json
import shutil
import subprocess
import sys
import argparse
from pathlib import Path

from scripts.run_real_video_benchmark import _apply_review_only_rejection, _interval_eligibility

def _static_region_report(timestamp_reports):
    """Placeholder: PaddleOCR-based static region report removed."""
    return {}


ROOT = Path(__file__).resolve().parents[1]
SAMPLE_VIDEOS = {
    name: ROOT / "test_videos" / "real_mini_benchmark" / f"{name}.mp4"
    for name in ("sample2", "sample4", "sample5")
}


def _run(args: list[str]) -> None:
    subprocess.run(args, cwd=ROOT, check=True)


def _probe(video: Path) -> dict:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_entries",
            "format=duration:stream=codec_type,width,height,avg_frame_rate",
            "-of", "json", str(video),
        ],
        check=True, capture_output=True, text=True,
    )
    payload = json.loads(result.stdout)
    stream = next(item for item in payload["streams"] if item["codec_type"] == "video")
    numerator, denominator = stream["avg_frame_rate"].split("/")
    return {
        "resolution": {"width": stream["width"], "height": stream["height"]},
        "fps": round(float(numerator) / float(denominator), 6),
        "duration": round(float(payload["format"]["duration"]), 6),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

def _artifact_metadata(video: Path, metadata: dict, digest: str, ocr_device: str) -> dict:
    return {
        "input_video_path": str(video.relative_to(ROOT)),
        "input_video_sha256": digest,
        **metadata,
        "artifact_schema_version": "static-source-cover-mvp-v2",
        "ocr_device_requested": ocr_device,
        "ocr_device_used": ocr_device,
        "python_executable": sys.executable,
        "cache_reused": False,
        "cache_validated": False,
        "cache_reject_reason": "fresh_source_run",
    }

def _person_risk_evidence(video: Path, intervals: dict) -> dict:
    import cv2

    detector = cv2.HOGDescriptor()
    detector.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    capture = cv2.VideoCapture(str(video))
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    evidence = {}
    for interval in intervals["intervals"]:
        capture.set(cv2.CAP_PROP_POS_FRAMES, round(interval["timestamp_start"] * fps))
        ok, frame = capture.read()
        people = []
        if ok:
            boxes, _ = detector.detectMultiScale(
                frame, winStride=(8, 8), padding=(8, 8), scale=1.05
            )
            people = [
                {"left": int(x), "top": int(y), "right": int(x + w), "bottom": int(y + h)}
                for x, y, w, h in boxes
            ]
        candidate = interval.get("selected_candidate") or {}
        bbox = candidate.get("bbox") or interval.get("source_bbox")
        overlap = bool(
            bbox and any(
                bbox["left"] < person["right"] and bbox["right"] > person["left"]
                and bbox["top"] < person["bottom"] and bbox["bottom"] > person["top"]
                for person in people
            )
        )
        interval["person_risk_score"] = 1.0 if overlap else 0.0
        interval["person_risk_evidence"] = {
            "detector": "opencv_hog_people",
            "detected_person_boxes": people,
            "candidate_overlap": overlap,
        }
        evidence[interval["interval_id"]] = interval["person_risk_evidence"]
    capture.release()
    return evidence


def _write_srt(path: Path, captions: list[dict]) -> None:
    def stamp(value: float) -> str:
        milliseconds = round(value * 1000)
        hours, remainder = divmod(milliseconds, 3_600_000)
        minutes, remainder = divmod(remainder, 60_000)
        seconds, milliseconds = divmod(remainder, 1000)
        return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"

    blocks = []
    for index, item in enumerate(captions, 1):
        target = item.get("translated_text")
        text = target if target and target != "[REVIEW]" else f"DỊCH_{index}"
        blocks.append(
            f"{index}\n{stamp(item['start'])} --> {stamp(item['end'])}\n{text}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")

def _review_timestamps(duration: float) -> str:
    return ",".join(
        f"{min(duration - 0.1, duration * ratio):g}" for ratio in (0.2, 0.4, 0.6, 0.8)
    )

def _copy(path: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, destination)

def _generate_sample(
    sample: str, video: Path, output: Path, ocr_device: str
) -> dict:
    if not video.is_file():
        raise FileNotFoundError(f"Missing input video: {video}")
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"Refusing to mix artifacts with existing output: {output}")
    metadata, digest = _probe(video), _sha256(video)
    timestamps = [float(second) for second in range(int(metadata["duration"]) + 1)]
    captions = [
        {
            "id": f"{sample}_{index}", "start": timestamp,
            "end": min(metadata["duration"], timestamp + 1.0),
            "text": "[REVIEW]", "translated_text": "[REVIEW]",
            "source": "ocr", "confidence": 1.0,
        }
        for index, timestamp in enumerate(timestamps)
    ]
    captions_path = output / "transcripts" / "normalized.json"
    _write_json(captions_path, captions)
    timestamp_arg = ",".join(f"{item:g}" for item in timestamps)
    _run([
        sys.executable, "-m", "scripts.targeted_ocr_localization",
        str(output), str(output), "--video", str(video),
        "--timestamps", timestamp_arg, "--caption-sampling", "",
        "--force-exact-timestamps", "--output-debug-overlays",
        "--ocr-device", ocr_device,
    ])
    region_path = output / "debug" / "adaptive_subtitle_region_report.json"
    region = _apply_review_only_rejection(
        json.loads(region_path.read_text(encoding="utf-8"))
    )
    region.update(_static_region_report(region["timestamp_reports"]))
    stable = {
        key: region.get(key)
        for key in (
            "stable_region_found", "stable_region_bbox", "stable_region_position",
            "stability_score", "frame_coverage_ratio", "candidate_count",
            "rejected_scene_text_count", "rejected_watermark_count",
            "rejected_person_attached_count", "reason_if_no_stable_region",
        )
    }
    _write_json(output / "debug" / "static_source_cover_report.json", stable)
    _write_json(region_path, region)

    intervals = _interval_eligibility(region, [])
    stable_bbox = stable["stable_region_bbox"]
    for interval in intervals["intervals"]:
        interval.update({
            "source_bbox": stable_bbox,
            "stability_score": stable["stability_score"],
            "static_region_found": stable["stable_region_found"],
            "fixed_bottom_fallback": False,
            "mask_area_ratio": (
                (stable_bbox["right"] - stable_bbox["left"])
                * (stable_bbox["bottom"] - stable_bbox["top"])
                / (metadata["resolution"]["width"] * metadata["resolution"]["height"])
                if stable_bbox else 0.0
            ),
            "scene_text_risk_score": 0.0,
            "watermark_risk_score": 0.0,
            "face_overlap_score": 0.0,
        })
    _person_risk_evidence(video, intervals)
    interval_path = output / "debug" / "interval_decision_report.json"
    _write_json(
        interval_path,
        {**intervals, **_artifact_metadata(video, metadata, digest, ocr_device)},
    )
    review_timestamps = _review_timestamps(metadata["duration"])
    videos = output / "videos"
    cover_rendered = False
    if stable["stable_region_found"]:
        cover_root = output / "_static_cover"
        _run([
            sys.executable, "-m", "scripts.render_adaptive_subtitles",
            "--video", str(video), "--captions", str(captions_path),
            "--region-report", str(region_path), "--interval-report", str(interval_path),
            "--output-root", str(cover_root), "--coverage-mode", "cover_source_subtitle",
            "--review-timestamps", review_timestamps, "--ffmpeg-encoder", "auto",
        ])
        _copy(cover_root / "videos" / "output.mp4", videos / "static_source_cover.mp4")
        _copy(
            cover_root / "review_frames" / "contact_sheet_all_caption_samples.jpg",
            output / "review_frames" / "contact_sheet_static_source_cover_output.jpg",
        )
        cover_report = json.loads(
            (cover_root / "debug" / "cover_source_subtitle_report.json").read_text(
                encoding="utf-8"
            )
        )
        rendered = sum(item.get("rendered", False) for item in cover_report["captions"])
        skipped = len(cover_report["captions"]) - rendered
        rendered_reports = [
            item for item in cover_report["captions"] if item.get("rendered")
        ]
        expansion = rendered_reports[0] if rendered_reports else {}
        person_overlap_audit = [
            {
                "interval_id": item.get("interval_id"),
                "time_range": item.get("time_range"),
                "source_bbox": item.get("source_subtitle_bbox"),
                "person_overlap_score": item.get("person_overlap_score"),
                "face_overlap_score": item.get("face_overlap_score"),
                "scene_text_overlap_score": item.get("scene_text_overlap_score"),
                "watermark_overlap_score": item.get("watermark_overlap_score"),
                "original_subtitle_occupies_region": bool(
                    item.get("source_subtitle_bbox")
                ),
                "static_region_person_overlap_override": item.get(
                    "static_region_person_overlap_override", False
                ),
                "override_reason": item.get("override_reason"),
                "keep_reject_reason": item.get("keep_reject_reason"),
                "visual_risk_label": (
                    "controlled_non_face_person_overlap"
                    if item.get("static_region_person_overlap_override")
                    else "true_person_overlap_risk"
                    if item.get("keep_reject_reason") == "true_person_overlap_risk"
                    else "none"
                ),
            }
            for item in cover_report["captions"]
            if item.get("person_overlap_score", 0.0) >= 0.35
        ]
        stable.update({
            "status": "rendered" if rendered else "skipped_no_safe_intervals",
            "rendered_count": rendered, "skipped_count": skipped,
            "original_stable_region_bbox": expansion.get(
                "original_stable_region_bbox", stable_bbox
            ),
            "expanded_cover_region_bbox": expansion.get(
                "expanded_cover_region_bbox"
            ),
            "expansion_px_x": expansion.get("expansion_px_x"),
            "expansion_px_y": expansion.get("expansion_px_y"),
            "expansion_reason": expansion.get("expansion_reason"),
            "cover_overflow_risk": expansion.get("cover_overflow_risk"),
            "person_overlap_audit": person_overlap_audit,
            "target_text_source_distribution": {
                "placeholder": sum(
                    item.get("target_text_source") == "placeholder"
                    for item in cover_report["captions"]
                )
            },
        })
        stable["sample5_rendered_after_override"] = (
            rendered if sample == "sample5" else None
        )
        stable["sample5_skipped_after_override"] = (
            skipped if sample == "sample5" else None
        )
        cover_rendered = rendered > 0
    else:
        stable.update({
            "status": "skipped_no_stable_region", "rendered_count": 0,
            "skipped_count": len(captions),
            "target_text_source_distribution": {"placeholder": len(captions)},
        })

    fallback_root = output / "_static_burn"
    _run([
        sys.executable, "-m", "scripts.render_adaptive_subtitles",
        "--video", str(video), "--captions", str(captions_path),
        "--region-report", str(region_path), "--output-root", str(fallback_root),
        "--coverage-mode", "full_coverage_fallback",
        "--allow-fallback-caption-placement", "--review-timestamps", review_timestamps,
        "--ffmpeg-encoder", "auto",
    ])
    _copy(fallback_root / "videos" / "output.mp4", videos / "static_subtitle_burn.mp4")
    _copy(
        fallback_root / "review_frames" / "contact_sheet_all_caption_samples.jpg",
        output / "review_frames" / "contact_sheet_static_subtitle_burn_output.jpg",
    )
    _copy(
        fallback_root / "subtitles" / "subtitle.ass",
        output / "subtitles" / "output.ass",
    )
    _write_srt(output / "subtitles" / "output.srt", captions)
    detection_sheet = output / "debug" / "adaptive_overlays" / "contact_sheet_5s_10s_15s.jpg"
    if detection_sheet.exists():
        _copy(
            detection_sheet,
            output / "review_frames" / "contact_sheet_static_region_detection.jpg",
        )
        _copy(
            detection_sheet,
            output / "review_frames" / "contact_sheet_policy_decisions.jpg",
        )
    _write_json(output / "debug" / "static_source_cover_report.json", stable)
    provenance = {
        **_artifact_metadata(video, metadata, digest, ocr_device),
        "sample": sample, "subtitle_output_mode": "static_source_cover",
        "fallback_generated": True,
        "static_source_cover_generated": cover_rendered,
        "dynamic_subtitle_replacement": "OUT_OF_SCOPE",
        "production_ready": "NO",
        "generated_artifacts": {
            "static_source_cover": (
                "videos/static_source_cover.mp4" if cover_rendered else None
            ),
            "static_subtitle_burn": "videos/static_subtitle_burn.mp4",
            "sidecars": ["subtitles/output.ass", "subtitles/output.srt"],
            "contact_sheets": "review_frames",
        },
    }
    _write_json(output / "debug" / "artifact_provenance_report.json", provenance)
    return {**stable, **provenance}

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--subtitle-output-mode",
        choices=("static_source_cover",), default="static_source_cover",
    )
    parser.add_argument("--ocr-device", choices=("gpu", "cpu"), default="gpu")
    parser.add_argument("--samples", default="sample2,sample4,sample5")
    parser.add_argument(
        "--output-root", type=Path,
        default=ROOT / "output" / "static_source_cover_mvp",
    )
    args = parser.parse_args()
    samples = [item.strip() for item in args.samples.split(",") if item.strip()]
    unknown = sorted(set(samples) - SAMPLE_VIDEOS.keys())
    if unknown:
        raise ValueError(f"Unknown samples: {', '.join(unknown)}")
    results = {}
    for sample in samples:
        results[sample] = _generate_sample(
            sample, SAMPLE_VIDEOS[sample],
            args.output_root / sample,
            args.ocr_device,
        )
    _write_json(
        args.output_root / "summary.json", results
    )

if __name__ == "__main__":
    main()
