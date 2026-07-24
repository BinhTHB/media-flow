import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from statistics import median

from PIL import Image, ImageDraw

from src.application.services.adaptive_subtitle_region_service import (
    AdaptiveSubtitleRegionService,
    BoundingBox,
    TextBox,
)
from src.domain.entities.frame import Frame
from src.infrastructure.providers.subtitle.paddle_subtitle_region_detector_provider import (
    PaddleOCRSubtitleRegionDetectorProvider,
    polygon_bbox,
    transform_polygon,
)

def _timestamps(value: str) -> list[float]:
    return [float(item) for item in value.split(",") if item.strip()]

def _validate_ocr_device(requested: str) -> dict:
    if requested == "cpu":
        return {
            "python_executable": sys.executable,
            "ocr_device_requested": "cpu",
            "ocr_device_used": "cpu",
            "paddle_cuda_available": False,
            "paddle_device": "cpu",
            "fallback_to_cpu": False,
            "fallback_reason": None,
        }
    try:
        import paddle

        cuda_available = bool(paddle.device.is_compiled_with_cuda())
        paddle_device = str(paddle.device.get_device())
    except Exception as error:
        raise RuntimeError(
            "OCR_DEVICE_GPU_REQUIRED_BUT_UNAVAILABLE"
        ) from error
    if not cuda_available or not paddle_device.lower().startswith("gpu"):
        raise RuntimeError("OCR_DEVICE_GPU_REQUIRED_BUT_UNAVAILABLE")
    return {
        "python_executable": sys.executable,
        "ocr_device_requested": requested,
        "ocr_device_used": "gpu",
        "paddle_cuda_available": True,
        "paddle_device": paddle_device,
        "fallback_to_cpu": False,
        "fallback_reason": None,
    }


def _extract_frame(video_path: Path, timestamp: float, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-ss", str(timestamp), "-i", str(video_path), "-frames:v", "1", "-q:v", "2", "-y", str(path)],
        check=True, capture_output=True, text=True,
    )


def _draw_bbox(draw: ImageDraw.ImageDraw, bbox: dict, color: str, width: int = 4) -> None:
    draw.rectangle((bbox["left"], bbox["top"], bbox["right"], bbox["bottom"]), outline=color, width=width)


def _box_from_record(box: dict) -> TextBox:
    bbox = box["bbox"]
    return TextBox(
        BoundingBox(bbox["left"], bbox["top"], bbox["right"], bbox["bottom"]),
        box["confidence"], box.get("text", ""), box.get("polygon", []), box.get("source", "paddleocr"),
    )


def _frame_report(record: dict, service: AdaptiveSubtitleRegionService) -> dict:
    width, height = record["frame_width"], record["frame_height"]
    candidate_reports = []
    provider_selected = []
    for box in record.get("boxes", []):
        classification = service.classify_candidate(box.get("text", ""))
        ignored_reason = classification["ignored_reason"] or box.get("ignored_reason")
        candidate = {
            "raw_ocr_text": box.get("text", ""),
            **classification,
            "selected": ignored_reason is None,
            "ignored_reason": ignored_reason,
            "subtitle_track_score": None,
            "style_score": None,
            "temporal_consistency_score": None,
            "final_candidate_score": (
                round(box["confidence"] * classification["letter_or_digit_ratio"], 4)
                if ignored_reason is None else 0.0
            ),
            "bbox": box["bbox"],
        }
        candidate_reports.append(candidate)
        if ignored_reason is None:
            provider_selected.append(box)
    provider_ignored = [
        {**box, "ignored_reason": box["ignored_reason"]}
        for box in record.get("boxes", []) if box.get("ignored_reason") is not None
    ]
    boxes = [_box_from_record(box) for box in provider_selected]
    selected, ignored = service.filter_boxes(boxes, width, height)
    ignored = provider_ignored + ignored
    lines = service.group_lines(selected)
    blocks = service.group_blocks(lines)
    block = max(blocks, key=lambda item: item.bbox.width * item.bbox.height) if blocks else None
    reliability = service.reliability_score(block, 1.0, width, height)
    strategy = service.select_strategy(block.bbox if block else None, reliability, width, height)
    padding = max(12, round((block.bbox.height if block else 0) * 0.15))
    if block:
        bbox = block.bbox
        mask = BoundingBox(max(0, bbox.left - padding), max(0, bbox.top - padding), min(width, bbox.right + padding), min(height, bbox.bottom + padding * 1.5))
        area_ratio = mask.area / (width * height)
        mask_dict = mask.to_dict()
    else:
        area_ratio, mask_dict = 1.0, None
    return {
        "actual_sampled_timestamp": record["frame_timestamp"],
        "frame_width": width, "frame_height": height,
        "selected_source_bbox": block.bbox.to_dict() if block else None,
        "selected_source_boxes": [box.to_dict() for box in selected],
        "ignored_boxes": ignored,
        "selected_lines": [line.to_dict() for line in lines],
        "selected_block": block.to_dict() if block else None,
        "candidates": candidate_reports,
        "strategy": strategy,
        "mask_bbox": mask_dict,
        "mask_area_ratio": round(area_ratio, 6),
        "target_text_bbox": mask_dict,
        "reliability_score": reliability,
        "temporal_consistency_score": None,
        "fallback_reason": "missing_or_unreliable_detection" if strategy == "full_width_fallback" else None,
        "production_style_replacement": service.production_style_replacement(strategy, area_ratio),
    }


def _apply_track_prior(reports: dict[str, dict]) -> None:
    stable = [
        report for report in reports.values()
        if report.get("selected_source_bbox")
        and any(candidate["text_classification"] == "meaningful_text" and candidate["selected"] for candidate in report["candidates"])
        and (report["selected_source_bbox"]["right"] - report["selected_source_bbox"]["left"]) / report["frame_width"] >= 0.20
    ]
    if not stable:
        return
    points = [
        (report["actual_sampled_timestamp"], BoundingBox(**report["selected_source_bbox"]))
        for report in stable
    ]
    service = AdaptiveSubtitleRegionService()
    track = service.classify_track(points, stable[0]["frame_height"])
    dynamic = track["track_mode"] in {"slowly_moving_track", "moving_track"}
    track_y = sorted(
        (item["selected_source_bbox"]["top"] + item["selected_source_bbox"]["bottom"]) / 2
        for item in stable
    )[len(stable) // 2]
    track_height = sorted(
        item["selected_source_bbox"]["bottom"] - item["selected_source_bbox"]["top"]
        for item in stable
    )[len(stable) // 2]
    track_width = sorted(
        item["selected_source_bbox"]["right"] - item["selected_source_bbox"]["left"]
        for item in stable
    )[len(stable) // 2]
    for report in reports.values():
        bbox = report.get("selected_source_bbox")
        report.update({
            "track_mode": track["track_mode"],
            "y_center_trajectory": track["y_center_trajectory"],
            "track_velocity": track["track_velocity"],
            "moving_track_score": track["moving_track_score"],
            "rejected_as_jump": False,
            "accepted_as_moving_track": False,
            "interpolation_used": False,
        })
        expected = service.interpolate_track_bbox(report["actual_sampled_timestamp"], points, track)
        if dynamic and bbox:
            actual = BoundingBox(**bbox)
            trajectory_distance = abs(actual.y_center - expected.y_center)
            trajectory_score = max(0.0, 1.0 - trajectory_distance / max(track["trajectory_tolerance"], 1.0))
            for candidate in report["candidates"]:
                candidate["trajectory_score"] = round(trajectory_score, 4)
                candidate["subtitle_track_score"] = round(trajectory_score, 4)
            if trajectory_score >= 0.55:
                report["accepted_as_moving_track"] = True
                continue
        if not bbox:
            if dynamic and expected:
                report["selected_source_bbox"] = expected.to_dict()
                report["strategy"] = "local_band"
                report["fallback_reason"] = "moving_track_interpolated"
                report["interpolation_used"] = True
            continue
        center_y = (bbox["top"] + bbox["bottom"]) / 2
        distance = abs(center_y - track_y)
        track_score = max(0.0, 1.0 - distance / max(track_height * 3, report["frame_height"] * 0.15))
        for candidate in report["candidates"]:
            candidate["subtitle_track_score"] = round(track_score, 4)
        if track_score >= 0.35:
            continue
        half_width, half_height = track_width / 2, track_height / 2
        center_x = report["frame_width"] / 2
        report["selected_source_bbox"] = {
            "left": max(0.0, center_x - half_width),
            "top": max(0.0, track_y - half_height),
            "right": min(float(report["frame_width"]), center_x + half_width),
            "bottom": min(float(report["frame_height"]), track_y + half_height),
        }
        report["strategy"] = "local_band"
        report["fallback_reason"] = "track_jump_interpolated"
        report["production_style_replacement"] = "CONDITIONAL"
        report["rejected_as_jump"] = True
        for candidate in report["candidates"]:
            if candidate["selected"]:
                candidate["selected"] = False
                candidate["ignored_reason"] = "track_jump"

def _static_region_report(reports: dict[str, dict]) -> dict:
    frames = list(reports.values())
    selected = [item for item in frames if item.get("selected_source_bbox")]
    candidate_count = sum(len(item.get("candidates", [])) for item in frames)
    rejected = [
        candidate
        for item in frames
        for candidate in item.get("candidates", [])
        if not candidate.get("selected")
    ]
    base = {
        "candidate_count": candidate_count,
        "rejected_scene_text_count": sum(
            item.get("ignored_reason") in {"scene_text", "signage_text", "menu_text"}
            for item in rejected
        ),
        "rejected_watermark_count": sum(
            item.get("ignored_reason") in {"watermark", "logo"}
            for item in rejected
        ),
        "rejected_person_attached_count": sum(
            item.get("ignored_reason") == "person_attached_text" for item in rejected
        ),
    }
    if len(selected) < 2:
        return {
            **base, "stable_region_found": False, "stable_region_bbox": None,
            "stable_region_position": None, "stability_score": 0.0,
            "frame_coverage_ratio": round(len(selected) / max(1, len(frames)), 4),
            "reason_if_no_stable_region": "insufficient_temporal_coverage",
        }
    boxes = [BoundingBox(**item["selected_source_bbox"]) for item in selected]
    aggregate, position_stability = AdaptiveSubtitleRegionService().aggregate(boxes)
    assert aggregate is not None
    widths = [box.width for box in boxes]
    heights = [box.height for box in boxes]
    size_stability = max(
        0.0,
        1.0 - (
            median(abs(value - median(widths)) for value in widths) / max(median(widths), 1)
            + median(abs(value - median(heights)) for value in heights) / max(median(heights), 1)
        ) / 2,
    )
    coverage = len(selected) / max(1, len(frames))
    style_scores = [
        candidate.get("final_candidate_score", 0.0)
        for item in selected
        for candidate in item.get("candidates", [])
        if candidate.get("selected")
    ]
    style_consistency = sum(style_scores) / len(style_scores) if style_scores else 0.0
    stability = round(
        0.40 * position_stability + 0.25 * size_stability
        + 0.25 * coverage + 0.10 * style_consistency,
        4,
    )
    track_modes = {item.get("track_mode") for item in selected}
    found = (
        coverage >= 0.30 and position_stability >= 0.65 and size_stability >= 0.65
        and not track_modes.intersection({"moving_track", "slowly_moving_track"})
        and stability >= 0.65
    )
    width, height = selected[0]["frame_width"], selected[0]["frame_height"]
    x, y = aggregate.x_center / width, aggregate.y_center / height
    position = (
        "top" if y < 0.33 else "bottom" if y > 0.67 else
        "left" if x < 0.33 else "right" if x > 0.67 else "middle"
    )
    reason = None
    if not found:
        reason = (
            "dynamic_or_moving_region"
            if track_modes.intersection({"moving_track", "slowly_moving_track"})
            else "insufficient_region_stability"
        )
    return {
        **base, "stable_region_found": found,
        "stable_region_bbox": aggregate.to_dict() if found else None,
        "stable_region_position": position if found else None,
        "stability_score": stability, "position_stability_score": round(position_stability, 4),
        "size_stability_score": round(size_stability, 4),
        "style_consistency_score": round(style_consistency, 4),
        "frame_coverage_ratio": round(coverage, 4),
        "reason_if_no_stable_region": reason,
    }


def _write_overlay(record: dict, report: dict, path: Path) -> None:
    image = Image.open(record["source_image_path"]).convert("RGB")
    draw = ImageDraw.Draw(image)
    for box in report["selected_source_boxes"]:
        _draw_bbox(draw, box["bbox"], "lime")
    for box in report["ignored_boxes"]:
        _draw_bbox(draw, box["bbox"], "orange", 2)
    if report["selected_source_bbox"]:
        _draw_bbox(draw, report["selected_source_bbox"], "cyan", 5)
    if report["mask_bbox"]:
        _draw_bbox(draw, report["mask_bbox"], "red", 5)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def _write_coordinate_sanity(record: dict, output_dir: Path) -> None:
    timestamp = f"{record['frame_timestamp']:g}s"
    source = Image.open(record["source_image_path"]).convert("RGB")
    output_dir.mkdir(parents=True, exist_ok=True)
    source.save(output_dir / f"frame_{timestamp}_raw.jpg")
    transforms = {
        "identity": "identity",
        "vflip": "vertical_flip",
        "hflip": "horizontal_flip",
        "rot180": "rotate180",
    }
    for suffix, transform in transforms.items():
        image = source.copy()
        draw = ImageDraw.Draw(image)
        for box in record.get("boxes", []):
            polygon = transform_polygon(
                box.get("raw_polygon_from_provider", box["polygon"]),
                transform,
                record["frame_width"],
                record["frame_height"],
            )
            _draw_bbox(draw, polygon_bbox(polygon), "magenta", 4)
        image.save(output_dir / f"frame_{timestamp}_{suffix}_overlay.jpg")
    image = source.copy()
    draw = ImageDraw.Draw(image)
    for box in record.get("boxes", []):
        _draw_bbox(draw, box.get("transformed_bbox", box["bbox"]), "lime", 5)
    image.save(output_dir / f"frame_{timestamp}_best_transform_overlay.jpg")


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact_root", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--video", type=Path, default=Path("sample.mp4"))
    parser.add_argument("--timestamps", default="5,10,15", type=_timestamps)
    parser.add_argument("--caption-sampling", default="start,mid,end")
    parser.add_argument("--force-exact-timestamps", action="store_true")
    parser.add_argument("--output-debug-overlays", action="store_true")
    parser.add_argument("--ocr-device", choices=("gpu", "cpu", "auto"), default="gpu")
    return parser

def main() -> None:
    args = _argument_parser().parse_args()
    device_report = _validate_ocr_device(args.ocr_device)
    normalized = json.loads((args.artifact_root / "transcripts" / "normalized.json").read_text(encoding="utf-8"))
    requested = set(args.timestamps)
    if args.caption_sampling:
        for caption in normalized:
            start, end = float(caption["start"]), float(caption["end"])
            if "start" in args.caption_sampling:
                requested.add(start)
            if "mid" in args.caption_sampling:
                requested.add((start + end) / 2)
            if "end" in args.caption_sampling:
                requested.add(end)
    video_path = args.video
    if not video_path.exists():
        raise FileNotFoundError(f"Source video does not exist: {video_path}")
    target_frames = args.output_root / "frames"
    frames = []
    for timestamp in sorted(requested):
        path = target_frames / f"frame_{timestamp:g}s.jpg"
        _extract_frame(video_path, timestamp, path)
        frames.append(Frame(timestamp=timestamp, image_path=str(path)))
    started = time.monotonic()
    PaddleOCRSubtitleRegionDetectorProvider(
        use_gpu=device_report["ocr_device_used"] == "gpu"
    ).detect_regions(frames)
    elapsed = time.monotonic() - started
    records = [json.loads(path.read_text(encoding="utf-8")) for path in sorted((args.output_root / "debug" / "subtitle_region_frames").glob("*.json"))]
    service = AdaptiveSubtitleRegionService()
    reports = {f"{record['frame_timestamp']:g}s": _frame_report(record, service) for record in records}
    _apply_track_prior(reports)
    for caption in normalized:
        start, end = float(caption["start"]), float(caption["end"])
        caption_reports = [
            reports[f"{record['frame_timestamp']:g}s"]
            for record in records
            if start <= float(record["frame_timestamp"]) <= end
        ]
        temporal_boxes = [
            BoundingBox(**item["selected_source_bbox"])
            for item in caption_reports
            if item["selected_source_bbox"]
        ]
        _, consistency = service.aggregate(temporal_boxes)
        for item in caption_reports:
            item["temporal_consistency_score"] = round(consistency, 4)
    if args.output_debug_overlays:
        overlay_dir = args.output_root / "debug" / "adaptive_overlays"
        for record in records:
            key = f"{record['frame_timestamp']:g}s"
            overlay = overlay_dir / f"frame_{key}.jpg"
            _write_overlay(record, reports[key], overlay)
            if record["frame_timestamp"] in args.timestamps:
                _write_coordinate_sanity(
                    record, args.output_root / "debug" / "coordinate_sanity"
                )
        overlay_paths = [overlay_dir / f"frame_{timestamp:g}s.jpg" for timestamp in args.timestamps]
        if overlay_paths:
            images = [Image.open(path).convert("RGB") for path in overlay_paths]
            width, height = images[0].size
            contact = Image.new("RGB", (width, height * len(images)))
            for index, image in enumerate(images):
                contact.paste(image, (0, index * height))
            contact.save(overlay_dir / "contact_sheet_5s_10s_15s.jpg")
    report = {
        "schema_version": 1, "frames_re_ocred": len(records),
        **device_report,
        "ocr_elapsed_seconds": round(elapsed, 3),
        "ocr_runtime_seconds": round(
            sum(record.get("processing_duration", 0.0) for record in records) or elapsed,
            3,
        ),
        "individual_boxes_extracted": sum(len(record.get("boxes", [])) for record in records),
        "frames": records, "timestamp_reports": reports,
        "fallback_count": sum(item["strategy"] == "full_width_fallback" for item in reports.values()),
        **_static_region_report(reports),
    }
    debug = args.output_root / "debug"
    debug.mkdir(parents=True, exist_ok=True)
    for name in ("precise_subtitle_localization.json", "adaptive_subtitle_region_report.json"):
        (debug / name).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
