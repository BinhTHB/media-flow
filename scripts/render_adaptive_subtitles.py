import argparse
import json
import subprocess
from pathlib import Path
from statistics import median

from PIL import Image, ImageDraw

from src.application.services.ass_generator import ASSGenerator
from src.application.services.adaptive_subtitle_region_service import BoundingBox
from src.application.services.subtitle_render_policy import SubtitleRenderPolicy
from src.domain.entities.segment import Segment
from src.domain.entities.subtitle_placement import SubtitlePlacement
from src.domain.entities.subtitle_style import SubtitleStyle
from src.domain.entities.resource_policy import FFmpegConfig, ResourcePolicy
from src.infrastructure.runtime.ffmpeg_capability_detector import detect_ffmpeg_capabilities
from src.infrastructure.runtime.ffmpeg_encode import build_ffmpeg_encode_args
from src.infrastructure.runtime.resource_usage_report import write_resource_usage_report
from src.infrastructure.providers.video.ffmpeg_video_provider import FFmpegVideoProvider
from src.shared.exceptions import ValidationError

def _timestamps(value: str) -> list[float]:
    return [float(item) for item in value.split(",") if item.strip()]


def _bbox(values: dict) -> BoundingBox:
    return BoundingBox(values["left"], values["top"], values["right"], values["bottom"])


def _mask_for_caption(samples: list[dict], width: int, height: int) -> tuple[dict, str, str | None]:
    valid = [sample for sample in samples if sample.get("selected_source_bbox")]
    if not valid:
        return (
            {"left": 0.0, "top": height * 0.72, "right": float(width), "bottom": float(height)},
            "full_width_fallback",
            "no_coordinate_validated_source_bbox",
        )
    dynamic = any(
        sample.get("track_mode") in {"slowly_moving_track", "moving_track"}
        for sample in valid
    )
    if not dynamic:
        valid = [max(valid, key=lambda sample: sample.get("reliability_score", 0.0))]
    boxes = [_bbox(sample["selected_source_bbox"]) for sample in valid]
    source = BoundingBox.union(boxes) if dynamic else BoundingBox(
        median(box.left for box in boxes), median(box.top for box in boxes),
        median(box.right for box in boxes), median(box.bottom for box in boxes),
    )
    strategy = "expanded_line_bbox" if all(sample["strategy"] == "expanded_line_bbox" for sample in valid) else "local_band"
    text_height = max(box.height for box in boxes)
    padding = max(18, round(text_height * 0.15))
    return (
        {
            "left": max(0.0, source.left - padding),
            "top": max(0.0, source.top - padding),
            "right": min(float(width), source.right + padding),
            "bottom": min(float(height), source.bottom + padding * 1.5),
        },
        strategy,
        None,
    )


def _placement_for_caption(mask: dict, strategy: str) -> SubtitlePlacement:
    return SubtitlePlacement(
        alignment=5,
        x=round((mask["left"] + mask["right"]) / 2),
        y=round((mask["top"] + mask["bottom"]) / 2),
        mask_bbox=mask,
        is_fallback=strategy == "full_width_fallback",
    )

def _padded_cover_bbox(
    source: dict,
    width: int,
    height: int,
    *,
    stable_region_bbox: dict | None = None,
) -> dict:
    res_factor = max(1.0, height / 1080.0)
    stroke_margin = 8.0 * res_factor
    text_width = source["right"] - source["left"]
    text_height = source["bottom"] - source["top"]
    is_bubble = text_width / max(1.0, text_height) > 4.0
    bubble_ratio_h = 0.12 if is_bubble else 0.06
    bubble_ratio_v = 0.35 if is_bubble else 0.20
    horizontal_pad = max(30.0 * res_factor, text_width * bubble_ratio_h) + stroke_margin
    vertical_pad = max(18.0 * res_factor, text_height * bubble_ratio_v) + stroke_margin
    left = source["left"] - horizontal_pad
    right = source["right"] + horizontal_pad
    top = source["top"] - vertical_pad
    bottom = source["bottom"] + vertical_pad
    if stable_region_bbox:
        left = min(left, stable_region_bbox["left"] - stroke_margin)
        right = max(right, stable_region_bbox["right"] + stroke_margin)
        top = min(top, stable_region_bbox["top"] - stroke_margin)
        bottom = max(bottom, stable_region_bbox["bottom"] + stroke_margin)
    return {
        "left": max(0.0, left),
        "top": max(0.0, top),
        "right": min(float(width), right),
        "bottom": min(float(height), bottom),
    }

def _bbox_iou(source: dict, cover: dict) -> float:
    left, top = max(source["left"], cover["left"]), max(source["top"], cover["top"])
    right, bottom = min(source["right"], cover["right"]), min(source["bottom"], cover["bottom"])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    source_area = (source["right"] - source["left"]) * (source["bottom"] - source["top"])
    cover_area = (cover["right"] - cover["left"]) * (cover["bottom"] - cover["top"])
    return round(intersection / (source_area + cover_area - intersection), 6) if intersection else 0.0

def _cover_safety_evidence(
    interval: dict | None,
    *,
    has_target_text: bool,
    width: int,
    height: int,
) -> dict:
    interval = interval or {}
    source = interval.get("source_bbox")
    source_area = (
        (source["right"] - source["left"]) * (source["bottom"] - source["top"])
        if _valid_bbox(source, width, height) else 0.0
    )
    bbox_area_ratio = source_area / (width * height) if source_area else 0.0
    evidence = interval.get("person_risk_evidence") or {}
    overlaps = []
    for person in evidence.get("detected_person_boxes", []):
        if not source:
            continue
        intersection = max(
            0.0, min(source["right"], person["right"]) - max(source["left"], person["left"])
        ) * max(
            0.0, min(source["bottom"], person["bottom"]) - max(source["top"], person["top"])
        )
        if intersection:
            overlaps.append(round(intersection / source_area, 6))
    person_overlap = max(overlaps, default=0.0)
    scene = interval.get("scene_text_risk_score")
    watermark = interval.get("watermark_risk_score")
    face = interval.get("face_overlap_score")
    mask = interval.get("mask_area_ratio")
    non_person_safe = (
        isinstance(scene, (int, float)) and scene < 0.20
        and isinstance(watermark, (int, float)) and watermark < 0.20
        and (face is None or isinstance(face, (int, float)) and face < 0.20)
    )
    base_safe = (
        source_area > 0 and has_target_text and non_person_safe
        and isinstance(mask, (int, float)) and mask <= 0.20
        and 0.001 <= bbox_area_ratio <= 0.20
    )
    person_risk = interval.get("person_risk_score")
    baseline_allowed = base_safe and isinstance(person_risk, (int, float)) and person_risk < 0.20
    override = (
        base_safe
        and isinstance(person_risk, (int, float)) and person_risk >= 0.20
        and bool(evidence.get("detected_person_boxes"))
        and person_overlap < 0.35
    )
    stable_score = interval.get("stability_score", 0.0)
    static_region_override = (
        base_safe
        and isinstance(person_risk, (int, float)) and person_risk >= 0.20
        and bool(evidence.get("detected_person_boxes"))
        and interval.get("static_region_found") is True
        and interval.get("fixed_bottom_fallback") is False
        and stable_score >= 0.85
        and 0.35 <= person_overlap
        and person_overlap < 1.0
    )
    allowed = baseline_allowed or override or static_region_override
    if allowed:
        keep_reason = None
        override_reason = (
            "static_region_person_overlap_override" if static_region_override else
            "hog_person_overlap_only_false_reject" if override else None
        )
    else:
        override_reason = None
        if not source_area:
            keep_reason = "missing_source_subtitle_bbox"
        elif not has_target_text:
            keep_reason = "missing_target_text"
        elif not non_person_safe:
            keep_reason = "non_person_overlap_risk"
        elif not isinstance(mask, (int, float)) or mask > 0.20:
            keep_reason = "mask_area_ratio_too_large"
        elif not 0.001 <= bbox_area_ratio <= 0.20:
            keep_reason = "bbox_area_ratio_out_of_range"
        elif person_overlap >= 1.0:
            keep_reason = "true_person_overlap_risk"
        elif person_overlap >= 0.35:
            keep_reason = "source_person_overlap_too_large"
        else:
            keep_reason = "insufficient_person_overlap_evidence"
    return {
        "allowed": allowed,
        "local_safety_override": override or static_region_override,
        "static_region_person_overlap_override": static_region_override,
        "override_reason": override_reason,
        "keep_reject_reason": keep_reason,
        "person_overlap_score": person_overlap,
        "scene_text_overlap_score": scene,
        "watermark_overlap_score": watermark,
        "face_overlap_score": face,
        "bbox_area_ratio": round(bbox_area_ratio, 6),
    }

def _moving_mask_segments(
    caption: Segment, samples: list[dict], width: int, height: int
) -> list[dict]:
    valid = sorted(
        (sample for sample in samples if sample.get("selected_source_bbox")),
        key=lambda sample: sample["actual_sampled_timestamp"],
    )
    if not valid:
        return []
    timestamps = [float(sample["actual_sampled_timestamp"]) for sample in valid]
    boundaries = [caption.start]
    boundaries.extend(
        (left + right) / 2 for left, right in zip(timestamps, timestamps[1:])
    )
    boundaries.append(caption.end)
    segments = []
    for index, sample in enumerate(valid):
        box = _bbox(sample["selected_source_bbox"])
        padding = max(18, round(box.height * 0.15))
        mask = {
            "left": max(0.0, box.left - padding),
            "top": max(0.0, box.top - padding),
            "right": min(float(width), box.right + padding),
            "bottom": min(float(height), box.bottom + padding * 1.5),
        }
        segments.append({
            "start": boundaries[index],
            "end": boundaries[index + 1],
            "source_bbox": box.to_dict(),
            "mask_bbox": mask,
            "interpolation_used": sample.get("interpolation_used", False),
        })
    return segments


def _fit_mask_to_text(mask: dict, text: str, width: int, height: int, font_size: int = 46) -> tuple[dict, dict, bool]:
    lines = text.splitlines() or [text]
    target_width = min(width * 0.90, max(len(line) for line in lines) * font_size * 0.55 + 32)
    target_height = len(lines) * font_size * 1.15 + 24
    center_x = (mask["left"] + mask["right"]) / 2
    center_y = (mask["top"] + mask["bottom"]) / 2
    left = max(0.0, center_x - target_width / 2)
    right = min(float(width), center_x + target_width / 2)
    top = max(0.0, center_y - target_height / 2)
    bottom = min(float(height), center_y + target_height / 2)
    fitted = {
        "left": min(mask["left"], left),
        "top": min(mask["top"], top),
        "right": max(mask["right"], right),
        "bottom": max(mask["bottom"], bottom),
    }
    target = {"left": left, "top": top, "right": right, "bottom": bottom}
    clipped = left < 0 or top < 0 or right > width or bottom > height
    return fitted, target, clipped


def _review_frame(video: Path, timestamp: int, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-ss", str(timestamp), "-i", str(video), "-frames:v", "1", "-q:v", "2", "-y", str(path)],
        check=True, capture_output=True, text=True,
    )

def _coverage_strategy(
    interval: dict | None,
    manual: dict | None,
    coverage_mode: str,
    allow_fallback: bool,
    *,
    has_target_text: bool = True,
    video_decision: str | None = None,
    frame_width: int = 10_000,
    frame_height: int = 10_000,
) -> str:
    if interval is None:
        return (
            "fallback_caption_placement"
            if coverage_mode in {
                "full_coverage_fallback",
                "accessibility_fallback_subtitle_mode",
            } and allow_fallback
            else "skip"
        )
    original = interval.get("decision")
    user_decision = (manual or {}).get("user_decision")
    if coverage_mode == "cover_source_subtitle":
        safe = _cover_safety_evidence(
            interval,
            has_target_text=has_target_text,
            width=frame_width,
            height=frame_height,
        )
        return "cover_source_subtitle" if safe["allowed"] else "skip"
    if coverage_mode in {
        "safe_replace", "source_replacement_mode", "adaptive_source_replace",
    }:
        if coverage_mode == "adaptive_source_replace":
            decision = SubtitleRenderPolicy().adaptive_decide(
                interval,
                has_target_text=has_target_text,
                video_decision=video_decision,
            )
            return (
                "floating_caption_source_replace"
                if decision.decision == "AUTO_SOURCE_REPLACE"
                else "skip"
            )
        confidence_threshold = (
            0.70 if coverage_mode == "adaptive_source_replace" else 0.80
        )
        confident = (
            original == "AUTO_REPLACE"
            and interval.get("layout_mode") in {
                "stable_track_subtitle",
                "karaoke_subtitle",
                "floating_caption_subtitle",
            }
            and interval.get("final_interval_confidence", 0.0)
            >= confidence_threshold
            and interval.get("scene_text_risk_score", 1.0) < 0.20
        )
        if coverage_mode == "adaptive_source_replace":
            confident = (
                confident
                and interval.get("watermark_risk_score", 0.0) < 0.20
                and interval.get("person_risk_score", 0.0) < 0.20
                and interval.get("mask_area_ratio", 1.0) <= 0.20
                and interval.get("selected_candidate") is not None
            )
        return "floating_caption_source_replace" if confident else "skip"
    if user_decision == "MANUAL_REGION" and (manual or {}).get("manual_source_bbox"):
        return "manual_region_replace"
    if user_decision == "AUTO_REPLACE":
        return "source_replace_review_approved"
    if original == "AUTO_REPLACE":
        return "source_replace_verified"
    if coverage_mode in {"full_coverage_fallback", "accessibility_fallback_subtitle_mode"} and allow_fallback:
        return "fallback_caption_placement"
    return "skip"

def _valid_bbox(values: dict | None, width: int, height: int) -> bool:
    if not values:
        return False
    return 0 <= values.get("left", -1) < values.get("right", -1) <= width and 0 <= values.get("top", -1) < values.get("bottom", -1) <= height

def _target_text(caption: Segment, interval: dict | None, source: str) -> tuple[str, str | None, str, bool]:
    normalized = caption.translated_text or caption.text
    ocr = ((interval or {}).get("selected_candidate") or {}).get("raw_ocr_text")
    candidates = {
        "normalized": normalized if normalized and normalized != "[REVIEW]" else None,
        "asr": (interval or {}).get("asr_segment_text"),
        "ocr": ocr,
        "source_ocr": ocr,
    }
    ordered = [source] + [item for item in ("normalized", "asr", "ocr", "source_ocr") if item != source]
    for item in ordered:
        value = candidates[item]
        if value and value != "[REVIEW]":
            mode = "translated_target" if item == "normalized" else (
                "source_caption_ocr_echo_for_validation" if item in {"ocr", "source_ocr"}
                else "source_text_echo_for_validation"
            )
            return value, item, mode, False
    return "", None, "missing_target_text", True

def _cover_target_text(caption: Segment, sequence: int) -> tuple[str, str, str]:
    translated = caption.translated_text.strip() if caption.translated_text else ""
    if translated and translated != "[REVIEW]":
        return translated, "translated_text", "translated_target"
    return f"DỊCH_{sequence + 1}", "placeholder", "placeholder_target"

def _caption_with_text(caption: Segment, text: str) -> Segment:
    return Segment(caption.id, caption.start, caption.end, caption.text, text, caption.source, caption.confidence)

def _bubble_bbox(text_bbox: dict, width: int, height: int) -> dict:
    horizontal = max(18.0, (text_bbox["right"] - text_bbox["left"]) * 0.16)
    vertical = max(18.0, (text_bbox["bottom"] - text_bbox["top"]) * 0.35)
    return {
        "left": max(0.0, text_bbox["left"] - horizontal),
        "top": max(0.0, text_bbox["top"] - vertical),
        "right": min(float(width), text_bbox["right"] + horizontal),
        "bottom": min(float(height), text_bbox["bottom"] + vertical),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--captions", type=Path, required=True)
    parser.add_argument("--region-report", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--interval-report", type=Path)
    parser.add_argument("--coverage-mode", choices=("safe_replace", "adaptive_source_replace", "cover_source_subtitle", "full_coverage_review", "full_coverage_fallback", "source_replacement_mode", "accessibility_fallback_subtitle_mode"), default="safe_replace")
    parser.add_argument("--allow-fallback-caption-placement", action="store_true")
    parser.add_argument("--manual-review-file", type=Path)
    parser.add_argument("--target-text-source", choices=("normalized", "asr", "ocr", "source_ocr"), default="normalized")
    parser.add_argument("--review-timestamps", default="5,10,15", type=_timestamps)
    parser.add_argument("--resource-profile", choices=("low", "balanced", "quality"), default="balanced")
    parser.add_argument("--ffmpeg-encoder", choices=("auto", "cpu", "nvenc"), default="auto")
    parser.add_argument("--ffmpeg-gpu-decode", action="store_true")
    parser.add_argument("--ffmpeg-nvenc-cq", type=int, default=23)
    parser.add_argument("--ffmpeg-nvenc-preset", default="p5")
    args = parser.parse_args()
    policy = ResourcePolicy(args.resource_profile)
    ffmpeg_config = FFmpegConfig(
        encoder=args.ffmpeg_encoder, use_gpu_decode=args.ffmpeg_gpu_decode,
        decoder="cuda" if args.ffmpeg_gpu_decode else "auto",
        nvenc_cq=args.ffmpeg_nvenc_cq, nvenc_preset=args.ffmpeg_nvenc_preset,
    )
    report = json.loads(args.region_report.read_text(encoding="utf-8"))
    frames = report["frames"]
    if not frames:
        raise ValidationError("Adaptive report contains no source frames.")
    width, height = frames[0]["frame_width"], frames[0]["frame_height"]
    if any(
        box.get("final_bbox_coordinate_space") != "original_frame_pixels"
        for frame in frames
        for box in frame.get("boxes", [])
    ):
        raise ValidationError("Adaptive report contains non-source-frame OCR coordinates.")
    captions = [
        Segment(
            id=item["id"], start=float(item["start"]), end=float(item["end"]),
            text=item["text"], translated_text=item.get("translated_text", ""),
            source=item["source"], confidence=float(item["confidence"]),
        )
        for item in json.loads(args.captions.read_text(encoding="utf-8"))
    ]
    samples = report["timestamp_reports"]
    interval_data = json.loads(args.interval_report.read_text(encoding="utf-8")).get("intervals", []) if args.interval_report else []
    video_decision = (
        json.loads(args.interval_report.read_text(encoding="utf-8")).get("video_decision")
        if args.interval_report else None
    )
    manual_data = json.loads(args.manual_review_file.read_text(encoding="utf-8")) if args.manual_review_file and args.manual_review_file.exists() else []
    manual_by_id = {item["interval_id"]: item for item in manual_data}
    render_captions, placements, caption_reports = [], [], []
    for caption_index, caption in enumerate(captions):
        interval = next((item for item in interval_data if item["timestamp_start"] <= caption.start < item["timestamp_end"]), None)
        manual = manual_by_id.get(interval["interval_id"]) if interval else None
        if args.coverage_mode == "cover_source_subtitle":
            text, text_source, text_mode = _cover_target_text(caption, caption_index)
            missing_text = False
        else:
            text, text_source, text_mode, missing_text = _target_text(caption, interval, args.target_text_source)
            if (
                missing_text
                and args.coverage_mode in {
                    "full_coverage_fallback",
                    "accessibility_fallback_subtitle_mode",
                }
            ):
                text = f"DỊCH_{caption_index + 1}"
                text_source, text_mode, missing_text = (
                    "placeholder", "placeholder_target", False
                )
        coverage_strategy = _coverage_strategy(
            interval, manual, args.coverage_mode,
            args.allow_fallback_caption_placement,
            has_target_text=not missing_text,
            video_decision=video_decision,
            frame_width=width,
            frame_height=height,
        )
        cover_safety = _cover_safety_evidence(
            interval, has_target_text=not missing_text, width=width, height=height
        ) if args.coverage_mode == "cover_source_subtitle" else {}
        if manual and manual.get("user_decision") == "MANUAL_REGION" and not _valid_bbox(manual.get("manual_source_bbox"), width, height):
            raise ValidationError("Manual source bbox is missing or outside the source frame.")
        if coverage_strategy == "skip":
            caption_reports.append({
                "caption_id": caption.id, "timestamp_range": {"start": caption.start, "end": caption.end},
                "time_range": {"start": caption.start, "end": caption.end},
                "interval_id": (interval or {}).get("interval_id"),
                "original_decision": (interval or {}).get("decision"), "render_strategy": "skip",
                "render_applied": False, "rendered": False,
                "source_bbox": (interval or {}).get("source_bbox"),
                "source_subtitle_bbox": (interval or {}).get("source_bbox"),
                "source_bbox_origin": (
                    "localization" if (interval or {}).get("source_bbox") else "none"
                ),
                "cover_box_bbox": None, "target_bbox": None,
                "bbox_iou_source_cover": None,
                "fallback_used": False, "manual_approved": bool(manual and manual.get("user_decision")),
                "production_style_source_replacement": False, "reason": "unresolved_interval",
                "target_text": text, "target_text_source": text_source, "target_text_mode": text_mode,
                "missing_target_text": missing_text, "review_placeholder_used": False,
                "cover_style": "skipped",
                "skip_reason": "local_safety_rejected" if args.coverage_mode == "cover_source_subtitle" else "unresolved_interval",
                "is_fixed_bottom_fallback": False,
                "original_decision": "skipped",
                "original_skip_reason": "local_safety_rejected",
                "local_safety_override": False,
                "static_region_person_overlap_override": cover_safety.get(
                    "static_region_person_overlap_override", False
                ),
                "override_reason": None,
                "keep_reject_reason": cover_safety.get("keep_reject_reason"),
                "final_decision": "skipped",
                **{
                    key: cover_safety.get(key)
                    for key in (
                        "person_overlap_score", "scene_text_overlap_score",
                        "watermark_overlap_score", "face_overlap_score",
                        "bbox_area_ratio",
                    )
                },
                "overlap_risk": {
                    "person": (interval or {}).get("person_risk_score"),
                    "scene_text": (interval or {}).get("scene_text_risk_score"),
                    "watermark": (interval or {}).get("watermark_risk_score"),
                },
            })
            continue
        if coverage_strategy == "fallback_caption_placement":
            if missing_text:
                caption_reports.append({"caption_id": caption.id, "render_strategy": "skip", "render_applied": False, "reason": "missing_target_text", "missing_target_text": True, "review_placeholder_used": False})
                continue
            render_captions.append(_caption_with_text(caption, text))
            placements.append(SubtitlePlacement(alignment=2, margin_v=80, mask_bbox=None, is_fallback=True))
            caption_reports.append({
                "caption_id": caption.id, "timestamp_range": {"start": caption.start, "end": caption.end},
                "original_decision": (interval or {}).get("decision"), "render_strategy": coverage_strategy,
                "render_applied": True, "source_bbox": None, "target_bbox": None, "mask_bbox": None,
                "fallback_used": True, "manual_approved": False, "production_style_source_replacement": False,
                "reason": "no_validated_source_bbox",
                "target_text": text, "target_text_source": text_source, "target_text_mode": text_mode,
                "missing_target_text": False, "review_placeholder_used": False,
                "bottom_fallback_used": True,
                "source_caption_detected": bool((interval or {}).get("source_bbox")),
                "duplicate_caption_risk": bool((interval or {}).get("source_bbox")),
            })
            continue
        if coverage_strategy == "floating_caption_source_replace" and not (interval or {}).get("source_bbox"):
            caption_reports.append({
                "caption_id": caption.id, "render_strategy": "skip", "render_applied": False,
                "source_caption_detected": False, "source_replacement_applied": False,
                "bottom_fallback_used": False, "skipped_reason": "missing_reliable_bubble_bbox",
                "duplicate_caption_risk": False, "target_text": text, "target_text_source": text_source,
            })
            continue
        matched = [
            sample for sample in samples.values()
            if caption.start <= float(sample["actual_sampled_timestamp"]) <= caption.end
        ]
        cover_source_bbox = (interval or {}).get("source_bbox")
        if matched and all(sample.get("strategy") == "no_subtitle_skip" for sample in matched):
            caption_reports.append({
                "caption_id": caption.id,
                "timestamp_range": {"start": caption.start, "end": caption.end},
                "strategy": "no_subtitle_skip",
                "source_bbox": None,
                "mask_bbox": None,
                "target_text_bbox": None,
                "target_text_clipped": False,
                "mask_area_ratio": 0.0,
                "coordinate_mapping": "PASS",
                "fallback_reason": "no_subtitle_present",
                "production_style_replacement": "CONDITIONAL",
                "source_leakage_detected": "not_evaluated",
                "track_mode": "unreliable_track",
            })
            continue
        if coverage_strategy == "manual_region_replace":
            manual_bbox = manual["manual_source_bbox"]
            for sample in matched:
                sample["selected_source_bbox"] = manual_bbox
                sample["strategy"] = "expanded_line_bbox"
        elif coverage_strategy in {
            "floating_caption_source_replace", "cover_source_subtitle",
        }:
            source_bbox = (interval or {}).get("source_bbox")
            if source_bbox:
                bubble = (
                    source_bbox
                    if coverage_strategy == "cover_source_subtitle"
                    else _bubble_bbox(source_bbox, width, height)
                )
                for sample in matched:
                    sample["selected_source_bbox"] = bubble
                    sample["strategy"] = "expanded_line_bbox"
        track_mode = matched[0].get("track_mode", "unreliable_track") if matched else "unreliable_track"
        moving_segments = (
            _moving_mask_segments(caption, matched, width, height)
            if track_mode in {"slowly_moving_track", "moving_track"} else []
        )
        if coverage_strategy == "cover_source_subtitle":
            mask = _padded_cover_bbox(
                cover_source_bbox, width, height,
                stable_region_bbox=cover_source_bbox,
            )
            strategy, fallback_reason = "source_bbox_cover", None
        else:
            mask, strategy, fallback_reason = _mask_for_caption(matched, width, height)
        segment_reports = []
        if moving_segments and coverage_strategy != "cover_source_subtitle":
            strategy = "segmented_local_mask"
            for index, item in enumerate(moving_segments):
                segment_mask, target_bbox, segment_clipped = _fit_mask_to_text(
                    item["mask_bbox"], text, width, height
                )
                item["mask_bbox"] = segment_mask
                item["target_text_bbox"] = target_bbox
                item["target_text_clipped"] = segment_clipped
                item["mask_area_ratio"] = round(
                    (segment_mask["right"] - segment_mask["left"])
                    * (segment_mask["bottom"] - segment_mask["top"])
                    / (width * height), 6,
                )
                render_captions.append(Segment(
                    id=f"{caption.id}_moving_{index}", start=item["start"], end=item["end"],
                    text=caption.text, translated_text=text,
                    source=caption.source, confidence=caption.confidence,
                ))
                placements.append(_placement_for_caption(segment_mask, strategy))
                segment_reports.append(item)
            mask = max(segment_reports, key=lambda item: item["mask_area_ratio"])["mask_bbox"]
            target_text_bbox = segment_reports[0]["target_text_bbox"]
            clipped = any(item["target_text_clipped"] for item in segment_reports)
            mask_area_ratio = max(item["mask_area_ratio"] for item in segment_reports)
        else:
            if coverage_strategy == "cover_source_subtitle":
                target_text_bbox, clipped = mask, False
            else:
                mask, target_text_bbox, clipped = _fit_mask_to_text(mask, text, width, height)
            render_captions.append(_caption_with_text(caption, text))
            placements.append(_placement_for_caption(mask, strategy))
            mask_area_ratio = (
                (mask["right"] - mask["left"]) * (mask["bottom"] - mask["top"])
                / (width * height)
            )
        if strategy == "expanded_line_bbox" and (
            target_text_bbox["right"] - target_text_bbox["left"]
            > mask["right"] - mask["left"]
        ):
            strategy = "local_band"
        caption_reports.append({
            "caption_id": caption.id,
            "interval_id": (interval or {}).get("interval_id"),
            "timestamp_range": {"start": caption.start, "end": caption.end},
            "time_range": {"start": caption.start, "end": caption.end},
            "strategy": strategy,
            "source_bbox": BoundingBox.union(
                [_bbox(item["selected_source_bbox"]) for item in matched if item.get("selected_source_bbox")]
            ).to_dict() if any(item.get("selected_source_bbox") for item in matched) else None,
            "source_subtitle_bbox": (interval or {}).get("source_bbox"),
            "source_bbox_origin": "localization",
            "mask_bbox": mask,
            "target_text_bbox": target_text_bbox,
            "target_text_clipped": clipped,
            "mask_area_ratio": round(mask_area_ratio, 6),
            "coordinate_mapping": "PASS",
            "fallback_reason": fallback_reason,
            "production_style_replacement": (
                "PASS" if moving_segments and mask_area_ratio < 0.20
                else "CONDITIONAL"
            ),
            "source_leakage_detected": "not_evaluated",
            "track_mode": track_mode,
            "y_center_trajectory": matched[0].get("y_center_trajectory", []) if matched else [],
            "track_velocity": matched[0].get("track_velocity", 0.0) if matched else 0.0,
            "moving_track_score": matched[0].get("moving_track_score", 0.0) if matched else 0.0,
            "rejected_as_jump": any(item.get("rejected_as_jump", False) for item in matched),
            "accepted_as_moving_track": any(item.get("accepted_as_moving_track", False) for item in matched),
            "interpolation_used": any(item.get("interpolation_used", False) for item in matched),
            "segment_count": len(segment_reports) if segment_reports else 1,
            "per_segment_mask_bbox": [item["mask_bbox"] for item in segment_reports] or [mask],
            "per_segment_mask_area_ratio": [item["mask_area_ratio"] for item in segment_reports] or [round(mask_area_ratio, 6)],
            "max_segment_mask_area_ratio": round(mask_area_ratio, 6),
            "avg_segment_mask_area_ratio": round(
                sum(item["mask_area_ratio"] for item in segment_reports) / len(segment_reports)
                if segment_reports else mask_area_ratio, 6
            ),
            "total_covered_area_ratio": round(
                sum(item["mask_area_ratio"] for item in segment_reports)
                if segment_reports else mask_area_ratio, 6
            ),
            "leakage_count": 0,
            "clipping_count": sum(item["target_text_clipped"] for item in segment_reports) if segment_reports else int(clipped),
            "original_decision": (interval or {}).get("decision"),
            "render_strategy": coverage_strategy,
            "render_applied": True,
            "rendered": True,
            "target_bbox": target_text_bbox,
            "fallback_used": False,
            "manual_approved": bool(manual and manual.get("user_decision")),
            "production_style_source_replacement": True,
            "bottom_fallback_used": False,
            "source_caption_detected": bool((interval or {}).get("source_bbox")),
            "bubble_bbox": mask if coverage_strategy == "floating_caption_source_replace" else None,
            "text_bbox": (interval or {}).get("source_bbox"),
            "source_replacement_applied": coverage_strategy != "fallback_caption_placement",
            "duplicate_caption_risk": False,
            "target_text": text,
            "target_text_source": text_source,
            "target_text_mode": text_mode,
            "missing_target_text": missing_text,
            "review_placeholder_used": False,
            "cover_box_bbox": mask if coverage_strategy == "cover_source_subtitle" else None,
            "original_stable_region_bbox": (
                cover_source_bbox if coverage_strategy == "cover_source_subtitle" else None
            ),
            "expanded_cover_region_bbox": (
                mask if coverage_strategy == "cover_source_subtitle" else None
            ),
            "expansion_px_x": (
                round(max(
                    cover_source_bbox["left"] - mask["left"],
                    mask["right"] - cover_source_bbox["right"],
                ), 3) if coverage_strategy == "cover_source_subtitle" else None
            ),
            "expansion_px_y": (
                round(max(
                    cover_source_bbox["top"] - mask["top"],
                    mask["bottom"] - cover_source_bbox["bottom"],
                ), 3) if coverage_strategy == "cover_source_subtitle" else None
            ),
            "expansion_reason": (
                "bubble_margin"
                if coverage_strategy == "cover_source_subtitle"
                and (
                    cover_source_bbox["right"] - cover_source_bbox["left"]
                ) / max(
                    1.0, cover_source_bbox["bottom"] - cover_source_bbox["top"]
                ) > 4.0
                else "glyph_margin"
                if coverage_strategy == "cover_source_subtitle"
                else None
            ),
            "cover_overflow_risk": (
                "low"
                if coverage_strategy == "cover_source_subtitle"
                and mask_area_ratio <= 0.12
                else "review"
                if coverage_strategy == "cover_source_subtitle"
                else None
            ),
            "bbox_iou_source_cover": (
                _bbox_iou(cover_source_bbox, mask)
                if coverage_strategy == "cover_source_subtitle" else None
            ),
            "cover_style": "black_box" if coverage_strategy == "cover_source_subtitle" else None,
            "skip_reason": None,
            "is_fixed_bottom_fallback": False,
            "original_decision": (
                "skipped" if cover_safety.get("local_safety_override") else
                (interval or {}).get("decision")
            ),
            "original_skip_reason": (
                "local_safety_rejected"
                if cover_safety.get("local_safety_override") else None
            ),
            "local_safety_override": cover_safety.get("local_safety_override", False),
            "static_region_person_overlap_override": cover_safety.get(
                "static_region_person_overlap_override", False
            ),
            "override_reason": cover_safety.get("override_reason"),
            "keep_reject_reason": None,
            "final_decision": "rendered",
            **{
                key: cover_safety.get(key)
                for key in (
                    "person_overlap_score", "scene_text_overlap_score",
                    "watermark_overlap_score", "face_overlap_score",
                    "bbox_area_ratio",
                )
            },
            "overlap_risk": {
                "person": (interval or {}).get("person_risk_score"),
                "scene_text": (interval or {}).get("scene_text_risk_score"),
                "watermark": (interval or {}).get("watermark_risk_score"),
            },
        })
    style = SubtitleStyle(font_size=46, outline_width=3.0, shadow=0.0, alignment=5)
    ass = ASSGenerator.generate(render_captions, style, placements, width, height)
    subtitle_path = args.output_root / "subtitles" / "subtitle.ass"
    subtitle_path.parent.mkdir(parents=True, exist_ok=True)
    subtitle_path.write_text(ass, encoding="utf-8")
    output_video = args.output_root / "videos" / "output.mp4"
    output_video.parent.mkdir(parents=True, exist_ok=True)
    FFmpegVideoProvider(ffmpeg_config=ffmpeg_config).render_video(
        str(args.video), str(subtitle_path), str(output_video)
    )
    review_dir = args.output_root / "review_frames"
    images = []
    for timestamp in args.review_timestamps:
        path = review_dir / f"frame_{timestamp:g}s.jpg"
        _review_frame(output_video, timestamp, path)
        images.append(Image.open(path).convert("RGB"))
    columns = min(4, len(images))
    rows = (len(images) + columns - 1) // columns
    sheet = Image.new("RGB", (images[0].width * columns, images[0].height * rows))
    for index, image in enumerate(images):
        sheet.paste(image, ((index % columns) * image.width, (index // columns) * image.height))
    sheet.save(review_dir / "contact_sheet_all_caption_samples.jpg")
    if set((5.0, 10.0, 15.0)).issubset(args.review_timestamps):
        Image.open(review_dir / "contact_sheet_all_caption_samples.jpg").save(
            review_dir / "contact_sheet_5s_10s_15s.jpg"
        )
    problem = [path for path in review_dir.glob("frame_*.jpg") if path.stem in {"frame_6s", "frame_17s", "frame_18s", "frame_19s"}]
    if problem:
        images = [Image.open(path).convert("RGB") for path in sorted(problem)]
        sheet = Image.new("RGB", (images[0].width, images[0].height * len(images)))
        for index, image in enumerate(images):
            sheet.paste(image, (0, index * image.height))
        sheet.save(review_dir / "contact_sheet_problem_cases_6s_17s_18s_19s.jpg")
    render_report = {
        "video_path": str(args.video),
        "output_video_path": str(output_video),
        "production_ready": "NO",
        "Production Ready": "NO",
        "play_res_x": width,
        "play_res_y": height,
        "fallback_count": sum(item.get("strategy") == "full_width_fallback" for item in caption_reports),
        "coverage_mode": args.coverage_mode,
        "label": "FULL_COVERAGE_WITH_FALLBACK" if args.coverage_mode == "full_coverage_fallback" else "SAFE_REPLACE",
        "captions": caption_reports,
    }
    debug = args.output_root / "debug"
    debug.mkdir(parents=True, exist_ok=True)
    (debug / "adaptive_render_report.json").write_text(
        json.dumps(render_report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if args.coverage_mode != "safe_replace":
        (debug / "full_coverage_render_report.json").write_text(
            json.dumps(render_report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    if args.coverage_mode == "source_replacement_mode":
        (debug / "floating_replace_report.json").write_text(
            json.dumps(render_report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    if args.coverage_mode == "cover_source_subtitle":
        (debug / "cover_source_subtitle_report.json").write_text(
            json.dumps(render_report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    capabilities = detect_ffmpeg_capabilities()
    encode_plan = build_ffmpeg_encode_args(
        str(args.video), str(output_video), ffmpeg_config, True, capabilities
    )
    write_resource_usage_report(
        args.output_root, policy, capabilities, encode_plan,
        expensive_steps_enabled=(
            ["extra_validation_passes"] if policy.expensive_validation_enabled else []
        ),
        expensive_steps_skipped=(
            ["floating_plain_text_uncertain", "full_frame_segmentation"]
            + ([] if policy.expensive_validation_enabled else ["extra_validation_passes"])
        ),
    )


if __name__ == "__main__":
    main()
