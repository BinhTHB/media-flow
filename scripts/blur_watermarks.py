import argparse
import json
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from src.application.services.adaptive_subtitle_region_service import BoundingBox
from src.application.services.watermark_detection_service import (
    WatermarkDetectionService,
    WatermarkObservation,
)
from src.domain.entities.resource_policy import FFmpegConfig, ResourcePolicy
from src.infrastructure.runtime.ffmpeg_capability_detector import detect_ffmpeg_capabilities
from src.infrastructure.runtime.ffmpeg_encode import build_ffmpeg_encode_args
from src.infrastructure.runtime.resource_usage_report import write_resource_usage_report


def _sample_frames(video: Path, output_dir: Path, interval: float) -> list[tuple[float, Path]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-i", str(video), "-vf", f"fps=1/{interval}", "-q:v", "2", "-y", str(output_dir / "frame_%04d.jpg")],
        check=True, capture_output=True, text=True,
    )
    return [((index + 0.5) * interval, path) for index, path in enumerate(sorted(output_dir.glob("frame_*.jpg")))]


def _visual_observations(
    samples: list[tuple[float, Path]],
    edge_band_ratio: float,
    scan_full_frame: bool,
) -> tuple[list[WatermarkObservation], int, int]:
    observations: list[WatermarkObservation] = []
    edge_maps = []
    width = height = 0
    for timestamp, path in samples:
        image = cv2.imread(str(path))
        if image is None:
            continue
        height, width = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        median_value = float(np.median(gray))
        edges = cv2.Canny(gray, int(max(40, median_value * 0.65)), int(min(220, median_value * 1.35)))
        edges = cv2.morphologyEx(
            edges, cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (9, 5)),
        )
        edge_maps.append(edges > 0)
        count, _, stats, _ = cv2.connectedComponentsWithStats(edges, 8)
        for x, y, box_width, box_height, area in stats[1:count]:
            bbox_area = box_width * box_height
            edge_density = area / max(1, bbox_area)
            if (
                area < 30 or bbox_area < width * height * 0.00008
                or bbox_area > width * height * 0.025
                or edge_density < 0.025 or edge_density > 0.65
                or min(box_width, box_height) < 8
            ):
                continue
            bbox = BoundingBox(float(x), float(y), float(x + box_width), float(y + box_height))
            edge_distance = min(x, y, width - (x + box_width), height - (y + box_height))
            in_edge_band = (
                x <= width * edge_band_ratio
                or x + box_width >= width * (1 - edge_band_ratio)
                or y <= height * edge_band_ratio
                or y + box_height >= height * (1 - edge_band_ratio)
            )
            if not scan_full_frame and not in_edge_band:
                continue
            aspect = max(box_width, box_height) / max(1, min(box_width, box_height))
            observations.append(WatermarkObservation(
                timestamp=timestamp, bbox=bbox,
                overlay_style_score=min(1.0, 0.40 + (0.25 if aspect >= 1.5 else 0.0)),
                scene_text_risk_score=0.10 if edge_distance <= min(width, height) * 0.08 else 0.55,
            ))
    if edge_maps:
        persistent = (np.mean(edge_maps, axis=0) >= 0.45).astype("uint8") * 255
        persistent = cv2.morphologyEx(
            persistent, cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (15, 9)),
        )
        count, _, stats, _ = cv2.connectedComponentsWithStats(persistent, 8)
        for x, y, box_width, box_height, area in stats[1:count]:
            bbox_area = box_width * box_height
            if (
                area < 30 or bbox_area < width * height * 0.00008
                or bbox_area > width * height * 0.025
                or min(box_width, box_height) < 8
            ):
                continue
            in_edge_band = (
                x <= width * edge_band_ratio or x + box_width >= width * (1 - edge_band_ratio)
                or y <= height * edge_band_ratio or y + box_height >= height * (1 - edge_band_ratio)
            )
            if not scan_full_frame and not in_edge_band:
                continue
            edge_distance = min(x, y, width - x - box_width, height - y - box_height)
            scene_risk = 0.08 if edge_distance <= min(width, height) * 0.08 else 0.55
            bbox = BoundingBox(float(x), float(y), float(x + box_width), float(y + box_height))
            observations.extend(
                WatermarkObservation(
                    timestamp=timestamp, bbox=bbox,
                    overlay_style_score=0.95,
                    scene_text_risk_score=scene_risk,
                )
                for timestamp, _ in samples
            )
    return observations, width, height


def _padded_bbox(
    candidate: dict, width: int, height: int, padding_ratio: float = 0.0,
    bottom_padding_ratio: float = 0.0,
) -> dict:
    box = candidate["bbox"]
    padding = max(0, min(20, round(min(width, height) * padding_ratio)))
    bottom_padding = max(0, min(100, round(height * bottom_padding_ratio)))
    return WatermarkDetectionService.clamp_bbox(
        BoundingBox(box["left"] - padding, box["top"] - padding, box["right"] + padding, box["bottom"] + padding + bottom_padding),
        width, height,
    ).to_dict()


def _write_contact_sheet(samples: list[tuple[float, Path]], candidates: list[dict], output: Path) -> None:
    images = []
    for timestamp, path in samples[:12]:
        image = Image.open(path).convert("RGB")
        draw = ImageDraw.Draw(image)
        for candidate in candidates:
            if candidate["first_seen_timestamp"] <= timestamp <= candidate["last_seen_timestamp"]:
                box = candidate["bbox"]
                draw.rectangle((box["left"], box["top"], box["right"], box["bottom"]), outline="red", width=3)
                draw.text((box["left"], max(0, box["top"] - 14)), f'{candidate["candidate_id"]} {candidate["classification"]} {candidate["watermark_score"]:.2f}', fill="red")
        images.append(image)
    if not images:
        return
    thumb_width = min(480, images[0].width)
    thumbs = [image.resize((thumb_width, round(image.height * thumb_width / image.width))) for image in images]
    sheet = Image.new("RGB", (thumb_width * min(3, len(thumbs)), max(item.height for item in thumbs) * ((len(thumbs) + 2) // 3)))
    for index, image in enumerate(thumbs):
        sheet.paste(image, ((index % 3) * thumb_width, (index // 3) * image.height))
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)

def _write_before_after_sheet(
    samples: list[tuple[float, Path]], candidates: list[dict], output: Path,
    strength: int, passes: int, padding_ratio: float,
) -> None:
    rows = []
    for candidate in candidates[:8]:
        timestamp = (candidate["first_seen_timestamp"] + candidate["last_seen_timestamp"]) / 2
        _, path = min(samples, key=lambda item: abs(item[0] - timestamp))
        image = Image.open(path).convert("RGB")
        box = _padded_bbox(candidate, image.width, image.height, padding_ratio)
        margin = max(24, round(max(box["right"] - box["left"], box["bottom"] - box["top"]) * 0.8))
        zoom = (
            max(0, round(box["left"] - margin)), max(0, round(box["top"] - margin)),
            min(image.width, round(box["right"] + margin)), min(image.height, round(box["bottom"] + margin)),
        )
        before = image.crop(zoom)
        after = before.copy()
        local = (
            round(box["left"] - zoom[0]), round(box["top"] - zoom[1]),
            round(box["right"] - zoom[0]), round(box["bottom"] - zoom[1]),
        )
        blurred = after.crop(local)
        for _ in range(passes):
            blurred = blurred.filter(ImageFilter.GaussianBlur(strength))
        mask = Image.new("L", blurred.size)
        mask_draw = ImageDraw.Draw(mask)
        radius = max(4, min(blurred.size) // 8)
        inset = max(1, radius // 3)
        mask_draw.rounded_rectangle(
            (inset, inset, blurred.width - inset - 1, blurred.height - inset - 1),
            radius=radius, fill=255,
        )
        mask = mask.filter(ImageFilter.GaussianBlur(max(2, radius // 2)))
        after.paste(blurred, local[:2], mask)
        target_height = 260
        before.thumbnail((420, target_height))
        after.thumbnail((420, target_height))
        row = Image.new("RGB", (840, target_height + 24), "white")
        row.paste(before, (0, 24))
        row.paste(after, (420, 24))
        ImageDraw.Draw(row).text((4, 4), f'{candidate["candidate_id"]}: before / after', fill="black")
        rows.append(row)
    if not rows:
        output.parent.mkdir(parents=True, exist_ok=True)
        sheet = Image.new("RGB", (840, 120), "white")
        ImageDraw.Draw(sheet).text(
            (20, 48), "No candidates approved for automatic blur.", fill="black"
        )
        sheet.save(output)
        return
    sheet = Image.new("RGB", (840, sum(row.height for row in rows)), "white")
    y = 0
    for row in rows:
        sheet.paste(row, (0, y))
        y += row.height
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)


def _blur(
    video: Path, output: Path, candidates: list[dict], mode: str, strength: int,
    width: int, height: int, padding_ratio: float, bottom_padding_ratio: float,
    passes: int, feather_px: int, ffmpeg_config: FFmpegConfig,
) -> int:
    approved = [item for item in candidates if item["classification"] in {"watermark", "logo", "persistent_overlay"}]
    output.parent.mkdir(parents=True, exist_ok=True)
    if not approved:
        subprocess.run(
            ["ffmpeg", "-i", str(video), "-map", "0", "-c", "copy", "-y", str(output)],
            check=True, capture_output=True, text=True,
        )
        return 0
    filters = []
    current = "[0:v]"
    for index, candidate in enumerate(approved):
        box = _padded_bbox(candidate, width, height, padding_ratio, bottom_padding_ratio)
        x, y = round(box["left"]) // 2 * 2, round(box["top"]) // 2 * 2
        w, h = max(2, round(box["right"] - box["left"]) // 2 * 2), max(2, round(box["bottom"] - box["top"]) // 2 * 2)
        if mode in {"pixelate", "pixelate_feathered"}:
            blur = f"scale={max(1, w // max(2, strength))}:{max(1, h // max(2, strength))}:flags=neighbor,scale={w}:{h}:flags=neighbor"
        else:
            effective_strength = strength * 1.35 if mode == "gaussian_feathered_strong" else strength
            unit = f"{'boxblur' if mode == 'box' else 'gblur'}={strength if mode == 'box' else f'sigma={effective_strength}'}"
            blur = ",".join(unit for _ in range(passes))
        start, end = candidate["first_seen_timestamp"], candidate["last_seen_timestamp"]
        next_label = f"[v{index}]"
        if mode in {"gaussian_feathered", "gaussian_feathered_strong", "pixelate_feathered", "blur_fill_feathered"}:
            if mode == "blur_fill_feathered":
                margin = min(48, max(feather_px, strength * 2))
                source_x, source_y = max(0, x - margin), max(0, y - margin)
                source_right, source_bottom = min(width, x + w + margin), min(height, y + h + margin)
                source_x, source_y = source_x // 2 * 2, source_y // 2 * 2
                source_w, source_h = max(2, (source_right - source_x) // 2 * 2), max(2, (source_bottom - source_y) // 2 * 2)
                offset_x, offset_y = (x - source_x) // 2 * 2, (y - source_y) // 2 * 2
                blur_source = f"crop={source_w}:{source_h}:{source_x}:{source_y},{blur},crop={w}:{h}:{offset_x}:{offset_y}"
            else:
                blur_source = f"crop={w}:{h}:{x}:{y},{blur}"
            alpha = (
                "255*min(1\\,"
                f"min(min(X\\,{w - 1}-X)\\,min(Y\\,{h - 1}-Y))/{max(1, feather_px)})"
            )
            filters.extend([
                f"{current}split=2[base{index}][crop{index}]",
                f"[crop{index}]{blur_source},format=rgba[wm{index}]",
                f"color=c=white:s={w}x{h},format=gray,geq=lum='{alpha}'[alpha{index}]",
                f"[wm{index}][alpha{index}]alphamerge[soft{index}]",
                f"[base{index}][soft{index}]overlay={x}:{y}:format=auto:shortest=1:enable='between(t,{start},{end})'{next_label}",
            ])
        else:
            filters.extend([
                f"{current}crop={w}:{h}:{x}:{y},{blur}[wm{index}]",
                f"{current}[wm{index}]overlay={x}:{y}:enable='between(t,{start},{end})'{next_label}",
            ])
        current = next_label
    chain = ";".join(filters)
    try:
        plan = build_ffmpeg_encode_args(
            str(video), str(output), ffmpeg_config, True, detect_ffmpeg_capabilities()
        )
        command = list(plan.args)
        command[command.index("-i") + 2:command.index("-c:v")] = [
            "-filter_complex", chain, "-map", current, "-map", "0:a?",
        ]
        subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as error:
        raise RuntimeError(error.stderr[-2000:]) from error
    return len(approved)


def _residual_score(image: np.ndarray, box: dict) -> float:
    roi = image[round(box["top"]):round(box["bottom"]), round(box["left"]):round(box["right"])]
    if roi.size == 0:
        return 0.0
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    edges = cv2.magnitude(
        cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3),
        cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3),
    ) / 255.0
    return float(np.mean(edges * (gray >= 200)))


def _video_frame(video: Path, timestamp: float) -> np.ndarray:
    capture = cv2.VideoCapture(str(video))
    capture.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError(f"Could not read review frame from {video}")
    return frame


def _comparison_sheet(video: Path, variants: dict[str, Path], candidate: dict, output: Path) -> dict:
    timestamp = (candidate["first_seen_timestamp"] + candidate["last_seen_timestamp"]) / 2
    source = _video_frame(video, timestamp)
    box = candidate["bbox"]
    before_score = _residual_score(source, box)
    images = [("before", source)]
    metrics = {}
    for name, path in variants.items():
        frame = _video_frame(path, timestamp)
        after_score = _residual_score(frame, box)
        metrics[name] = {
            "residual_score_before": round(before_score, 6),
            "residual_score_after": round(after_score, 6),
            "residual_reduction_ratio": round(max(0.0, 1.0 - after_score / max(before_score, 1e-8)), 6),
        }
        images.append((name, frame))
    margin = 50
    left, top = max(0, round(box["left"]) - margin), max(0, round(box["top"]) - margin)
    right, bottom = min(source.shape[1], round(box["right"]) + margin), min(source.shape[0], round(box["bottom"]) + margin)
    panels = []
    for name, frame in images:
        panel = Image.fromarray(cv2.cvtColor(frame[top:bottom, left:right], cv2.COLOR_BGR2RGB))
        panel.thumbnail((320, 320))
        canvas = Image.new("RGB", (320, 350), "white")
        canvas.paste(panel, (0, 30))
        ImageDraw.Draw(canvas).text((4, 6), name, fill="black")
        panels.append(canvas)
    sheet = Image.new("RGB", (320 * len(panels), 350), "white")
    for index, panel in enumerate(panels):
        sheet.paste(panel, (index * 320, 0))
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--watermark-report", type=Path)
    parser.add_argument("--blur-mode", choices=("gaussian_feathered", "gaussian_feathered_strong", "pixelate_feathered", "blur_fill_feathered", "gaussian", "box", "pixelate"), default="blur_fill_feathered")
    parser.add_argument("--blur-strength", type=int, default=24)
    parser.add_argument("--blur-padding-ratio", type=float, default=0.005)
    parser.add_argument("--blur-padding-bottom-ratio", type=float, default=0.04)
    parser.add_argument("--blur-passes", type=int, default=2)
    parser.add_argument("--blur-feather-px", type=int, default=18)
    parser.add_argument("--review-only", action="store_true")
    parser.add_argument("--sample-rate", type=float, default=1.0)
    parser.add_argument("--min-watermark-score", type=float, default=0.67)
    parser.add_argument("--corner-prior", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--edge-band-ratio", type=float, default=0.18)
    parser.add_argument("--allow-center-watermark", action="store_true")
    parser.add_argument("--scan-full-frame-for-watermark", action="store_true")
    parser.add_argument("--resource-profile", choices=("low", "balanced", "quality"), default="balanced")
    parser.add_argument("--enable-inpaint-light", action="store_true")
    parser.add_argument("--ffmpeg-encoder", choices=("auto", "cpu", "nvenc"), default="auto")
    parser.add_argument("--ffmpeg-gpu-decode", action="store_true")
    parser.add_argument("--ffmpeg-nvenc-cq", type=int, default=23)
    parser.add_argument("--ffmpeg-nvenc-preset", default="p5")
    args = parser.parse_args()
    policy = ResourcePolicy(args.resource_profile, args.enable_inpaint_light)
    ffmpeg_config = FFmpegConfig(
        encoder=args.ffmpeg_encoder, use_gpu_decode=args.ffmpeg_gpu_decode,
        decoder="cuda" if args.ffmpeg_gpu_decode else "auto",
        nvenc_cq=args.ffmpeg_nvenc_cq, nvenc_preset=args.ffmpeg_nvenc_preset,
    )
    if args.blur_mode == "blur_fill_feathered" and policy.profile == "low":
        args.blur_mode = policy.watermark_mode
    samples = _sample_frames(
        args.video, args.output_root / "debug" / "watermark_samples",
        args.sample_rate * policy.sample_interval_multiplier,
    )
    observations, width, height = _visual_observations(
        samples, args.edge_band_ratio,
        args.scan_full_frame_for_watermark or not args.corner_prior,
    )
    report_path = args.watermark_report or args.output_root / "debug" / "watermark_detection_report.json"
    if args.watermark_report:
        detection = json.loads(args.watermark_report.read_text(encoding="utf-8"))
        candidates = detection["final_candidates"]
    else:
        service = WatermarkDetectionService(
            edge_band_ratio=args.edge_band_ratio,
            allow_center_watermark=args.allow_center_watermark,
        )
        grouped = service.detect(observations, len(samples), width, height)
        final_candidates = service.merge_nearby(
            [item for item in grouped if item["classification"] != "rejected"],
            width, height,
        )
        candidates = final_candidates
        rejected = [item for item in grouped if item["classification"] == "rejected"]
        uncertain = [item for item in candidates if item["classification"] == "uncertain"]
        reasons = {}
        for item in rejected + uncertain:
            for reason in item["rejected_reasons"]:
                reasons[reason] = reasons.get(reason, 0) + 1
        detection = {
            "video_path": str(args.video), "sampled_frames": len(samples),
            "frame_width": width, "frame_height": height,
            "raw_candidate_count": len(observations),
            "grouped_candidate_count": len(grouped),
            "final_candidate_count": len(candidates),
            "uncertain_candidate_count": len(uncertain),
            "rejected_candidate_count": len(rejected),
            "top_rejection_reasons": reasons,
            "corner_prior_used": args.corner_prior,
            "edge_band_ratio": args.edge_band_ratio,
            "scan_full_frame_for_watermark": args.scan_full_frame_for_watermark,
            "candidate_groups": grouped,
            "final_candidates": candidates,
            "rejected_candidates": rejected,
        }
    candidates = [
        WatermarkDetectionService.apply_auto_blur_safety(
            item, width, height, len(candidates), policy.profile
        )
        for item in candidates
    ]
    detection["final_candidates"] = candidates
    detection["uncertain_candidate_count"] = sum(
        item["classification"] == "uncertain" for item in candidates
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(detection, indent=2), encoding="utf-8")
    approved = [
        item for item in candidates
        if item["watermark_score"] >= args.min_watermark_score
        and item["classification"] in {"watermark", "logo", "persistent_overlay"}
        and item["frame_coverage_ratio"] >= 0.60
        and item["position_stability_score"] >= 0.85
        and item["corner_or_edge_score"] >= 0.70
        and item["subtitle_overlap_risk"] < 0.20
        and item["scene_attachment_risk"] < 0.35
        and item["balanced_auto_blur_allowed"]
    ]
    review_dir = args.output_root / "review_frames"
    _write_contact_sheet(samples, candidates, review_dir / "contact_sheet_final_watermark_candidates.jpg")
    _write_before_after_sheet(samples, approved, review_dir / "contact_sheet_blur_before_after.jpg", args.blur_strength, args.blur_passes, args.blur_padding_ratio)
    safety_rejected = [item for item in candidates if not item["balanced_auto_blur_allowed"]]
    _write_contact_sheet(samples, safety_rejected[:12], review_dir / "contact_sheet_watermark_rejected_safety.jpg")
    _write_contact_sheet(samples, (detection.get("rejected_candidates", []) + [item for item in candidates if item["classification"] == "uncertain"])[:12], review_dir / "contact_sheet_rejected_or_uncertain.jpg")
    output = args.output_root / "videos" / "output_watermark_blurred.mp4"
    variant_modes = (
        ("gaussian_feathered_strong", "pixelate_feathered", "blur_fill_feathered")
        if policy.expensive_validation_enabled else (args.blur_mode,)
    )
    variants = {mode: args.output_root / "variants" / f"{mode}.mp4" for mode in variant_modes}
    blurred = 0
    if not args.review_only:
        for mode, path in variants.items():
            blurred = _blur(args.video, path, approved, mode, args.blur_strength, width, height, args.blur_padding_ratio, args.blur_padding_bottom_ratio, args.blur_passes, args.blur_feather_px, ffmpeg_config)
        if args.blur_mode in variants:
            output.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(variants[args.blur_mode], output)
        else:
            blurred = _blur(args.video, output, approved, args.blur_mode, args.blur_strength, width, height, args.blur_padding_ratio, args.blur_padding_bottom_ratio, args.blur_passes, args.blur_feather_px, ffmpeg_config)
    residual_metrics = _comparison_sheet(args.video, variants, approved[0], review_dir / "contact_sheet_blur_mode_comparison.jpg") if approved and not args.review_only else {}
    blur_report = {"video_path": str(args.video), "output_video_path": str(output) if output.exists() else None, "blur_mode": args.blur_mode, "blur_strength": args.blur_strength, "blur_padding_ratio": args.blur_padding_ratio, "blur_padding_bottom_ratio": args.blur_padding_bottom_ratio, "blur_passes": args.blur_passes, "blur_feather_px": args.blur_feather_px, "blurred_candidate_count": blurred, "uncertain_candidate_count": detection.get("uncertain_candidate_count", 0), "rejected_candidate_count": detection.get("rejected_candidate_count", 0), "residual_metrics": residual_metrics, "blurred_candidates": [{**item, "padded_bbox": _padded_bbox(item, width, height, args.blur_padding_ratio, args.blur_padding_bottom_ratio), "blur_strength": args.blur_strength, "blur_passes": args.blur_passes, "blur_feather_px": args.blur_feather_px} for item in approved], "frame_count_affected": blurred * len(samples), "skipped_candidates": [item for item in candidates if item not in approved], "review_only": args.review_only}
    (args.output_root / "debug").mkdir(parents=True, exist_ok=True)
    (args.output_root / "debug" / "watermark_blur_report.json").write_text(json.dumps(blur_report, indent=2), encoding="utf-8")
    capabilities = detect_ffmpeg_capabilities()
    encode_plan = build_ffmpeg_encode_args(
        str(args.video), str(output), ffmpeg_config, True, capabilities
    )
    write_resource_usage_report(
        args.output_root, policy, capabilities, encode_plan,
        expensive_steps_enabled=["watermark_blur"] + (
            ["watermark_variants"] if policy.expensive_validation_enabled else []
        ),
        expensive_steps_skipped=(
            ([] if policy.expensive_validation_enabled else [
                "watermark_variants", "inpaint_light", "full_frame_segmentation",
            ])
            + (["watermark_auto_blur_safety_gate"] if safety_rejected else [])
        ),
    )


if __name__ == "__main__":
    main()
