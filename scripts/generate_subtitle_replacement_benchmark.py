"""Generate deterministic synthetic subtitle-replacement benchmark videos."""

import json
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


CASES = [
    ("bottom_stable", ["bottom_stable_subtitle", "white_subtitle"], "bottom"),
    ("center", ["center_subtitle", "yellow_subtitle"], "center"),
    ("top", ["top_subtitle"], "top"),
    ("moving", ["moving_subtitle"], "moving"),
    ("two_line", ["two_line_subtitle", "subtitle_with_shadow"], "two_line"),
    ("punctuation", ["punctuation_noise"], "punctuation"),
    ("sticker", ["sticker_noise"], "sticker"),
    ("ui_overlay", ["ui_overlay"], "ui"),
    ("watermark_scene", ["watermark", "scene_text", "complex_background"], "watermark"),
    ("landscape_bright", ["landscape_video", "bright_background"], "landscape"),
    ("portrait", ["portrait_video"], "bottom"),
    ("rotated_metadata", ["rotated_metadata_video"], "bottom"),
]


def _font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype("C:/Windows/Fonts/arial.ttf", size)


def _text_bbox(draw: ImageDraw.ImageDraw, text: str, position: tuple[int, int], font: ImageFont.FreeTypeFont, stroke: int = 4) -> dict:
    left, top, right, bottom = draw.textbbox(position, text, font=font, anchor="mm", stroke_width=stroke)
    return {"left": left, "top": top, "right": right, "bottom": bottom}


def _create_case(root: Path, identifier: str, categories: list[str], mode: str) -> dict:
    width, height = ((960, 540) if mode == "landscape" else (540, 960))
    if identifier == "portrait":
        width, height = 1440, 2560
    case_dir = root / identifier
    frames_dir = case_dir / "frames_source"
    frames_dir.mkdir(parents=True, exist_ok=True)
    checkpoints, expected, forbidden = [], {}, {}
    font = _font(max(34, round(width * 0.07)))
    small = _font(max(20, round(width * 0.035)))
    for index in range(6):
        timestamp = index * 0.5
        image = Image.new("RGB", (width, height), (45 + index * 9, 80, 110))
        draw = ImageDraw.Draw(image)
        for stripe in range(8):
            x = (stripe * width // 7 + index * 23) % width
            draw.rectangle((x, height * 0.15, min(width, x + width * 0.08), height * 0.72), fill=(70 + stripe * 10, 100, 120))
        draw.ellipse((width * 0.1, height * 0.18, width * 0.3, height * 0.38), fill=(220, 180, 70))
        y = height * 0.82
        if mode == "center":
            y = height * 0.52
        elif mode == "top":
            y = height * 0.18
        elif mode == "moving":
            y = height * (0.25 + index * 0.10)
        text = "SOURCE SUBTITLE"
        if mode == "two_line":
            text = "SOURCE SUBTITLE\nSECOND LINE"
        bbox = _text_bbox(draw, text, (width // 2, round(y)), font)
        draw.multiline_text((width // 2, round(y)), text, font=font, anchor="mm", align="center", fill="white", stroke_width=4, stroke_fill="black")
        frame_forbidden = []
        if mode in {"punctuation", "sticker"}:
            noise = "???  ...  !!!"
            noise_bbox = _text_bbox(draw, noise, (width // 2, height // 2), font)
            draw.text((width // 2, height // 2), noise, font=font, anchor="mm", fill="#ff69b4", stroke_width=2, stroke_fill="white")
            frame_forbidden.append({"type": "decorative_symbol", "bbox": noise_bbox})
        if mode == "ui":
            ui_bbox = _text_bbox(draw, "LIVE  SCORE  12:30", (width // 2, 40), small)
            draw.text((width // 2, 40), "LIVE  SCORE  12:30", font=small, anchor="mm", fill="white")
            frame_forbidden.append({"type": "ui_overlay", "bbox": ui_bbox})
        if mode == "watermark":
            wm_bbox = _text_bbox(draw, "@CHANNEL", (width - 90, 45), small)
            sign_bbox = _text_bbox(draw, "ROAD SIGN", (100, height // 2), small)
            draw.text((width - 90, 45), "@CHANNEL", font=small, anchor="mm", fill="#dddddd")
            draw.text((100, height // 2), "ROAD SIGN", font=small, anchor="mm", fill="yellow")
            frame_forbidden.extend(({"type": "watermark", "bbox": wm_bbox}, {"type": "scene_text", "bbox": sign_bbox}))
        path = frames_dir / f"frame_{index:03d}.png"
        image.save(path)
        key = f"{timestamp:g}"
        checkpoints.append(timestamp)
        expected[key] = bbox
        forbidden[key] = frame_forbidden
    video_path = case_dir / "video.mp4"
    subprocess.run(
        ["ffmpeg", "-framerate", "2", "-i", str(frames_dir / "frame_%03d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-y", str(video_path)],
        check=True, capture_output=True, text=True,
    )
    if identifier == "rotated_metadata":
        rotated = case_dir / "rotated.mp4"
        subprocess.run(["ffmpeg", "-i", str(video_path), "-c", "copy", "-metadata:s:v:0", "rotate=90", "-y", str(rotated)], check=True, capture_output=True, text=True)
        rotated.replace(video_path)
    caption = {
        "id": "caption_1", "start": 0.0, "end": 3.0, "text": "SOURCE SUBTITLE",
        "translated_text": "PHỤ ĐỀ ĐÃ DỊCH", "source": "ocr", "confidence": 1.0,
    }
    transcripts = case_dir / "transcripts"
    transcripts.mkdir(exist_ok=True)
    (transcripts / "normalized.json").write_text(json.dumps([caption], ensure_ascii=False, indent=2), encoding="utf-8")
    ground_truth = {
        "video_path": str(video_path.relative_to(root)),
        "category": categories,
        "checkpoints": checkpoints,
        "expected_source_subtitle_bboxes": {
            "caption_1": (
                {
                    "left": min(box["left"] for box in expected.values()),
                    "top": min(box["top"] for box in expected.values()),
                    "right": max(box["right"] for box in expected.values()),
                    "bottom": max(box["bottom"] for box in expected.values()),
                }
                if mode == "moving" else expected["1"]
            )
        },
        "expected_source_subtitle_bbox_per_timestamp": expected,
        "forbidden_regions_per_timestamp": forbidden,
        "forbidden_regions": [region for regions in forbidden.values() for region in regions],
        "expected_subtitle_track_position": mode,
        "allowed_strategy": ["expanded_line_bbox", "local_band"],
        "notes": "Synthetic mechanics benchmark; not evidence of real-world robustness.",
    }
    (case_dir / "ground_truth.json").write_text(json.dumps(ground_truth, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"id": identifier, "path": str(video_path.relative_to(root)), "categories": categories, "ground_truth": f"{identifier}/ground_truth.json"}


def main() -> None:
    root = Path("test_videos/subtitle_replacement_benchmark")
    root.mkdir(parents=True, exist_ok=True)
    samples = [_create_case(root, *case) for case in CASES]
    real_dir = root / "real_videos"
    real_dir.mkdir(exist_ok=True)
    (real_dir / "ground_truth.template.json").write_text(json.dumps({
        "video_path": "replace-with-video.mp4",
        "category": ["bottom_stable_subtitle"],
        "checkpoints": [1.0],
        "expected_source_subtitle_bbox_per_timestamp": {
            "1": {"left": 0, "top": 0, "right": 0, "bottom": 0}
        },
        "forbidden_regions_per_timestamp": {"1": []},
        "expected_subtitle_track_position": "bottom",
        "allowed_strategy": ["expanded_line_bbox", "local_band"],
        "notes": "Copy this file beside each real video and replace all placeholder values.",
    }, indent=2), encoding="utf-8")
    variant_dir = root / "sample_variants"
    variant_dir.mkdir(exist_ok=True)
    variant_filters = {
        "question_marks": "drawtext=fontfile=C\\:/Windows/Fonts/arial.ttf:text='???':x=(w-text_w)/2:y=h/2:fontsize=80:fontcolor=ff69b4",
        "top_ui": "drawtext=fontfile=C\\:/Windows/Fonts/arial.ttf:text='LIVE UI':x=40:y=40:fontsize=48:fontcolor=white",
        "watermark": "drawtext=fontfile=C\\:/Windows/Fonts/arial.ttf:text='@CHANNEL':x=w-text_w-30:y=30:fontsize=38:fontcolor=white",
        "bright_resize_crop": "eq=brightness=0.18,crop=iw*0.9:ih*0.9:(iw-iw*0.9)/2:(ih-ih*0.9)/2,scale=1440:2560",
    }
    sample = Path("sample.mp4")
    variant_samples = []
    if sample.exists():
        for name, filter_value in variant_filters.items():
            video = variant_dir / f"{name}.mp4"
            result = subprocess.run(
                ["ffmpeg", "-ss", "0", "-t", "6", "-i", str(sample), "-vf", filter_value, "-an", "-y", str(video)],
                check=False, capture_output=True, text=True,
            )
            if result.returncode:
                continue
            ground_truth = {
                "video_path": str(video.relative_to(root)),
                "category": ["sample_variant"],
                "checkpoints": [1.0],
                "expected_source_subtitle_bbox_per_timestamp": {},
                "forbidden_regions_per_timestamp": {"1": []},
                "expected_subtitle_track_position": "source-dependent",
                "allowed_strategy": ["expanded_line_bbox", "local_band"],
                "notes": "Derived from sample.mp4; not an independent real video.",
            }
            (variant_dir / f"{name}.ground_truth.json").write_text(json.dumps(ground_truth, indent=2), encoding="utf-8")
            variant_samples.append({
                "id": f"sample_{name}", "path": str(video.relative_to(root)),
                "categories": ["complex_background"],
                "ground_truth": f"sample_variants/{name}.ground_truth.json",
            })
    manifest = {
        "schema_version": 1,
        "dataset_tiers": {
            "synthetic": "Generated mechanics checks.",
            "sample_variants": "Derived from sample.mp4; not independent real videos.",
            "real": "User-supplied diverse videos; currently empty.",
        },
        "samples": samples + variant_samples,
    }
    (root / "benchmark_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
