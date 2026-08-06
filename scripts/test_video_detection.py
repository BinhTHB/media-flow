"""Test burn-in detection on a real video.

Decodes the video frame-by-frame, runs the YOLO burn-in detector, optionally
draws boxes/labels, writes an annotated output video (keeping audio, if present),
saves per-frame JSONL results, and prints final statistics.

Usage:
    python scripts/test_video_detection.py --input path/input.mp4 --output out.mp4
    # PowerShell equivalent:
    python scripts/test_video_detection.py --input .\input\clip.mp4 --output .\outputs\detected.mp4 --show-labels --save-json

Options:
    --conf-subtitle 0.15   per-class confidence override (also watermark/logo/...)
    --show-labels          draw class labels on boxes
    --save-json            write detections JSONL
    --device cpu|cuda|auto
    --imgsz 1280
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
import numpy as np

from src.infrastructure.detection import (
    DetectionConfig,
    YoloBurnInDetector,
    load_detection_config,
)

# BGR colors per class for drawing.
CLASS_COLORS = {
    "subtitle": (89, 178, 85),       # green
    "watermark": (77, 55, 216),      # red
    "logo": (255, 172, 27),          # blue
    "other_burn_in_text": (80, 165, 255),  # orange
}
FALLBACK_COLOR = (66, 204, 255)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Test burn-in detection on a video")
    p.add_argument("--input", required=True, help="Input video path")
    p.add_argument("--output", default="outputs/detected.mp4", help="Output video path")
    p.add_argument("--show-labels", action="store_true", help="Draw class+conf labels")
    p.add_argument("--save-json", action="store_true", help="Write detections JSONL")
    p.add_argument("--json-out", default=None, help="JSONL output path (default: input .jsonl)")
    p.add_argument("--device", default=None, choices=["cpu", "cuda", "auto"])
    p.add_argument("--imgsz", type=int, default=None, help="Inference image size")
    p.add_argument("--stride", type=int, default=1, help="Process 1 every N frames")
    p.add_argument("--conf-subtitle", type=float, default=None)
    p.add_argument("--conf-watermark", type=float, default=None)
    p.add_argument("--conf-logo", type=float, default=None)
    p.add_argument("--conf-other_burn_in_text", type=float, default=None)
    return p


def draw_detections(frame, detections, show_labels: bool) -> None:
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det.bbox_xyxy]
        x1, x2 = max(0, min(x1, w - 1)), max(0, min(x2, w - 1))
        y1, y2 = max(0, min(y1, h - 1)), max(0, min(y2, h - 1))
        color = CLASS_COLORS.get(det.class_name, FALLBACK_COLOR)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        if show_labels:
            label = f"{det.class_name} {det.confidence:.2f}"
            (tw, th), _ = cv2.getTextSize(label, font, 0.45, 1)
            bg_y1 = max(y1 - th - 4, 0)
            cv2.rectangle(frame, (x1, bg_y1), (x1 + tw + 2, y1), color, -1)
            cv2.putText(
                frame, label, (x1 + 1, y1 - 2), font, 0.45,
                (0, 0, 0), 1, cv2.LINE_AA,
            )


def has_audio_stream(path: Path) -> bool:
    """Return True when the input media contains at least one audio stream."""
    import subprocess

    try:
        r = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-select_streams", "a", "-show_entries",
                "stream=index", "-of", "csv=p=0", str(path),
            ],
            capture_output=True, text=True,
        )
        return bool(r.stdout.strip())
    except Exception:
        return False


def mux_audio(input_path: Path, video_only: Path, output_path: Path) -> bool:
    """Copy the input audio stream into the annotated video via ffmpeg.

    Returns True on success (or when there is no audio to copy).
    """
    import subprocess

    if not has_audio_stream(input_path):
        return True
    tmp_out = output_path.with_suffix(output_path.suffix + ".muxed.tmp.mp4")
    try:
        r = subprocess.run(
            [
                "ffmpeg", "-y", "-v", "error",
                "-i", str(video_only), "-i", str(input_path),
                "-map", "0:v:0", "-map", "1:a:0",
                "-c:v", "copy", "-c:a", "aac",
                str(tmp_out),
            ],
            capture_output=True, text=True,
        )
        if r.returncode != 0:
            print(f"WARNING: audio muxing failed: {r.stderr.strip()[-300:]}", file=sys.stderr)
            return False
        tmp_out.replace(output_path)
        return True
    except Exception as exc:
        print(f"WARNING: audio muxing failed: {exc}", file=sys.stderr)
        return False


def main() -> int:
    args = build_parser().parse_args()

    input_path = Path(args.input)
    if not input_path.is_file():
        print(f"ERROR: input video not found: {input_path}", file=sys.stderr)
        return 1

    cfg = load_detection_config()
    if args.device is not None:
        cfg.device = args.device
    if args.imgsz is not None:
        cfg.imgsz = args.imgsz
    conf = dict(cfg.confidence)
    if args.conf_subtitle is not None:
        conf["subtitle"] = args.conf_subtitle
    if args.conf_watermark is not None:
        conf["watermark"] = args.conf_watermark
    if args.conf_logo is not None:
        conf["logo"] = args.conf_logo
    if args.conf_other_burn_in_text is not None:
        conf["other_burn_in_text"] = args.conf_other_burn_in_text
    cfg.confidence = conf

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        print(f"ERROR: cannot open video: {input_path}", file=sys.stderr)
        return 1

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height), True)

    detector = YoloBurnInDetector(config=cfg)

    jsonl_path = args.json_out or (input_path.with_suffix(".jsonl"))
    jsonl_fh = None
    if args.save_json:
        jsonl_path = Path(jsonl_path)
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        jsonl_fh = open(jsonl_path, "w", encoding="utf-8")

    stats: Dict[str, int] = {}
    frame_no = 0
    processed = 0
    decode_t0 = time.monotonic()
    infer_t0 = None

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        fi = frame_no
        frame_no += 1
        if args.stride > 1 and (fi % args.stride) != 0:
            writer.write(frame)
            continue

        if infer_t0 is None:
            infer_t0 = time.monotonic()
        ts = fi / fps
        result = detector.detect_frame(frame, frame_index=fi, timestamp_seconds=ts)
        processed += 1

        if args.show_labels or args.save_json:
            draw_detections(frame, result.detections, args.show_labels)
        writer.write(frame)

        for det in result.detections:
            stats[det.class_name] = stats.get(det.class_name, 0) + 1
            if jsonl_fh is not None:
                jsonl_fh.write(json.dumps(det.to_json_compatible(), ensure_ascii=False) + "\n")

    cap.release()
    writer.release()
    if jsonl_fh is not None:
        jsonl_fh.close()
    detector.release()

    # Re-mux to keep the original audio track (cv2.VideoWriter writes video only).
    audio_muxed = mux_audio(input_path, output_path, output_path)
    has_audio = has_audio_stream(input_path)

    decode_elapsed = time.monotonic() - decode_t0
    infer_elapsed = (time.monotonic() - infer_t0) if infer_t0 else 0.0

    print("=== Burn-in detection results ===")
    print(f"Input            : {input_path}")
    print(f"Output           : {output_path}")
    print(f"Resolution       : {width}x{height}")
    print(f"FPS              : {fps}")
    print(f"Frame count      : {frame_no} (processed {processed})")
    print(f"Detections by class: {stats or 'none'}")
    print(f"Audio preserved  : {'yes' if has_audio and audio_muxed else 'no'}")
    if decode_elapsed > 0:
        print(f"Pipeline FPS     : {processed / decode_elapsed:.2f}")
    if infer_elapsed > 0:
        print(f"Inference FPS    : {processed / infer_elapsed:.2f}")
    if args.save_json:
        print(f"JSONL output     : {jsonl_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
