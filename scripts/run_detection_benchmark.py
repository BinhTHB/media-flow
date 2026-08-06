"""Run the YOLO burn-in detector over a directory of QA videos and report stats.

Benchmarks the detection pipeline end-to-end (decode -> detect -> render ->
audio mux) across multiple configurations, reusing a single loaded detector so
the model is never re-loaded between videos.

Usage (PowerShell):
    .venv\\Scripts\\python.exe scripts\\run_detection_benchmark.py `
        --input-dir qa_inputs `
        --output-dir outputs\\qa_benchmark `
        --device cuda

Options:
    --configs A,B,C        config presets to run (A=default, B=strict, C=lax)
    --imgsz 1280           inference image size override
    --max-frames 600       cap frames processed per video (0 = unlimited)
    --no-annotate          skip writing annotated videos (stats only)
    --no-audio             skip audio muxing entirely (speeds up runs)
    --batch 1              frames per model batch (>1 uses detect_batch)
    --stride 1             process 1 of every N frames (annotate copies others)

Outputs (under --output-dir):
    benchmark_results.json
    benchmark_results.csv
    benchmark_summary.md
    annotated/<video>/<config>/detected.mp4 + detections.jsonl (unless
    --no-annotate)

Exit code is non-zero when a video fails entirely or no video was found.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
import numpy as np

from src.infrastructure.detection import (
    DetectionConfig,
    YoloBurnInDetector,
    load_detection_config,
)

#: BGR colors per class for drawing.
CLASS_COLORS = {
    "subtitle": (89, 178, 85),           # green
    "watermark": (77, 55, 216),          # red
    "logo": (255, 172, 27),              # blue
    "other_burn_in_text": (80, 165, 255),  # orange
}
FALLBACK_COLOR = (66, 204, 255)

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}

#: Presets: name -> per-class confidence.
CONFIG_PRESETS: Dict[str, Dict[str, float]] = {
    "A": {"subtitle": 0.15, "watermark": 0.20, "logo": 0.20, "other_burn_in_text": 0.20},
    "B": {"subtitle": 0.20, "watermark": 0.25, "logo": 0.25, "other_burn_in_text": 0.25},
    "C": {"subtitle": 0.10, "watermark": 0.15, "logo": 0.15, "other_burn_in_text": 0.15},
}

CSV_COLUMNS = [
    "video",
    "device",
    "width",
    "height",
    "input_fps",
    "frames",
    "duration_seconds",
    "processing_seconds",
    "pipeline_fps",
    "inference_fps",
    "subtitle_count",
    "watermark_count",
    "logo_count",
    "other_burn_in_text_count",
    "frames_with_detection",
    "audio_present",
    "audio_preserved",
    "output_duration_delta_ms",
    "config_name",
    "imgsz",
    "batch",
]


@dataclass
class VideoStats:
    video: str
    config_name: str
    device: str
    width: int = 0
    height: int = 0
    input_fps: float = 0.0
    frames: int = 0
    duration_seconds: float = 0.0
    processing_seconds: float = 0.0
    decode_seconds: float = 0.0
    inference_seconds: float = 0.0
    render_seconds: float = 0.0
    mux_seconds: float = 0.0
    pipeline_fps: float = 0.0
    inference_fps: float = 0.0
    per_class: Dict[str, int] = field(default_factory=lambda: {
        "subtitle": 0, "watermark": 0, "logo": 0, "other_burn_in_text": 0,
    })
    frames_with_detection: int = 0
    audio_present: bool = False
    audio_preserved: bool = False
    output_duration_delta_ms: float = 0.0
    imgsz: int = 0
    batch: int = 1
    vram_mb: float = 0.0
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# ffmpeg / ffprobe helpers
# ---------------------------------------------------------------------------
def has_audio_stream(path: Path) -> bool:
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a",
             "-show_entries", "stream=index", "-of", "csv=p=0", str(path)],
            capture_output=True, text=True,
        )
        return bool(r.stdout.strip())
    except Exception:
        return False


def probe_duration_ms(path: Path) -> Optional[float]:
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", str(path)],
            capture_output=True, text=True,
        )
        if r.returncode == 0 and r.stdout.strip():
            return float(r.stdout.strip()) * 1000.0
    except Exception:
        pass
    return None


def mux_audio(input_path: Path, video_only: Path, output_path: Path) -> bool:
    """Copy the input audio track into the annotated video via ffmpeg."""
    if not has_audio_stream(input_path):
        return True
    tmp = output_path.with_suffix(output_path.suffix + ".muxed.tmp.mp4")
    try:
        r = subprocess.run(
            ["ffmpeg", "-y", "-v", "error",
             "-i", str(video_only), "-i", str(input_path),
             "-map", "0:v:0", "-map", "1:a:0",
             "-c:v", "copy", "-c:a", "aac",
             str(tmp)],
            capture_output=True, text=True,
        )
        if r.returncode != 0:
            print(f"    WARNING: audio muxing failed: {r.stderr.strip()[-300:]}", file=sys.stderr)
            return False
        tmp.replace(output_path)
        return True
    except Exception as exc:
        print(f"    WARNING: audio muxing failed: {exc}", file=sys.stderr)
        return False


def gpu_vram_mb() -> float:
    """Return used VRAM (MB) for device 0, or 0.0 when unavailable."""
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return round(mem.used / (1024.0 * 1024.0), 1)
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def draw_detections(frame, detections, show_labels: bool = True) -> None:
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
            cv2.putText(frame, label, (x1 + 1, y1 - 2), font, 0.45,
                        (0, 0, 0), 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Per-video runner
# ---------------------------------------------------------------------------
def run_one(
    video_path: Path,
    detector: YoloBurnInDetector,
    config_name: str,
    conf: Dict[str, float],
    *,
    device: str,
    imgsz: int,
    batch: int,
    stride: int,
    max_frames: int,
    annotate: bool,
    mux: bool,
    out_dir: Path,
) -> VideoStats:
    st = VideoStats(video=video_path.name, config_name=config_name, device=device,
                    imgsz=imgsz, batch=batch)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        st.error = f"cannot open video: {video_path}"
        return st

    st.input_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    st.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    st.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_avail = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    st.frames = min(total_avail, max_frames) if max_frames > 0 else total_avail
    st.duration_seconds = round(st.frames / st.input_fps, 3) if st.input_fps else 0.0

    # Annotated video writer (video-only; audio is muxed afterwards).
    writer = None
    jsonl_fh = None
    if annotate:
        vdir = out_dir / "annotated" / video_path.stem / config_name
        vdir.mkdir(parents=True, exist_ok=True)
        video_only = vdir / "detected_video_only.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(video_only), fourcc, st.input_fps,
                                 (st.width, st.height), True)
        jsonl_fh = open(vdir / "detections.jsonl", "w", encoding="utf-8")

    detector.config.confidence = dict(conf)
    detector.temporal.reset()

    # Override detector resolution for the run (stays for every video).
    detector.config.imgsz = imgsz

    frame_no = 0
    processed = 0
    decode_t = 0.0
    infer_t = 0.0
    render_t = 0.0
    t0 = time.monotonic()

    pending: List[Any] = []  # batched frames awaiting inference

    def flush_batch() -> None:
        nonlocal processed, infer_t
        if not pending:
            return
        n = len(pending)
        f0 = frame_no - n  # frame index of first frame in the batch
        t_inf = time.monotonic()
        results = detector.detect_batch(
            pending,
            frame_index_offset=f0,
            timestamps_seconds=[(f0 + i) / st.input_fps for i in range(n)],
        )
        infer_t += time.monotonic() - t_inf
        for res in results:
            _collect(res, jsonl_fh, writer)
        pending.clear()

    def _collect(res, jsonl_fh, writer) -> None:
        nonlocal processed, render_t, st
        processed += 1
        if res.detections:
            st.frames_with_detection += 1
        for det in res.detections:
            st.per_class[det.class_name] = st.per_class.get(det.class_name, 0) + 1
            if jsonl_fh is not None:
                jsonl_fh.write(json.dumps(det.to_json_compatible(), ensure_ascii=False) + "\n")
        if writer is not None and annotate:
            frame = res.bgr_image
            if frame is None:
                return
            frame = frame.copy()
            draw_detections(frame, res.detections)
            t_ren = time.monotonic()
            writer.write(frame)
            render_t += time.monotonic() - t_ren

    while True:
        t_dec = time.monotonic()
        ok, frame = cap.read()
        decode_t += time.monotonic() - t_dec
        if not ok:
            break
        fi = frame_no
        frame_no += 1
        if max_frames > 0 and processed >= max_frames:
            break
        if stride > 1 and (fi % stride) != 0:
            if writer is not None:
                writer.write(frame)
            continue
        if batch > 1:
            pending.append(frame)
            if len(pending) >= batch:
                flush_batch()
        else:
            ts = fi / st.input_fps
            t_inf = time.monotonic()
            res = detector.detect_frame(frame, frame_index=fi, timestamp_seconds=ts)
            infer_t += time.monotonic() - t_inf
            _collect(res, jsonl_fh, writer)

    flush_batch()
    cap.release()
    if writer is not None:
        writer.release()
    if jsonl_fh is not None:
        jsonl_fh.close()

    st.processing_seconds = round(time.monotonic() - t0, 3)
    st.decode_seconds = round(decode_t, 3)
    st.inference_seconds = round(infer_t, 3)
    st.render_seconds = round(render_t, 3)
    st.pipeline_fps = round(processed / st.processing_seconds, 2) if st.processing_seconds else 0.0
    st.inference_fps = round(processed / infer_t, 2) if infer_t else 0.0

    # Audio preservation + duration delta.
    st.audio_present = has_audio_stream(video_path)
    if annotate:
        vdir = out_dir / "annotated" / video_path.stem / config_name
        video_only = vdir / "detected_video_only.mp4"
        out_video = vdir / "detected.mp4"
        if mux:
            t_mux = time.monotonic()
            st.audio_preserved = mux_audio(video_path, video_only, out_video)
            st.mux_seconds = round(time.monotonic() - t_mux, 3)
            if st.audio_preserved and out_video.exists():
                in_ms = probe_duration_ms(video_path)
                out_ms = probe_duration_ms(out_video)
                if in_ms is not None and out_ms is not None:
                    st.output_duration_delta_ms = round(out_ms - in_ms, 1)
            # When the input has no audio, mux_audio returns early without
            # creating out_video; rename the video-only file then.
            if not out_video.exists() and video_only.exists():
                video_only.replace(out_video)
        else:
            video_only.replace(out_video)
        if not st.audio_preserved and st.audio_present:
            print(f"    ERROR: audio mux failed for {video_path.name} "
                  f"({config_name})", file=sys.stderr)
            st.error = st.error or "audio mux failed"
        # Remove the intermediate video-only file when both exist.
        try:
            if video_only.exists() and out_video.exists():
                video_only.unlink()
        except OSError:
            pass

    st.vram_mb = gpu_vram_mb() if device in {"cuda", "auto"} else 0.0
    return st


def stats_to_row(st: VideoStats) -> Dict[str, Any]:
    return {
        "video": st.video,
        "device": st.device,
        "width": st.width,
        "height": st.height,
        "input_fps": round(st.input_fps, 3),
        "frames": st.frames,
        "duration_seconds": st.duration_seconds,
        "processing_seconds": st.processing_seconds,
        "pipeline_fps": st.pipeline_fps,
        "inference_fps": st.inference_fps,
        "subtitle_count": st.per_class.get("subtitle", 0),
        "watermark_count": st.per_class.get("watermark", 0),
        "logo_count": st.per_class.get("logo", 0),
        "other_burn_in_text_count": st.per_class.get("other_burn_in_text", 0),
        "frames_with_detection": st.frames_with_detection,
        "audio_present": st.audio_present,
        "audio_preserved": st.audio_preserved,
        "output_duration_delta_ms": st.output_duration_delta_ms,
        "config_name": st.config_name,
        "imgsz": st.imgsz,
        "batch": st.batch,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="YOLO burn-in detection benchmark")
    p.add_argument("--input-dir", required=True, help="Directory with QA videos")
    p.add_argument("--output-dir", default="outputs/qa_benchmark")
    p.add_argument("--device", default="auto", choices=["cpu", "cuda", "auto"])
    p.add_argument("--configs", default="A,B", help="Comma-separated presets (A,B,C)")
    p.add_argument("--imgsz", type=int, default=None, help="Override inference size")
    p.add_argument("--max-frames", type=int, default=0, help="Cap frames per video (0=all)")
    p.add_argument("--batch", type=int, default=1, help="Frames per model batch")
    p.add_argument("--stride", type=int, default=1, help="Process 1 of every N frames")
    p.add_argument("--no-annotate", action="store_true", help="Skip annotated videos")
    p.add_argument("--no-audio", action="store_true", help="Skip audio muxing")
    return p


def main() -> int:
    args = build_parser().parse_args()
    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    videos = sorted(
        p for p in in_dir.iterdir()
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS
    )
    if not videos:
        print(f"ERROR: no videos found in {in_dir}", file=sys.stderr)
        return 2
    print(f"Found {len(videos)} videos in {in_dir}")

    configs = [c.strip().upper() for c in args.configs.split(",") if c.strip()]
    unknown = [c for c in configs if c not in CONFIG_PRESETS]
    if unknown:
        print(f"ERROR: unknown config presets: {unknown}", file=sys.stderr)
        return 2

    base_cfg = load_detection_config()
    device = args.device or base_cfg.device
    imgsz = args.imgsz or base_cfg.imgsz

    # One detector for the whole run: the model is loaded exactly once.
    print(f"Loading detector once (device={device}, imgsz={imgsz}) ...")
    t_load = time.monotonic()
    detector = YoloBurnInDetector(config=base_cfg, device=device, imgsz=imgsz)
    print(f"  detector ready in {time.monotonic() - t_load:.2f}s")

    all_stats: List[VideoStats] = []
    serious_errors = 0
    for video in videos:
        for cname in configs:
            print(f"[{cname}] {video.name} ({video.stat().st_size // 1024} KB) ...")
            try:
                st = run_one(
                    video, detector, cname, dict(CONFIG_PRESETS[cname]),
                    device=device, imgsz=imgsz, batch=args.batch,
                    stride=args.stride, max_frames=args.max_frames,
                    annotate=not args.no_annotate, mux=not args.no_audio,
                    out_dir=out_dir,
                )
            except Exception as exc:  # keep going with next video/config
                print(f"    ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
                st = VideoStats(video=video.name, config_name=cname, device=device,
                                imgsz=imgsz, batch=args.batch,
                                error=f"{type(exc).__name__}: {exc}")
                serious_errors += 1
            all_stats.append(st)
            if st.error and not st.frames:
                serious_errors += 1
            print(f"    frames={st.frames} det={dict(st.per_class)} "
                  f"pipeline={st.pipeline_fps} fps inference={st.inference_fps} fps "
                  f"vram={st.vram_mb} MB err={st.error or 'none'}")

    detector.release()

    rows = [stats_to_row(s) for s in all_stats]
    json_path = out_dir / "benchmark_results.json"
    csv_path = out_dir / "benchmark_results.csv"
    md_path = out_dir / "benchmark_summary.md"

    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump({
            "device": device,
            "imgsz": imgsz,
            "configs": configs,
            "n_videos": len(videos),
            "results": rows,
        }, fh, indent=2, ensure_ascii=False)

    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in CSV_COLUMNS})

    _write_summary(md_path, rows, device, imgsz, configs)

    print(f"\nWrote {json_path.name}, {csv_path.name}, {md_path.name} under {out_dir}")
    print(f"Serious errors: {serious_errors}")
    return 1 if serious_errors else 0


def _write_summary(md_path: Path, rows: List[Dict[str, Any]], device: str,
                   imgsz: int, configs: List[str]) -> None:
    lines = [
        "# YOLO detection benchmark summary",
        "",
        f"- Device: `{device}`",
        f"- imgsz: `{imgsz}`",
        f"- Configs: `{', '.join(configs)}`",
        f"- Runs: {len(rows)}",
        "",
        "| Video | Config | Frames | Pipeline FPS | Infer FPS | Subtitle | Watermark | Logo | Other text | Frames w/ det | Audio in | Audio out |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['video']} | {r['config_name']} | {r['frames']} | "
            f"{r['pipeline_fps']} | {r['inference_fps']} | "
            f"{r['subtitle_count']} | {r['watermark_count']} | {r['logo_count']} | "
            f"{r['other_burn_in_text_count']} | {r['frames_with_detection']} | "
            f"{'yes' if r['audio_present'] else 'no'} | "
            f"{'yes' if r['audio_preserved'] else 'no'} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
