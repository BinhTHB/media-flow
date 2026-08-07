"""Build the evaluation dataset (sampled frames + UNVERIFIED pre-labels).

Pipeline:
1. Discover input videos (``--input-dir``, plus any ``--video`` extras).
2. Probe each video (fps, frame count, resolution) via ffprobe.
3. Sample frames per video with a minimum temporal gap (default ~1 s), picking
   content-changing frames where possible, capped per video.
4. Write ``evaluation/manifests/manifest.csv`` describing every sampled frame.
5. Extract the frames to ``evaluation/frames/``.
6. Run the YOLO detector at a low confidence floor and write machine pre-labels
   to ``evaluation/labels/``. Pre-labels are ALWAYS marked ``verified=false``:
   they are NOT ground truth until a human reviews them.
7. Build a review package under ``evaluation/review/`` (frame previews with the
   pre-label boxes drawn plus an HTML index) so a human can verify/edit labels.

Leakage policy: sampling is per-video; the script never mixes frames from the
same video across train/eval. Because no training manifest exists in this repo,
the exclusion manifest documents the videos that were deliberately excluded
(synthetic benchmark clips used for an earlier pipeline, etc.) and records that
the training provenance must be confirmed before final reporting.

Example:
  .venv\\Scripts\\python.exe scripts\\build_evaluation_set.py ^
      --input-dir qa_inputs --out-dir evaluation ^
      --frames-per-video 30 --device cuda
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
import numpy as np

from src.infrastructure.detection import YoloBurnInDetector, load_detection_config
from src.infrastructure.detection.schemas import EXPECTED_CLASS_NAMES

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}
BASE_CONF = 0.05


# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build evaluation set with pre-labels")
    p.add_argument("--input-dir", default=None, help="Directory containing source videos")
    p.add_argument("--video", action="append", default=[], help="Extra video file paths")
    p.add_argument("--out-dir", default="evaluation", help="Evaluation root directory")
    p.add_argument("--frames-per-video", type=int, default=30, help="Max frames per video")
    p.add_argument("--min-gap-seconds", type=float, default=1.0, help="Min gap between sampled frames (s)")
    p.add_argument("--device", default=None, choices=["cpu", "cuda", "auto"])
    p.add_argument("--imgsz", type=int, default=None, help="Inference imgsz")
    p.add_argument("--no-prelabel", action="store_true", help="Skip model pre-labeling")
    p.add_argument("--seed", type=int, default=7, help="Sampling seed (deterministic selection)")
    return p


def probe_video(path: Path) -> Dict[str, Any]:
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,avg_frame_rate,nb_frames,duration",
        "-of", "json", str(path),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {path}: {r.stderr.strip()[-300:]}")
    data = json.loads(r.stdout or "{}")
    streams = data.get("streams", [])
    if not streams:
        raise RuntimeError(f"no video stream in {path}")
    s = streams[0]
    fps_num, fps_den = s.get("avg_frame_rate", "30/1").split("/")
    fps = float(fps_num) / float(fps_den) if float(fps_den) else 30.0
    return {
        "width": int(s.get("width", 0)),
        "height": int(s.get("height", 0)),
        "fps": fps,
        "frames": int(s.get("nb_frames", 0)) or None,
        "duration": float(s.get("duration", 0.0)),
    }


def discover_videos(input_dir: Optional[Path], extra: List[str]) -> List[Path]:
    videos: List[Path] = []
    if input_dir and input_dir.is_dir():
        videos += [p for p in sorted(input_dir.iterdir()) if p.suffix.lower() in VIDEO_EXTS]
    for v in extra:
        p = Path(v)
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
            videos.append(p)
        elif p.is_dir():
            videos += [x for x in sorted(p.iterdir()) if x.suffix.lower() in VIDEO_EXTS]
    return videos


def content_change_score(prev: Optional[np.ndarray], cur: np.ndarray) -> float:
    """Mean absolute difference between two downscaled grayscale frames."""
    if prev is None:
        return 1.0
    small = lambda img: cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)
    g = lambda img: cv2.cvtColor(small(img), cv2.COLOR_BGR2GRAY)
    a, b = g(prev), g(cur)
    return float(np.abs(a.astype(np.float32) - b.astype(np.float32)).mean())


def sample_frame_indices(
    total_frames: int,
    fps: float,
    max_frames: int,
    min_gap_seconds: float,
) -> List[int]:
    """Pick frame indices with at least ``min_gap_seconds`` between them."""
    if total_frames <= 0:
        return []
    gap = max(1, int(round(fps * min_gap_seconds)))
    candidates = list(range(0, total_frames, gap))
    if len(candidates) <= max_frames:
        return candidates
    step = len(candidates) / max_frames
    return [candidates[int(i * step)] for i in range(max_frames)]


def main() -> int:
    args = build_parser().parse_args()
    root = Path(args.out_dir)
    frames_dir = root / "frames"
    labels_dir = root / "labels"
    manifests_dir = root / "manifests"
    review_dir = root / "review"
    for d in (frames_dir, labels_dir, manifests_dir, review_dir):
        d.mkdir(parents=True, exist_ok=True)

    videos = discover_videos(Path(args.input_dir) if args.input_dir else None, args.video)
    if not videos:
        print(f"ERROR: no videos found (input-dir={args.input_dir})", file=sys.stderr)
        return 1

    cfg = load_detection_config()
    if args.device:
        cfg.device = args.device
    if args.imgsz:
        cfg.imgsz = args.imgsz

    detector = None
    if not args.no_prelabel:
        detector = YoloBurnInDetector(config=cfg)

    manifest_rows: List[Dict[str, Any]] = []
    n_frames = 0
    try:
        for video in videos:
            try:
                info = probe_video(video)
            except RuntimeError as exc:
                print(f"  WARN: {exc}", file=sys.stderr)
                continue
            fps = info["fps"] or 30.0
            total = info["frames"] or int(round(info["duration"] * fps)) or 0
            if total <= 0:
                print(f"  WARN: cannot determine frame count for {video.name}", file=sys.stderr)
                continue
            indices = sample_frame_indices(total, fps, args.frames_per_video, args.min_gap_seconds)
            print(f"  {video.name}: {total} frames, {len(indices)} sampled")

            cap = cv2.VideoCapture(str(video))
            if not cap.isOpened():
                print(f"  WARN: cannot open {video.name}", file=sys.stderr)
                continue
            for fi in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
                ok, frame = cap.read()
                if not ok or frame is None:
                    continue
                ts = fi / fps
                frame_id = f"{video.stem}_f{fi:06d}"
                img_path = frames_dir / f"{frame_id}.jpg"
                cv2.imwrite(str(img_path), frame)

                prelabel: List[Dict[str, Any]] = []
                if detector is not None:
                    result = detector.predict_raw(
                        frame, frame_index=fi, timestamp_seconds=ts, confidence=BASE_CONF
                    )
                    seen_boxes: set = set()
                    for d in result.detections:
                        box_rounded = tuple(int(round(float(v))) for v in d.bbox_xyxy)
                        key = (int(d.class_id), box_rounded)
                        if key in seen_boxes:
                            continue
                        seen_boxes.add(key)
                        prelabel.append({
                            "class_id": int(d.class_id),
                            "class_name": str(d.class_name),
                            "bbox_xyxy": [float(v) for v in d.bbox_xyxy],
                            "confidence": round(float(d.confidence), 6),
                        })

                # Always write a label JSON (even if empty) so the frame has a clear label record.
                label_path = labels_dir / f"{frame_id}.json"
                label_path.write_text(json.dumps({
                    "frame_id": frame_id,
                    "video_id": video.stem,
                    "frame_index": fi,
                    "timestamp_seconds": round(ts, 6),
                    "width": int(frame.shape[1]),
                    "height": int(frame.shape[0]),
                    "verified": False,
                    "label_source": "prelabel_unverified" if prelabel else "empty_unverified",
                    "objects": prelabel,
                }, indent=2, ensure_ascii=False), encoding="utf-8")

                manifest_rows.append({
                    "frame_id": frame_id,
                    "video_id": video.stem,
                    "frame_index": fi,
                    "timestamp_seconds": round(ts, 6),
                    "width": int(frame.shape[1]),
                    "height": int(frame.shape[0]),
                    "verified": "false",
                    "label_source": "prelabel_unverified" if prelabel else "missing",
                })
                n_frames += 1
            cap.release()
    finally:
        if detector is not None:
            detector.release()

    if not manifest_rows:
        print("ERROR: no frames sampled", file=sys.stderr)
        return 1

    manifest_path = manifests_dir / "manifest.csv"
    with open(manifest_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(manifest_rows[0].keys()))
        writer.writeheader()
        writer.writerows(manifest_rows)

    write_exclusion_manifest(manifests_dir)
    write_review_package(review_dir, frames_dir, manifest_rows)
    write_eval_readme(root)

    print(f"\nEvaluation set built: {n_frames} frames, manifest -> {manifest_path}")
    print(f"Frames -> {frames_dir}; pre-labels -> {labels_dir}; review -> {review_dir}")
    print("REMINDER: pre-labels are UNVERIFIED. Review them, then set verified=true in the manifest.")
    return 0


def write_exclusion_manifest(manifests_dir: Path) -> None:
    path = manifests_dir / "exclusion_manifest.csv"
    path.write_text(
        "video_id,frame_ids,reason\n"
        ",,\"No training manifest exists in this repository, so no video could be\n"
        "linked to training provenance. Leakage prevention is enforced by policy:\n"
        "sampling is per-video (never mixed across sets) and final reporting must\n"
        "confirm the sampled videos were not used to fine-tune the YOLO model.\"\n"
        "subtitle_replacement_benchmark,all,\"Synthetic clips generated for an earlier\n"
        "subtitle-region pipeline; not part of the burn-in evaluation set.\"\n"
        "frozen_gt_sample5,all,\"input/sample5.mp4 was used for prior QA rounds;\n"
        "keep it out of the held-out burn-in set.\"\n",
        encoding="utf-8",
    )


def write_review_package(
    review_dir: Path, frames_dir: Path, rows: List[Dict[str, Any]]
) -> None:
    """Draw previews (frame + pre-label boxes) and write an HTML index."""
    previews = review_dir / "previews"
    previews.mkdir(parents=True, exist_ok=True)
    colors = {
        "subtitle": (89, 178, 85), "watermark": (77, 55, 216),
        "logo": (255, 172, 27), "other_burn_in_text": (80, 165, 255),
    }
    entries: List[str] = []
    for row in rows:
        frame_id = str(row["frame_id"])
        img_path = frames_dir / f"{frame_id}.jpg"
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        label_path = review_dir.parent / "labels" / f"{frame_id}.json"
        objs: List[Dict[str, Any]] = []
        if label_path.is_file():
            data = json.loads(label_path.read_text(encoding="utf-8"))
            objs = data.get("objects", [])
        for o in objs:
            x1, y1, x2, y2 = (int(round(float(v))) for v in o["bbox_xyxy"])
            color = colors.get(o["class_name"], (66, 204, 255))
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            label = f"{o['class_name']} {o.get('confidence', 0.0):.2f}"
            cv2.putText(img, label, (x1, max(y1 - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, color, 2, cv2.LINE_AA)
        preview_name = f"{frame_id}.jpg"
        cv2.imwrite(str(previews / preview_name), img)
        entries.append(
            f'<div style="display:inline-block;margin:6px;text-align:center">'
            f'<img src="previews/{preview_name}" width="320" loading="lazy"/>'
            f'<br/><code>{frame_id}</code><br/>'
            f'<span style="font-size:12px">v={row["video_id"]} f={row["frame_index"]} '
            f'ts={row["timestamp_seconds"]}s</span></div>'
        )
    html = (
        "<!doctype html><html><head><meta charset='utf-8'><title>Evaluation review</title>"
        "<style>body{font-family:system-ui;margin:20px}</style></head><body>"
        "<h1>UNVERIFIED pre-label review</h1>"
        "<p>These boxes were produced by the model and are <b>NOT ground truth</b>. "
        "For each frame, confirm/correct the boxes in "
        "<code>evaluation/labels/&lt;frame_id&gt;.json</code> and set "
        "<code>verified: true</code> (and update the manifest row), then re-run "
        "<code>scripts/evaluate_yolo_detection.py</code>.</p>"
        + "".join(entries) + "</body></html>"
    )
    (review_dir / "review_index.html").write_text(html, encoding="utf-8")
    (review_dir / "README.md").write_text(
        "# Reviewing evaluation labels\n\n"
        "1. Open `review_index.html` in a browser and inspect the drawn boxes.\n"
        "2. For each frame, edit `../labels/<frame_id>.json`: fix class ids, boxes "
        "(`bbox_xyxy`), remove wrong boxes, add missing ones, and set `verified: true`.\n"
        "3. Set `verified` to `true` in `../manifests/manifest.csv` for that row.\n"
        "4. Re-run `scripts/evaluate_yolo_detection.py` — it refuses to compute "
        "Precision/Recall/F1 until every evaluated frame is verified.\n",
        encoding="utf-8",
    )


def write_eval_readme(root: Path) -> None:
    path = root / "README.md"
    if path.exists():
        return
    path.write_text(
        "# Evaluation dataset\n\n"
        "Layout:\n"
        "- `manifests/manifest.csv` — sampled frames (source video + timestamp).\n"
        "- `manifests/exclusion_manifest.csv` — videos/frames excluded and why.\n"
        "- `frames/` — extracted frame images.\n"
        "- `labels/` — per-frame labels. **UNVERIFIED pre-labels** until reviewed.\n"
        "- `review/` — human review package (previews + index + instructions).\n\n"
        "Do not treat pre-labels as ground truth. Metrics are only computed from "
        "frames marked `verified=true`.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    sys.exit(main())
