"""Add independent negative / background frames to the evaluation set.

The existing sampled frames are ~92% positive (contain a model box), which is a
positive-frame bias. This script appends additional frames per video that are
**not** selected by model detection: it picks interleaved timestamps in the gaps
between already-sampled frames, regardless of content. Pre-labels (if any) are
written but kept ``verified=false``; a human decides the final 0-object label in
the review UI. It never auto-verifies anything.

Usage:
  .venv\\Scripts\\python.exe scripts\\add_background_frames.py ^
      --manifest evaluation/manifests/manifest.csv ^
      --labels evaluation/labels ^
      --frames evaluation/frames ^
      --input-dir qa_inputs ^
      --extra-frames-per-video 5 ^
      --device cuda

New frames are appended to the manifest and labels; no existing frame is
touched. Run ``scripts/validate_evaluation_labels.py`` afterwards.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Set

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2

from src.infrastructure.detection import YoloBurnInDetector, load_detection_config
from src.infrastructure.detection.evaluation.review_store import MANIFEST_COLUMNS

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}
BASE_CONF = 0.05


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Add background frames to evaluation set")
    p.add_argument("--manifest", required=True)
    p.add_argument("--labels", required=True)
    p.add_argument("--frames", required=True)
    p.add_argument("--input-dir", required=True, help="Directory of source videos")
    p.add_argument("--extra-frames-per-video", type=int, default=6)
    p.add_argument("--device", default=None, choices=["cpu", "cuda", "auto"])
    p.add_argument("--imgsz", type=int, default=None)
    return p


def _read_manifest(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8", newline="") as fh:
        return [dict(r) for r in csv.DictReader(fh)]


def _write_manifest(path: Path, rows: List[Dict[str, Any]]) -> None:
    cols = MANIFEST_COLUMNS + [c for c in rows[0] if c not in MANIFEST_COLUMNS]
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    args = build_parser().parse_args()
    manifest = _read_manifest(Path(args.manifest))
    existing_frames = {(str(r["video_id"]), int(r["frame_index"])) for r in manifest}

    cfg = load_detection_config()
    if args.device:
        cfg.device = args.device
    if args.imgsz:
        cfg.imgsz = args.imgsz
    detector = YoloBurnInDetector(config=cfg)

    in_dir = Path(args.input_dir)
    frames_dir = Path(args.frames)
    labels_dir = Path(args.labels)
    video_files = sorted(p for p in in_dir.iterdir() if p.suffix.lower() in VIDEO_EXTS)

    added = 0
    try:
        for video in video_files:
            vid = video.stem
            cap = cv2.VideoCapture(str(video))
            if not cap.isOpened():
                print(f"  WARN: cannot open {video.name}", file=sys.stderr)
                continue
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            sampled = sorted(set(fi for (v, fi) in existing_frames if v == vid))
            candidates: List[int] = []
            if sampled:
                for i in range(len(sampled) - 1):
                    lo, hi = sampled[i], sampled[i + 1]
                    gap = hi - lo
                    if gap >= 3:
                        mid = (lo + hi) // 2
                        candidates.append(mid)
                # also a frame before the first and after the last
                if sampled[0] >= 3:
                    candidates.append(sampled[0] - 1)
                if total and sampled[-1] + 1 < total:
                    candidates.append(sampled[-1] + 1)
            else:
                candidates = list(range(0, max(total - 1, 0), max(int(fps), 1)))[: args.extra_frames_per_video]

            # de-dup and cap
            seen: Set[int] = set()
            picked = []
            for fi in candidates:
                if fi in seen or fi < 0 or (total and fi >= total):
                    continue
                if (vid, fi) in existing_frames:
                    continue
                seen.add(fi)
                picked.append(fi)
            picked = picked[: args.extra_frames_per_video]

            for fi in picked:
                cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
                ok, frame = cap.read()
                if not ok or frame is None:
                    continue
                ts = fi / fps
                frame_id = f"{vid}_f{fi:06d}"
                cv2.imwrite(str(frames_dir / f"{frame_id}.jpg"), frame)

                prelabel: List[Dict[str, Any]] = []
                result = detector.predict_raw(
                    frame, frame_index=fi, timestamp_seconds=ts, confidence=BASE_CONF
                )
                for d in result.detections:
                    prelabel.append({
                        "class_id": int(d.class_id),
                        "class_name": str(d.class_name),
                        "bbox_xyxy": [float(v) for v in d.bbox_xyxy],
                        "confidence": round(float(d.confidence), 6),
                    })

                (labels_dir / f"{frame_id}.json").write_text(json.dumps({
                    "frame_id": frame_id, "video_id": vid, "frame_index": fi,
                    "timestamp_seconds": round(ts, 6),
                    "width": int(frame.shape[1]), "height": int(frame.shape[0]),
                    "verified": False, "needs_review": False, "needs_second_review": False,
                    "evaluation_role": "unknown", "leakage_risk": False,
                    "label_source": "prelabel_unverified" if prelabel else "empty_unverified",
                    "objects": prelabel,
                }, indent=2, ensure_ascii=False), encoding="utf-8")

                manifest.append(_interval_manifest_row(
                    frame_id, vid, fi, ts, int(frame.shape[1]), int(frame.shape[0]), bool(prelabel)))
                existing_frames.add((vid, fi))
                added += 1
            cap.release()
    finally:
        detector.release()

    _write_manifest(Path(args.manifest), manifest)
    print(f"Added {added} frames. Total manifest rows: {len(manifest)}")
    print("Run scripts/validate_evaluation_labels.py to verify schema.")
    return 0


def _interval_manifest_row(frame_id: str, vid: str, fi: int, ts: float, w: int, h: int, has_prelabel: bool) -> Dict[str, Any]:
    return {
        "frame_id": frame_id, "video_id": vid, "frame_index": fi,
        "timestamp_seconds": round(ts, 6), "width": w, "height": h,
        "verified": "false", "needs_review": "false", "needs_second_review": "false",
        "evaluation_role": "unknown", "leakage_risk": "false",
        "label_source": "prelabel_unverified" if has_prelabel else "missing",
    }


if __name__ == "__main__":
    sys.exit(main())
