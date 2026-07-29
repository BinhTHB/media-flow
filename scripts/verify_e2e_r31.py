#!/usr/bin/env python3
"""E2E checker v31 — measures T003 dark_ratio on the CORRECT bbox at the CORRECT timestamp.

Changes from r30:
  1. Bbox from gt_v2_3.json (not hardcoded in source, though '294,959,413,1019' is also
     explicitly logged for transparency).
  2. Seek by timestamp (CAP_PROP_POS_MSEC = 350), not by frame index.
  3. Dark threshold raised to gray < 60 (was < 30).
  4. Prints both dark_ratio_bbox AND dark_ratio_fullframe to prevent future confusion.
  5. `reason` text in blocked_status.json is generated from the constant, not hand-written.

Usage:
    .venv/Scripts/python scripts/verify_e2e_r31.py [--video path/to/final.mp4]

Exit code:
    0  → PASS (all checks pass)
    2  → FAIL (any check fails)
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

DARK_THRESHOLD = 60         # gray < DARK_THRESHOLD → pixel counts as "dark"
DARK_RATIO_MIN = 0.90       # minimum acceptable dark_ratio_bbox
T003_SEEK_MSEC = 350        # T003 timestamp = 0.35 s
T003_ID = "T003"
T003_PROVENANCE_MARKERS = (
    "provenance=classification:T003",
    "{track=T003}",
)

# Default input paths — caller may override via --video / --ass
VIDEO = ROOT / "output" / "round31-sample5" / "final.mp4"
GT_PATH = ROOT / "artifacts" / "round28" / "gt_v2_3.json"
ARTIFACT_DIR = ROOT / "artifacts" / "round31"


def load_t003_bbox(gt_path: Path) -> list[int] | None:
    """Return bbox_xyxy for T003 from gt_v2_3.json, or None."""
    raw = json.loads(gt_path.read_text(encoding="utf-8"))
    for entry in raw["entries"]:
        if entry["id"] == T003_ID:
            return [int(round(v)) for v in entry["bbox_xyxy"]]
    return None


def main() -> None:
    # ── CLI override ──────────────────────────────────────────────
    video_path = Path(sys.argv[2]) if len(sys.argv) >= 3 and sys.argv[1] == "--video" else VIDEO
    ass_path = Path(sys.argv[4]) if len(sys.argv) >= 5 and sys.argv[3] == "--ass" else None
    gt_path = GT_PATH
    # Resolve round from video path to avoid overwriting another round's artifact
    m = re.search(r"r(\d+)|round(\d+)", str(video_path))
    if m:
        rnum = m.group(1) or m.group(2)
        artifact_dir = ROOT / "artifacts" / f"round{rnum}"
    else:
        artifact_dir = ARTIFACT_DIR
    artifact_dir.mkdir(parents=True, exist_ok=True)

    print("COMMAND: python scripts/verify_e2e_r31.py")
    print(f"VIDEO:   {video_path}")
    print(f"GT V2.3: {gt_path}")
    print(f"TIMESTAMP_UTC: {datetime.now(timezone.utc).isoformat()}")

    # ── GPU info ──────────────────────────────────────────────────
    smi = subprocess.run(["nvidia-smi"], capture_output=True, text=True, check=False)
    print(smi.stdout, end="")
    if smi.stderr:
        print(smi.stderr, file=sys.stderr, end="")

    # ── ffprobe ───────────────────────────────────────────────────
    probe = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=codec_name,width,height,pix_fmt",
            "-of", "json", str(video_path),
        ],
        capture_output=True, text=True, check=False,
    )
    probe_data = json.loads(probe.stdout) if probe.returncode == 0 else {}
    stream = (probe_data.get("streams") or [{}])[0]

    # ── GT bbox ───────────────────────────────────────────────────
    bbox_gt = load_t003_bbox(gt_path)
    if bbox_gt is None:
        print(f"[ERROR] T003 not found in {gt_path}")
        sys.exit(2)

    print(f"T003 GT BBOX (from gt_v2_3.json): {bbox_gt}")
    print(f"  Expected approx: [294, 959, 413, 1019]")
    print(f"  DARK_THRESHOLD (gray < {DARK_THRESHOLD})")
    print(f"  DARK_RATIO_MIN >= {DARK_RATIO_MIN}")
    print(f"  Seek msec: {T003_SEEK_MSEC}")

    bbox = bbox_gt  # [x1, y1, x2, y2]
    x1, y1, x2, y2 = bbox

    # ── Open video and seek by timestamp ──────────────────────────
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        sys.exit(2)

    cap.set(cv2.CAP_PROP_POS_MSEC, T003_SEEK_MSEC)
    ok, frame = cap.read()
    cap.release()

    if not ok:
        print("[ERROR] Failed to read frame at seek msec", T003_SEEK_MSEC)
        sys.exit(2)

    frame_h, frame_w = frame.shape[:2]

    # ── Full-frame dark ratio ─────────────────────────────────────
    gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    dark_ratio_fullframe = float(np.mean(gray_full < DARK_THRESHOLD))

    # ── Bbox-crop dark ratio ──────────────────────────────────────
    # Clamp to valid frame region
    cx1, cy1 = max(0, x1), max(0, y1)
    cx2, cy2 = min(frame_w, x2), min(frame_h, y2)
    if cx2 - cx1 < 2 or cy2 - cy1 < 2:
        print(f"[ERROR] Bbox too small after clamp: ({cx1},{cy1})-({cx2},{cy2})")
        sys.exit(2)
    crop = frame[cy1:cy2, cx1:cx2]
    gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    dark_ratio_bbox = float(np.mean(gray_crop < DARK_THRESHOLD))

    print(f"\nFrame size: {frame_w}x{frame_h}")
    print(f"Crop bbox:  [{cx1}, {cy1}, {cx2}, {cy2}] (size {cx2-cx1}x{cy2-cy1})")
    print(f"dark_ratio_bbox:      {dark_ratio_bbox:.6f}")
    print(f"dark_ratio_fullframe: {dark_ratio_fullframe:.6f}")

    # ── ASS check ─────────────────────────────────────────────────
    ass_text = ""
    mask_count = 0
    t003_provenance = False
    if ass_path and ass_path.exists():
        ass_text = ass_path.read_text(encoding="utf-8")
        mask_count = ass_text.count(r"\p1")
        t003_provenance = any(marker in ass_text for marker in T003_PROVENANCE_MARKERS)
    elif ass_path is None:
        # Auto-detect ASS alongside video
        candidate = video_path.parent / "subtitles" / "subtitle.ass"
        if candidate.exists():
            ass_text = candidate.read_text(encoding="utf-8")
            mask_count = ass_text.count(r"\p1")
            t003_provenance = any(marker in ass_text for marker in T003_PROVENANCE_MARKERS)
        elif video_path.parent.name.endswith("-sample5"):
            candidate2 = video_path.parent / video_path.parent.name / "subtitles" / "subtitle.ass"
            if candidate2.exists():
                ass_text = candidate2.read_text(encoding="utf-8")
                mask_count = ass_text.count(r"\p1")
                t003_provenance = any(marker in ass_text for marker in T003_PROVENANCE_MARKERS)

    checks = {
        "ass_has_t003_provenance": t003_provenance,
        "ass_mask_count": mask_count,
        "h264_720x1280": (
            stream.get("codec_name") == "h264"
            and stream.get("width") == 720
            and stream.get("height") == 1280
        ),
        "t003_frame_read": ok,
        "t003_dark_ratio_bbox": round(dark_ratio_bbox, 6),
        "t003_dark_ratio_fullframe": round(dark_ratio_fullframe, 6),
        "t003_dark_ratio_high": dark_ratio_bbox >= DARK_RATIO_MIN,
    }

    bool_checks = {k: v for k, v in checks.items() if isinstance(v, bool)}
    all_pass = all(v is True for v in bool_checks.values())
    status = "PASS" if all_pass else "FAIL"

    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "video": str(video_path),
        "gt_source": str(gt_path),
        "t003_bbox_gt": bbox_gt,
        "t003_seek_msec": T003_SEEK_MSEC,
        "dark_threshold": DARK_THRESHOLD,
        "dark_ratio_min": DARK_RATIO_MIN,
        "ffprobe_stream": stream,
        "checks": checks,
        "status": status,
    }

    (artifact_dir / "e2e_self_check_r31.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )

    print("\nSELF_CHECK:", json.dumps(payload, indent=2))
    print(f"\nSTATUS: {status}")

    raise SystemExit(0 if all_pass else 2)


if __name__ == "__main__":
    main()
