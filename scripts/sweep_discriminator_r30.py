#!/usr/bin/env python3
"""Static discriminator sweep on all 42 GT V2.3 entries — pixel variance only.
Tín hiệu speech_overlap bỏ (vòng 29 cho thấy ~1.0 cho mọi track).

Usage:
    .venv/Scripts/python scripts/sweep_discriminator_r30.py

Outputs:
    artifacts/round30/discriminator_42.json
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

VIDEO_PATH = ROOT / "test_videos" / "real_mini_benchmark" / "sample5.mp4"
GT_PATH = ROOT / "artifacts" / "round28" / "gt_v2_3.json"
OUTPUT_PATH = ROOT / "artifacts" / "round30" / "discriminator_42.json"

# Positive track IDs from GT V2.3
POSITIVE_IDS = {
    "T003", "T006", "T010", "T014", "T016", "T019",
    "T116", "T120", "T127", "T133", "T135", "T137", "T138", "T146",
}

FRAME_HEIGHT = 1280
FRAME_WIDTH = 720


def load_gt_entries() -> list[dict]:
    raw = json.loads(GT_PATH.read_text(encoding="utf-8"))
    return raw["entries"]


def decode_frames() -> tuple[list[np.ndarray], float]:
    """Decode all frames once; GT is bounded to one 24-second video."""
    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {VIDEO_PATH}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames: list[np.ndarray] = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames, fps


def extract_temporal_variance(
    entry: dict, frames: list[np.ndarray], fps: float, half_window_frames: int = 6
) -> tuple[float, int]:
    """Pixel variance across aligned, same-bbox crops over ±0.2 seconds."""
    bbox = entry["bbox_xyxy"]
    x1, y1, x2, y2 = [int(round(value)) for value in bbox]
    center = int(round(float(entry["timestamp_s"]) * fps))
    crops: list[np.ndarray] = []
    for frame_index in range(
        max(0, center - half_window_frames),
        min(len(frames), center + half_window_frames + 1),
    ):
        frame = frames[frame_index]
        height, width = frame.shape[:2]
        cx1, cy1 = max(0, min(width - 1, x1)), max(0, min(height - 1, y1))
        cx2, cy2 = max(0, min(width, x2)), max(0, min(height, y2))
        if cx2 - cx1 < 2 or cy2 - cy1 < 2:
            continue
        crop = frame[cy1:cy2, cx1:cx2]
        crops.append(cv2.resize(crop, (96, 48), interpolation=cv2.INTER_AREA))
    if len(crops) < 2:
        return 0.0, len(crops)
    stack = np.stack(crops).astype(np.float32)
    return float(np.mean(np.var(stack, axis=0))), len(crops)


def main():
    print("=" * 60)
    print("STATIC DISCRIMINATOR SWEEP — 42 GT V2.3 tracks")
    print(f"  Video:       {VIDEO_PATH.name}")
    print(f"  GT V2.3:     {GT_PATH.name}")
    print("=" * 60)

    entries = load_gt_entries()
    frames, fps = decode_frames()
    print(f"[DECODE] {len(frames)} video frames at {fps:.3f} fps")

    # Extract per-track variance
    per_track: dict[str, dict] = {}

    for entry in entries:
        eid = entry["id"]
        expected = entry["is_subtitle"]
        var, samples = extract_temporal_variance(entry, frames, fps)
        per_track[eid] = {
            "expected_is_subtitle": expected,
            "temporal_pixel_variance": round(var, 4),
            "sample_count": samples,
            "bbox_xyxy": entry["bbox_xyxy"],
            "timestamp_s": entry["timestamp_s"],
        }
        label = "POS" if expected else "NEG"
        print(f"  {eid:<20} ({label}) temporal_var={var:>10.2f}  samples={samples}")

    # Split into classes
    pos_var = [d["temporal_pixel_variance"] for d in per_track.values() if d["expected_is_subtitle"]]
    neg_var = [d["temporal_pixel_variance"] for d in per_track.values() if not d["expected_is_subtitle"]]

    print(f"\n{'='*60}")
    print("VARIANCE DISTRIBUTION")
    print(f"{'='*60}")
    print(f"  Positive (n={len(pos_var)}): min={np.min(pos_var):.2f}  mean={np.mean(pos_var):.2f}  max={np.max(pos_var):.2f}")
    print(f"  Negative (n={len(neg_var)}): min={np.min(neg_var):.2f}  mean={np.mean(neg_var):.2f}  max={np.max(neg_var):.2f}")

    # Sweep variance thresholds that keep 0 FN
    max_variance = max(pos_var + neg_var)
    thresholds = np.linspace(0.0, max_variance, 81).round(2).tolist()
    sweep_rows = []
    for th in thresholds:
        pos_kept = sum(1 for v in pos_var if v >= th)
        neg_rejected = sum(1 for v in neg_var if v < th)
        sweep_rows.append({
            "temporal_variance_threshold": th,
            "positives_kept": pos_kept,
            "positives_lost": len(pos_var) - pos_kept,
            "negatives_rejected": neg_rejected,
            "negatives_kept": len(neg_var) - neg_rejected,
        })

    # Find best: 0 FN + max negatives rejected
    best_th = None
    best_neg_rej = -1
    for r in sweep_rows:
        if r["positives_lost"] == 0 and r["negatives_rejected"] > best_neg_rej:
            best_neg_rej = r["negatives_rejected"]
            best_th = r["temporal_variance_threshold"]

    print(f"\nSWEEP TABLE (top 10 rows):")
    print(f"  {'Thresh':<8} {'Pos kept':<10} {'Pos lost':<10} {'Neg rej':<10} {'Neg kept':<10}")
    for r in sweep_rows:
        if r["temporal_variance_threshold"] == best_th or r["positives_lost"] == 0:
            marker = " <-- BEST" if r["temporal_variance_threshold"] == best_th else ""
            print(f"  {r['temporal_variance_threshold']:<8} {r['positives_kept']:<10} {r['positives_lost']:<10} {r['negatives_rejected']:<10} {r['negatives_kept']:<10}{marker}")

    print(f"\n{'='*60}")
    print("RECOMMENDATION")
    print(f"{'='*60}")
    if best_th:
        # Show the gap between positive min and negative max
        print(f"  Best variance threshold:   {best_th}")
        print(f"  Positives kept:            {sum(1 for v in pos_var if v >= best_th)}/{len(pos_var)}")
        print(f"  Negatives rejected:        {sum(1 for v in neg_var if v < best_th)}/{len(neg_var)}")
        print(f"  Positive variance range:   [{np.min(pos_var):.0f}, {np.max(pos_var):.0f}]")
        print(f"  Negative variance range:   [{np.min(neg_var):.0f}, {np.max(neg_var):.0f}]")
        print(f"  Overlap region:            [{max(np.min(pos_var), np.min(neg_var)):.0f}, {min(np.max(pos_var), np.max(neg_var)):.0f}]")

        # Per-track decision at best threshold
        confusion = {"tp": 0, "fn": 0, "tn": 0, "fp": 0}
        failed_pos = []
        for eid, d in sorted(per_track.items()):
            pred = d["temporal_pixel_variance"] >= best_th
            exp = d["expected_is_subtitle"]
            if pred and exp:
                confusion["tp"] += 1
            elif pred and not exp:
                confusion["fp"] += 1
            elif not pred and exp:
                confusion["fn"] += 1
                failed_pos.append(eid)
            else:
                confusion["tn"] += 1
        print(f"\n  Confusion matrix at th={best_th}:")
        print(f"    TP={confusion['tp']}  FP={confusion['fp']}")
        print(f"    FN={confusion['fn']}  TN={confusion['tn']}")
        if failed_pos:
            print(f"    FAILED POSITIVES: {failed_pos}")
    else:
        print("  [FAIL] No threshold with FN=0 found")

    # Output JSON
    output = {
        "video": str(VIDEO_PATH),
        "gt_source": str(GT_PATH),
        "per_track": per_track,
        "variance_distribution": {
            "positive": {"min": float(np.min(pos_var)), "mean": float(np.mean(pos_var)), "max": float(np.max(pos_var))},
            "negative": {"min": float(np.min(neg_var)), "mean": float(np.mean(neg_var)), "max": float(np.max(neg_var))},
        },
        "sweep_table": sweep_rows,
        "recommendation": {
            "temporal_variance_threshold": best_th,
            "positives_kept": sum(1 for v in pos_var if v >= best_th) if best_th else 0,
            "negatives_rejected": sum(1 for v in neg_var if v < best_th) if best_th else 0,
            "total_positives": len(pos_var),
            "total_negatives": len(neg_var),
            "confusion_matrix": confusion,
        },
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[OUTPUT] Wrote to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
