#!/usr/bin/env python
"""Benchmark HybridSubtitleRegionDetectorProvider vs GT on 'moving' set,
with live GeminiTrackVerifier enabled.

Usage:
    .venv/Scripts/python scripts/benchmark_verifier_live_moving.py [--output-dir PATH]
"""
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.domain.entities.frame import Frame
from src.infrastructure.benchmark.hybrid_subtitle.gemini_verifier import GeminiTrackVerifier
from src.infrastructure.providers.subtitle.hybrid_subtitle_region_detector_provider import (
    HybridSubtitleRegionDetectorProvider,
)
from src.infrastructure.gemini.runtime import GeminiRuntime, RuntimeConfig


SET_NAME = "moving"
GT_DIR = REPO_ROOT / "test_videos" / "subtitle_replacement_benchmark" / SET_NAME
BEST_PT = REPO_ROOT / "best.pt"


def _gt_key(ts: float) -> str:
    if float(ts) == int(ts):
        return str(int(ts))
    return str(ts)


def iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def main():
    import cv2

    # Load GT
    with open(GT_DIR / "ground_truth.json") as f:
        data = json.load(f)
    checkpoints = [float(x) for x in data["checkpoints"]]
    gt_per_ts = data["expected_source_subtitle_bbox_per_timestamp"]

    frames_dir = GT_DIR / "frames_source"
    frame_files = sorted(frames_dir.glob("*.png"))
    if len(frame_files) != len(checkpoints):
        print(f"Frame count mismatch: {len(frame_files)} vs {len(checkpoints)}")
        sys.exit(1)

    frames = [
        Frame(timestamp=checkpoints[i], image_path=str(frame_files[i]))
        for i in range(len(checkpoints))
    ]

    # Build runtime and verifier
    api_key = os.environ.get("GEMINI_API_KEY", None)
    if not api_key:
        # Try config.yml
        try:
            from src.interfaces.config.config_loader import ConfigLoader
            loader = ConfigLoader()
            loader.load()
            api_key = loader.gemini_api_key
        except Exception:
            pass
    if not api_key:
        print("ERROR: GEMINI_API_KEY not found. Cannot run verifier-live benchmark.")
        sys.exit(1)

    runtime = GeminiRuntime.create(
        RuntimeConfig(api_key=api_key, cache_enabled=True)
    )
    verifier = GeminiTrackVerifier(runtime.get_client(model="gemini-3.1-flash-lite"), model="gemini-3.1-flash-lite")

    provider = HybridSubtitleRegionDetectorProvider(
        weights=str(BEST_PT),
        semantic_verifier=verifier,
        min_confidence=0.30,
        device="cpu",
    )

    regions = provider.detect_regions(frames)

    img = cv2.imread(str(frame_files[0]))
    H, W = img.shape[:2]

    hits = 0
    misses = 0
    ious = []
    for ts in checkpoints:
        raw = gt_per_ts.get(_gt_key(ts), {})
        if not raw:
            continue
        gt_norm = [raw["left"] / W, raw["top"] / H, raw["right"] / W, raw["bottom"] / H]
        active = [r for r in regions if r.start_time <= ts <= r.end_time]
        best_iou = 0.0
        for r in active:
            region_box = [r.x, r.y, r.x + r.width, r.y + r.height]
            best_iou = max(best_iou, iou_xyxy(gt_norm, region_box))
        ious.append(best_iou)
        if best_iou >= 0.5:
            hits += 1
        else:
            misses += 1

    hit_rate = hits / (hits + misses) if (hits + misses) else 0.0
    mean_iou = sum(ious) / len(ious) if ious else 0.0

    print(f"Verifier-live benchmark on '{SET_NAME}':")
    print(f"  checkpoints: {len(checkpoints)}")
    print(f"  detector regions returned: {len(regions)}")
    print(f"  hits: {hits}, misses: {misses}")
    print(f"  hit_rate: {hit_rate:.4f}")
    print(f"  mean_iou: {mean_iou:.4f}")
    print(f"  verifier counters:")
    print(f"    tracks_received={verifier.tracks_received}")
    print(f"    requests_sent={verifier.requests_sent}")
    print(f"    cache_hits={verifier.cache_hits}")
    print(f"    responses_parsed_ok={verifier.responses_parsed_ok}")
    print(f"    parse_failures={verifier.parse_failures}")
    print(f"    fail_open_count={verifier.fail_open_count}")


if __name__ == "__main__":
    main()
