
"""Hybrid subtitle detector GT benchmark.

Chay HybridSubtitleRegionDetectorProvider tren cac frame co san trong
test_videos/subtitle_replacement_benchmark/<set>/frames_source/ va doi chieu
voi ground truth trong ground_truth.json. KHONG dung evaluate_tracks_vs_gt,
chi tinh IoU tai tung checkpoint.
"""

import json
import os
import sys
from pathlib import Path

import cv2

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.domain.entities.frame import Frame
from src.infrastructure.providers.subtitle.hybrid_subtitle_region_detector_provider import (
    HybridSubtitleRegionDetectorProvider,
)
from src.infrastructure.benchmark.hybrid_subtitle.subtitle_classifier import (
    HybridSubtitleClassifier,
)

SETS = ["bottom_stable", "moving"]
GT_DIR = REPO_ROOT / "test_videos" / "subtitle_replacement_benchmark"


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


def run_set(set_name: str):
    base = GT_DIR / set_name
    gt_path = base / "ground_truth.json"
    frames_dir = base / "frames_source"
    with open(gt_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    checkpoints = [float(x) for x in data["checkpoints"]]
    gt_per_ts = data["expected_source_subtitle_bbox_per_timestamp"]

    frame_files = sorted(frames_dir.glob("*.png"))
    if len(frame_files) != len(checkpoints):
        print(f"[DEFERRED] set={set_name}: so frame ({len(frame_files)}) != so checkpoint ({len(checkpoints)})")
        return None

    frames = [
        Frame(timestamp=checkpoints[i], image_path=str(frame_files[i]))
        for i in range(len(checkpoints))
    ]

    img = cv2.imread(str(frame_files[0]))
    H, W = img.shape[:2]
    provider = HybridSubtitleRegionDetectorProvider(
        classifier=HybridSubtitleClassifier(frame_height=H)
    )
    regions = provider.detect_regions(frames)

    hits = 0
    misses = 0
    ious = []
    for ts in checkpoints:
        raw = gt_per_ts[_gt_key(ts)]
        gt_norm = [raw["left"] / W, raw["top"] / H, raw["right"] / W, raw["bottom"] / H]
        active = [r for r in regions if r.start_time <= ts <= r.end_time]
        best_iou = 0.0
        for r in active:
            region_box = [r.x, r.y, r.x + r.width, r.y + r.height]
            iou = iou_xyxy(gt_norm, region_box)
            best_iou = max(best_iou, iou)
        ious.append(best_iou)
        if best_iou >= 0.5:
            hits += 1
        else:
            misses += 1

    hit_rate = hits / (hits + misses) if (hits + misses) else 0.0
    mean_iou = sum(ious) / len(ious) if ious else 0.0
    return {
        "set": set_name,
        "checkpoints": len(checkpoints),
        "hits": hits,
        "miss": misses,
        "hit_rate": round(hit_rate, 4),
        "mean_iou": round(mean_iou, 4),
    }


def main():
    weights_path = REPO_ROOT / "models" / "yolo" / "best.pt"
    if not weights_path.exists():
        print(f"ERROR: YOLO weights khong ton tai tai {weights_path}. Thoat.")
        sys.exit(1)

    results = []
    for set_name in SETS:
        res = run_set(set_name)
        if res is not None:
            results.append(res)

    # print console
    print("\n=== Hybrid Subtitle GT Benchmark ===")
    for res in results:
        print(f"set={res['set']} checkpoints={res['checkpoints']} "
              f"hits={res['hits']} miss={res['miss']} "
              f"hit_rate={res['hit_rate']} mean_iou={res['mean_iou']}")

    # write doc
    doc_path = REPO_ROOT / "docs" / "HYBRID_SUBTITLE_GT_BENCHMARK.md"
    lines = []
    lines.append("# Hybrid Subtitle Detector - Ground Truth Benchmark\n")
    lines.append("Chay tren cac frame co san trong `test_videos/subtitle_replacement_benchmark/`.")
    lines.append("Doi chieu region du doan voi ground truth theo IoU >= 0.5 tai tung checkpoint.\n")
    lines.append("| set | checkpoints | hits | miss | hit_rate | mean_iou |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for res in results:
        lines.append(
            f"| {res['set']} | {res['checkpoints']} | {res['hits']} | "
            f"{res['miss']} | {res['hit_rate']} | {res['mean_iou']} |"
        )
    lines.append("")
    lines.append("## Ghi chu")
    lines.append("- Set `moving` dung median box ca track nen IoU thap la gioi han da biet cua heuristic offline.")
    lines.append("- Danh gia nghiem tuc hon can Gemini layer (live verification).")
    lines.append("")
    doc_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nDa ghi: {doc_path}")


if __name__ == "__main__":
    main()

