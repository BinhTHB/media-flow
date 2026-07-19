import argparse
import json
import os
import sys
from pathlib import Path

import cv2

REPO_ROOT = Path(__file__).resolve().parents[1]


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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--regions-path",
        default="output/hybrid_e2e/e2e_hybrid/debug/subtitle_regions.json",
    )
    parser.add_argument(
        "--gt-dir",
        default="test_videos/subtitle_replacement_benchmark/bottom_stable",
    )
    args = parser.parse_args()

    regions_path = REPO_ROOT / args.regions_path.replace("/", os.sep)
    gt_dir = REPO_ROOT / args.gt_dir.replace("/", os.sep)

    if not regions_path.exists():
        print(f"ERROR: regions file not found at {regions_path}")
        sys.exit(1)

    with open(regions_path, "r", encoding="utf-8") as f:
        regions = json.load(f)

    gt_path = gt_dir / "ground_truth.json"
    with open(gt_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    checkpoints = [float(x) for x in data["checkpoints"]]
    gt_per_ts = data["expected_source_subtitle_bbox_per_timestamp"]

    frame_0 = gt_dir / "frames_source" / "frame_000.png"
    img = cv2.imread(str(frame_0))
    H, W = img.shape[:2]

    # Evaluate
    hits = 0
    total = len(checkpoints)
    results = []

    for ts in checkpoints:
        raw = gt_per_ts[_gt_key(ts)]
        gt_norm = [raw["left"] / W, raw["top"] / H, raw["right"] / W, raw["bottom"] / H]
        active = [r for r in regions if r["start_time"] <= ts <= r["end_time"]]

        best_iou = 0.0
        for r in active:
            region_box = [r["x"], r["y"], r["x"] + r["width"], r["y"] + r["height"]]
            iou = iou_xyxy(gt_norm, region_box)
            best_iou = max(best_iou, iou)

        hit = best_iou >= 0.5
        if hit:
            hits += 1

        results.append({"timestamp": ts, "iou": round(best_iou, 4), "hit": hit})

    hit_rate = hits / total if total > 0 else 0.0
    mean_iou = sum(r["iou"] for r in results) / total if total > 0 else 0.0

    # Print results
    print("\n=== E2E Regions vs Ground Truth ===")
    print(f"Regions file: {regions_path}")
    print(f"GT dir: {gt_dir}")
    print(f"Total checkpoints: {total}")
    print(f"Hits (IoU >= 0.5): {hits} / {total}")
    print(f"Hit rate: {round(hit_rate, 4)}")
    print(f"Mean IoU: {round(mean_iou, 4)}")
    print("\nPer-checkpoint details:")
    print("ts | best_iou | hit")
    for r in results:
        print(f"{r['timestamp']} | {r['iou']} | {r['hit']}")

    # Write HYBRID_E2E_SMOKE.md
    doc_path = REPO_ROOT / "docs" / "HYBRID_E2E_SMOKE.md"
    lines = []
    lines.append("# Hybrid Subtitle Detector - E2E Smoke Test\n")
    lines.append("## CLI Command run")
    lines.append("```bash")
    lines.append(
        "python src/interfaces/cli/demo.py test_videos/subtitle_replacement_benchmark/bottom_stable/video.mp4 "
        "--subtitle --subtitle-placement --placement-mode replace_source_subtitle --video-provider ffmpeg "
        "--frame-provider ffmpeg --frame-interval 0.5 --subtitle-region-detector hybrid_yolo "
        "--output-dir output/hybrid_e2e --job-id e2e_hybrid"
    )
    lines.append("```\n")
    lines.append(f"Number of regions detected: {len(regions)}\n")
    lines.append("## Checkpoint IoU evaluation")
    lines.append("| timestamp | best_iou | hit |")
    lines.append("| --- | --- | --- |")
    for r in results:
        lines.append(f"| {r['timestamp']} | {r['iou']} | {'True' if r['hit'] else 'False'} |")
    lines.append("")
    lines.append(f"**Hits (IoU >= 0.5):** {hits} / {total}\n")
    lines.append(f"**Hit Rate:** {round(hit_rate, 4)}\n")
    lines.append(f"**Mean IoU:** {round(mean_iou, 4)}\n")
    lines.append("## Conclusion")
    if hit_rate >= 0.8:
        lines.append("The hybrid_yolo region detector passes the end-to-end integration smoke test.")
        lines.append("It successfully identifies the source subtitles at the bottom of the video frames.")
    else:
        lines.append("The hybrid_yolo region detector hit rate is below 80% on the e2e bottom_stable video.")
        lines.append("Further fine-tuning or live validation may be required.")
    lines.append("")

    doc_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nCreated smoke doc at: {doc_path}")


if __name__ == "__main__":
    main()
