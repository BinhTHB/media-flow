"""Map frozen-GT mask geometry to regenerated baseline tracks."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

OUT = Path("artifacts/round23")

def iou_xyxy(left, right):
    ax1, ay1, ax2, ay2 = left
    bx1, by1, bx2, by2 = right
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - intersection
    return intersection / union if union else 0.0

def shrink_mask(bbox, pad=8):
    x1, y1, x2, y2 = bbox
    return [x1 + pad, y1 + pad, x2 - pad, y2 - pad]

def main():
    frozen = {item["id"]: item for item in json.loads((Path("scripts/frozen_gt/sample5/frozen_gt.json")).read_text(encoding="utf-8"))}
    sidecar = json.loads((OUT / "gt_mask_geometry.json").read_text(encoding="utf-8"))
    tracks = json.loads((OUT / "baseline_tracks_conf_0.30.json").read_text(encoding="utf-8"))

    matches = []
    unmatched = []
    ambiguous = []
    for gt in sidecar["tracks"]:
        gt_id = gt["track_id"]
        gt_start, gt_end = gt["start_time_s"], gt["end_time_s"]
        gt_duration = max(gt_end - gt_start, 0.001)
        target_bbox = shrink_mask(gt["bbox_xyxy"])
        candidates = []
        for track in tracks:
            track_start = max(0.0, track["start_ms"] / 1000.0 - 0.08)
            track_end = track["end_ms"] / 1000.0 + 0.08
            overlap = max(0.0, min(gt_end, track_end) - max(gt_start, track_start))
            temporal_overlap = overlap / gt_duration
            if temporal_overlap < 0.5:
                continue
            aligned_ious = [
                iou_xyxy(target_bbox, group["union_bbox_xyxy"])
                for group in track["groups"]
                if gt_start <= group["timestamp_ms"] / 1000.0 <= gt_end
            ]
            if not aligned_ious:
                continue
            mean_iou = sum(aligned_ious) / len(aligned_ious)
            max_iou = max(aligned_ious)
            if max_iou >= 0.5:
                candidates.append({
                    "base_id": track["track_id"],
                    "temporal_overlap": temporal_overlap,
                    "mean_iou": mean_iou,
                    "max_iou": max_iou,
                })
        candidates.sort(key=lambda c: (c["max_iou"], c["temporal_overlap"]), reverse=True)
        if not candidates:
            unmatched.append({"gt_id": gt_id, "is_subtitle": frozen[gt_id]["expected_is_subtitle"], "gt_mask_bbox": gt["bbox_xyxy"], "unpadded_bbox": target_bbox})
            continue
        if len(candidates) > 1:
            ambiguous.append({"gt_id": gt_id, "candidates": candidates})
        winner = candidates[0]
        matches.append({
            "gt_id": gt_id,
            "base_id": winner["base_id"],
            "is_subtitle": frozen[gt_id]["expected_is_subtitle"],
            "mask_bbox_xyxy": gt["bbox_xyxy"],
            "unpadded_bbox_xyxy": target_bbox,
            "temporal_overlap": round(winner["temporal_overlap"], 4),
            "mean_iou": round(winner["mean_iou"], 4),
            "max_iou": round(winner["max_iou"], 4),
        })

    payload = {
        "frozen_gt_sha256": sidecar["frozen_gt_sha256"],
        "baseline_tracks_sha256": (OUT / "baseline_tracks_conf_0.30.sha256").read_text(encoding="ascii").strip(),
        "criteria": {
            "temporal_overlap_min": 0.5,
            "time_aligned_max_iou_min": 0.5,
            "mask_pad_removed_px": 8,
            "geometry_parser": sidecar["parser"],
        },
        "matched_count": len(matches),
        "unmatched_count": len(unmatched),
        "ambiguous_count": len(ambiguous),
        "matches": matches,
        "unmatched": unmatched,
        "ambiguous": ambiguous,
    }
    (OUT / "gt_track_mapping.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    sub_matches = [item for item in matches if item["is_subtitle"]]
    sub_missing = [item["gt_id"] for item in unmatched if item["is_subtitle"]]
    print(f"matched={len(matches)}/147 ambiguous={len(ambiguous)}")
    print(f"subtitles_matched={len(sub_matches)}/19 missing={sub_missing}")
    for item in sub_missing:
        print("missing", item)

if __name__ == "__main__":
    main()
