"""Regenerate baseline tracks at min_confidence=0.30 and match to GT using sidecar timestamps."""
import json
import hashlib
import os
from pathlib import Path
from src.infrastructure.benchmark.hybrid_subtitle.core import group_frame_candidates, build_tracks

RAW_DET_PATH = Path("artifacts/hybrid_subtitle_benchmark/sample5_raw_yolo_detections.jsonl")
GT_PATH = Path("scripts/frozen_gt/sample5/frozen_gt.json")
TIMESTAMPS_PATH = Path("artifacts/round23/gt_track_timestamps.json")
GT_GEOM_PATH = Path("artifacts/round23/gt_tracks_with_geometry.json")
OUT_DIR = Path("artifacts/round23")
H, W = 1280, 720
MIN_CONF = 0.30

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

def load_raw_dets():
    dets = []
    with open(RAW_DET_PATH) as f:
        for line in f:
            dets.append(json.loads(line))
    return dets

def filter_and_track(dets, min_conf):
    filtered = [d for d in dets if d["confidence"] >= min_conf]
    by_frame = {}
    for d in filtered:
        by_frame.setdefault(d["frame_index"], []).append(d)
    flat_groups = []
    for fi in sorted(by_frame):
        groups = group_frame_candidates(by_frame[fi], horizontal_gap_ratio=2.0)
        flat_groups.extend(groups)
    tracks = build_tracks(flat_groups, iou_threshold=0.25, max_gap_frames=5)
    return tracks

def main():
    print("Loading raw detections...")
    raw_dets = load_raw_dets()
    print("Building baseline tracks...")
    baseline_tracks = filter_and_track(raw_dets, MIN_CONF)
    print(f"  Baseline tracks: {len(baseline_tracks)}")

    print("Loading GT data...")
    with open(GT_PATH) as f:
        gt_data = json.load(f)
    gt_labels = {t["id"]: t["expected_is_subtitle"] for t in gt_data}
    
    with open(TIMESTAMPS_PATH) as f:
        gt_times = json.load(f)
        
    with open(GT_GEOM_PATH) as f:
        gt_geom = json.load(f)
    gt_geom_dict = {t["track_id"]: t for t in gt_geom}

    # Match GT -> Baseline
    matched = []
    unmatched_gt = []

    for gt in gt_data:
        gt_id = gt["id"]
        is_sub = gt["expected_is_subtitle"]
        
        # Get original ASS time window and geometry
        t_meta = gt_times.get(gt_id)
        g_meta = gt_geom_dict.get(gt_id)
        if not t_meta or not g_meta:
            unmatched_gt.append(gt_id)
            continue
            
        gt_start = t_meta["start_time_s"]
        gt_end = t_meta["end_time_s"]
        gt_bbox = g_meta["bbox_xyxy"]
        gt_duration = max(0.1, t_meta["duration_s"])

        best_base = None
        best_score = -1.0

        for bt in baseline_tracks:
            # Baseline track time window: derived from frames
            # Frame cadence gap
            frame_gap = (bt["end_ms"] - bt["start_ms"]) / 1000.0 if len(bt["groups"]) >= 2 else 0.0
            
            # Apply padding (half frame gap)
            bt_start = max(0.0, bt["start_ms"] / 1000.0 - 0.08)
            bt_end = bt["end_ms"] / 1000.0 + 0.08
            
            # Temporal overlap
            overlap_start = max(gt_start, bt_start)
            overlap_end = min(gt_end, bt_end)
            overlap = max(0.0, overlap_end - overlap_start)
            temporal_overlap = overlap / gt_duration
            
            if temporal_overlap < 0.50:
                continue
                
            # Spatial overlap: IoU between GT bbox and baseline representative bbox
            # Rebuilt track representative_bbox_xyxy is in pixels.
            # GT bbox is also in pixels.
            base_bbox = bt["representative_bbox_xyxy"]
            iou = iou_xyxy(gt_bbox, base_bbox)
            
            if iou < 0.50:
                continue
                
            # Combined score
            score = temporal_overlap + iou
            if score > best_score:
                best_score = score
                best_base = bt["track_id"]

        if best_base:
            matched.append({
                "gt_id": gt_id,
                "base_id": best_base,
                "gt_is_subtitle": is_sub,
                "score": best_score,
            })
        else:
            unmatched_gt.append(gt_id)

    # Save mapping
    mapping_path = OUT_DIR / "gt_track_mapping.json"
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump({
            "matched_count": len(matched),
            "unmatched_gt_count": len(unmatched_gt),
            "unmatched_gt_ids": sorted(unmatched_gt),
            "matches": matched,
        }, f, indent=2)
        
    print(f"Matched GT tracks: {len(matched)} / 147")
    print(f"Unmatched GT: {unmatched_gt}")
    
    # Sanity check: 19/19 GT subtitles must match
    sub_matches = [m for m in matched if m["gt_is_subtitle"]]
    print(f"Sanity gate: {len(sub_matches)} / 19 GT subtitles matched.")
    
    # Dump missing subs
    sub_ids = {t["id"] for t in gt_data if t["expected_is_subtitle"]}
    matched_subs = {m["gt_id"] for m in sub_matches}
    missing_subs = sub_ids - matched_subs
    if missing_subs:
        print(f"  BLOCKED: Missing subtitles: {sorted(missing_subs)}")
    else:
        print("  PASS: all 19 GT subtitles matched successfully!")

if __name__ == "__main__":
    main()
