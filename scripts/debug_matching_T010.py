"""Debug GT track matching for T010 and other subtitle tracks."""
import json
from pathlib import Path

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
    with open("artifacts/round23/baseline_tracks.json") as f:
        baseline_tracks = json.load(f)
    with open("scripts/frozen_gt/sample5/frozen_gt.json") as f:
        gt_data = json.load(f)
    with open("artifacts/round23/gt_track_timestamps.json") as f:
        gt_times = json.load(f)
    with open("artifacts/round23/gt_tracks_with_geometry.json") as f:
        gt_geom = json.load(f)
    gt_geom_dict = {t["track_id"]: t for t in gt_geom}

    print("Baseline tracks count:", len(baseline_tracks))
    
    # Target T010
    target_id = "T010"
    gt = next(t for t in gt_data if t["id"] == target_id)
    t_meta = gt_times[target_id]
    g_meta = gt_geom_dict[target_id]
    
    gt_start = t_meta["start_time_s"]
    gt_end = t_meta["end_time_s"]
    gt_bbox = g_meta["bbox_xyxy"]
    gt_duration = max(0.1, t_meta["duration_s"])
    
    print(f"\nTarget {target_id}: start={gt_start}s, end={gt_end}s, bbox={gt_bbox}, duration={gt_duration}s")
    
    # Find closest baseline tracks in time
    candidates = []
    for bt in baseline_tracks:
        bt_start = max(0.0, bt["start_ms"] / 1000.0 - 0.08)
        bt_end = bt["end_ms"] / 1000.0 + 0.08
        
        # Temporal overlap
        overlap_start = max(gt_start, bt_start)
        overlap_end = min(gt_end, bt_end)
        overlap = max(0.0, overlap_end - overlap_start)
        temporal_overlap = overlap / gt_duration
        
        if temporal_overlap > 0.0:
            candidates.append((bt, temporal_overlap))
            
    print(f"Found {len(candidates)} baseline tracks with non-zero temporal overlap:")
    for bt, temp_ol in candidates:
        bt_start = bt["start_ms"] / 1000.0
        bt_end = bt["end_ms"] / 1000.0
        base_bbox = bt["representative_bbox_xyxy"]
        iou = iou_xyxy(gt_bbox, base_bbox)
        print(f"  Base {bt['track_id']}: start={bt_start:.2f}s, end={bt_end:.2f}s, bbox={base_bbox}, temp_overlap={temp_ol:.2%}, IoU={iou:.2%}")

if __name__ == "__main__":
    main()
