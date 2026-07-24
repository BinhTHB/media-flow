import json
from pathlib import Path
from src.infrastructure.benchmark.hybrid_subtitle.core import group_frame_candidates, build_tracks
from src.infrastructure.benchmark.hybrid_subtitle.subtitle_classifier import HeuristicSubtitleClassifier

def run_debug():
    raw_path = Path("artifacts/hybrid_subtitle_benchmark/sample5_raw_yolo_detections.jsonl")
    raw_dets = []
    with open(raw_path) as f:
        for line in f:
            raw_dets.append(json.loads(line))
            
    ref_path = Path("artifacts/hybrid_subtitle_benchmark/sample5_text_tracks.json")
    with open(ref_path) as f:
        ref_tracks = json.load(f)
        
    candidate_to_ref = {}
    for ref in ref_tracks:
        ref_id = ref["track_id"]
        for group in ref["groups"]:
            for cid in group["member_candidate_ids"]:
                candidate_to_ref[cid] = ref_id
                
    gt_path = Path("scripts/frozen_gt/sample5/frozen_gt.json")
    with open(gt_path) as f:
        gt_data = json.load(f)
    gt_labels = {t["id"]: t["expected_is_subtitle"] for t in gt_data}
    gt_sub_ids = {t["id"] for t in gt_data if t["expected_is_subtitle"]}
    
    thresh = 0.30
    frame_height = 1280
    
    filtered_dets = [d for d in raw_dets if d["confidence"] >= thresh]
    
    by_frame = {}
    for d in filtered_dets:
        by_frame.setdefault(d["frame_index"], []).append(d)
        
    flat_groups = []
    for fi in sorted(by_frame):
        groups = group_frame_candidates(by_frame[fi], horizontal_gap_ratio=2.0)
        flat_groups.extend(groups)
        
    tracks = build_tracks(flat_groups, iou_threshold=0.25, max_gap_frames=5)
    
    classifier = HeuristicSubtitleClassifier()
    matched_tracks = {}
    
    for t in tracks:
        votes = {}
        for group in t["groups"]:
            for cid in group["member_candidate_ids"]:
                ref_id = candidate_to_ref.get(cid)
                if ref_id:
                    votes[ref_id] = votes.get(ref_id, 0) + 1
        if not votes:
            continue
        matched_ref_id = max(votes, key=votes.get)
        matched_tracks.setdefault(matched_ref_id, []).append(t)

    print("=== DEBUG GT SUBTITLE TRACKS ===")
    for gt_id in sorted(gt_sub_ids):
        print(f"\nGT ID: {gt_id}")
        # Find matching rebuilt tracks
        rebuilt = matched_tracks.get(gt_id, [])
        if not rebuilt:
            print("  -> Status: LOST (No rebuilt track matched this GT ID)")
            # Check if any candidates survived at all
            survived_cids = []
            ref_track = next(r for r in ref_tracks if r["track_id"] == gt_id)
            all_cids = [cid for g in ref_track["groups"] for cid in g["member_candidate_ids"]]
            raw_matched = [d for d in raw_dets if d["candidate_id"] in all_cids]
            survived = [d for d in raw_matched if d["confidence"] >= thresh]
            print(f"  -> Raw candidate count: {len(all_cids)}, Survived >= 0.30: {len(survived)}")
            if survived:
                print(f"  -> Survived confidences: {[d['confidence'] for d in survived]}")
            continue
            
        print(f"  -> Matched {len(rebuilt)} rebuilt track(s)")
        for i, t in enumerate(rebuilt):
            res = classifier.classify(t, frame_height)
            print(f"    Rebuilt {i+1}: is_subtitle={res.is_subtitle}, reason={res.reason}")
            # print track stats
            bbox = (
                t.get("median_bbox_xyxy")
                or t.get("representative_bbox_xyxy")
                or t.get("temporal_union_bbox_xyxy")
                or []
            )
            _, y1, _, y2 = bbox
            bottom_y_ratio = y2 / frame_height
            groups = t.get("groups", [])
            duration_s = 0.0
            if groups:
                duration_s = (groups[-1]["timestamp_ms"] - groups[0]["timestamp_ms"]) / 1000.0
            print(f"      bbox_y2={y2} (ratio={bottom_y_ratio:.3f}), duration={duration_s:.2f}s, frame_count={len(groups)}")

if __name__ == "__main__":
    run_debug()
