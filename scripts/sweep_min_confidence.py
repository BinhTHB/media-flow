import json
from pathlib import Path
from src.infrastructure.benchmark.hybrid_subtitle.core import group_frame_candidates, build_tracks
from src.infrastructure.benchmark.hybrid_subtitle.subtitle_classifier import HeuristicSubtitleClassifier

def run_sweep():
    # Load raw detections
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
    
    frame_height = 1280
    thresholds = [0.30, 0.35, 0.40, 0.45, 0.50]
    
    print("| min_conf | total | heur_pos | heur_neg | gt_sub_hit | gt_sub_miss | prec_gt_sub_hit | note |", flush=True)
    print("| --- | --- | --- | --- | --- | --- | --- | --- |", flush=True)
    
    for thresh in thresholds:
        # Filter raw detections BY MIN_CONFIDENCE (applied BEFORE grouping, like real pipeline)
        filtered_dets = [d for d in raw_dets if d["confidence"] >= thresh]
        
        # Group by frame
        by_frame = {}
        for d in filtered_dets:
            by_frame.setdefault(d["frame_index"], []).append(d)
            
        flat_groups = []
        for fi in sorted(by_frame):
            groups = group_frame_candidates(by_frame[fi], horizontal_gap_ratio=2.0)
            flat_groups.extend(groups)
            
        # Build tracks
        tracks = build_tracks(flat_groups, iou_threshold=0.25, max_gap_frames=5)
        
        classifier = HeuristicSubtitleClassifier()
        heur_pos = 0
        heur_neg = 0
        gt_sub_hit = set()
        
        for t in tracks:
            res = classifier.classify(t, frame_height)
            
            # Map rebuilt track to reference track via candidate ID voting
            votes = {}
            for group in t["groups"]:
                for cid in group["member_candidate_ids"]:
                    ref_id = candidate_to_ref.get(cid)
                    if ref_id:
                        votes[ref_id] = votes.get(ref_id, 0) + 1
            if not votes:
                continue
            matched_ref_id = max(votes, key=votes.get)
            is_gt_sub = gt_labels.get(matched_ref_id, False)
            
            if res.is_subtitle:
                heur_pos += 1
                if is_gt_sub:
                    gt_sub_hit.add(matched_ref_id)
            else:
                heur_neg += 1
                
        total_gt_sub = 19
        gt_sub_hit_n = len(gt_sub_hit)
        
        # Check precision: among heur_pos, how many are GT subtitle?
        # We need to count how many heur_pos tracks match GT subtitle vs GT non-sub
        heur_pos_precision = f"{gt_sub_hit_n}/19"
        
        # Note about which GT subs hit
        missed = [tid for tid in gt_labels if gt_labels[tid] and tid not in gt_sub_hit]
        note = f"missed: {missed}" if missed else "all GT sub hit"
        
        print(f"| {thresh:.2f} | {len(tracks)} | {heur_pos} | {heur_neg} | {gt_sub_hit_n} | {total_gt_sub - gt_sub_hit_n} | {heur_pos_precision} | {note} |", flush=True)

if __name__ == "__main__":
    run_sweep()
