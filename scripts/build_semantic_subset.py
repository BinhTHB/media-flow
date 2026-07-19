"""Create an independently reviewable semantic evaluation subset from local tracks."""
from __future__ import annotations
import json
from pathlib import Path
from collections import defaultdict
import cv2
import numpy as np

ROOT = Path("artifacts/hybrid_subtitle_benchmark")
OUT = ROOT / "semantic_subset"
SAMPLES = [f"sample{i}" for i in range(1, 6)]


def active_tracks(tracks, frame):
    return [t for t in tracks if t["start_frame"] <= frame <= t["end_frame"]]


def pick_tracks(tracks, height):
    """Pick 20 local-only stratified scenes. Labels remain pending human review."""
    used, rows = set(), []
    bottom = lambda t: t["median_bbox_xyxy"][1] >= height * .60
    duration = lambda t: t["frame_count"]
    confidence = lambda t: t["mean_detector_confidence"]
    # Selection describes local signals only, never semantic labels.
    pools = [
        ("clear_bottom_long_track", sorted([t for t in tracks if bottom(t) and duration(t) >= 15], key=confidence, reverse=True), 5),
        ("bottom_track_with_concurrent_text", sorted([t for t in tracks if bottom(t)], key=lambda t: (duration(t), -confidence(t)), reverse=True), 5),
        ("non_bottom_text_dense_scene", sorted([t for t in tracks if not bottom(t)], key=duration, reverse=True), 3),
        ("low_confidence_or_motion_candidate", sorted(tracks, key=confidence), 3),
        ("transition_boundary", sorted(tracks, key=lambda t: t["start_frame"]), 2),
        ("fragmented_or_short_candidate", sorted(tracks, key=duration), 2),
    ]
    for reason, pool, count in pools:
        added = 0
        for track in pool:
            if track["track_id"] in used:
                continue
            used.add(track["track_id"])
            rows.append((reason, track))
            added += 1
            if added == count:
                break
    # deterministic completion if a category cannot fill
    for track in tracks:
        if len(rows) >= 20:
            break
        if track["track_id"] not in used:
            used.add(track["track_id"])
            rows.append(("local_stratified_completion", track))
    return rows[:20]


def label_frame(frame, candidates):
    image = frame.copy()
    for cid, track in candidates:
        x1,y1,x2,y2 = [int(v) for v in track["median_bbox_xyxy"]]
        cv2.rectangle(image, (x1,y1), (x2,y2), (0,255,255), 2)
        tx, ty = x1, y1 - 8 if y1 > 25 else y2 + 18
        cv2.rectangle(image, (tx, ty-15), (tx+34, ty+3), (0,255,255), -1)
        cv2.putText(image, cid, (tx+2,ty-2), cv2.FONT_HERSHEY_SIMPLEX, .45, (0,0,0), 1, cv2.LINE_AA)
    return image


def main():
    (OUT / "review_frames").mkdir(parents=True, exist_ok=True)
    (OUT / "candidate_crops").mkdir(parents=True, exist_ok=True)
    (OUT / "review_contact_sheets").mkdir(parents=True, exist_ok=True)
    manifest, candidates = [], []
    for sample in SAMPLES:
        tracks = json.loads((ROOT / f"{sample}_text_tracks.json").read_text(encoding="utf-8"))
        video = ROOT / "cache" / f"{sample}_h264.mp4"
        cap = cv2.VideoCapture(str(video))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        sheet = []
        for number, (reason, primary) in enumerate(pick_tracks(tracks, height), 1):
            scene_id = f"{sample}_scene_{number:03d}"
            mid = (primary["start_frame"] + primary["end_frame"]) // 2
            frame_ids = sorted({primary["start_frame"], mid, primary["end_frame"]})
            concurrent = [t for t in active_tracks(tracks, mid) if t["track_id"] != primary["track_id"]]
            selected = [primary] + sorted(concurrent, key=lambda t: t["mean_detector_confidence"], reverse=True)[:4]
            local = [(f"C{i+1}", track) for i, track in enumerate(selected)]
            frame_paths, crop_paths = [], []
            for frame_index in frame_ids:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index); ok, frame = cap.read()
                if not ok: continue
                labelled = label_frame(frame, local)
                path = OUT / "review_frames" / f"{scene_id}_F{frame_index:05d}.jpg"
                cv2.imwrite(str(path), labelled, [cv2.IMWRITE_JPEG_QUALITY, 88])
                frame_paths.append(str(path).replace('\\','/'))
                if frame_index == mid:
                    sheet.append(labelled)
            for cid, track in local:
                x1,y1,x2,y2 = [int(v) for v in track["median_bbox_xyxy"]]
                cap.set(cv2.CAP_PROP_POS_FRAMES, mid); ok, frame = cap.read()
                if not ok: continue
                margin = 12
                crop = frame[max(0,y1-margin):min(height,y2+margin), max(0,x1-margin):min(width,x2+margin)]
                crop_path = OUT / "candidate_crops" / f"{scene_id}_{cid}.jpg"
                cv2.imwrite(str(crop_path), crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
                crop_paths.append(str(crop_path).replace('\\','/'))
                candidates.append({"scene_id": scene_id, "candidate_id": cid, "track_id": track["track_id"], "bbox_xyxy": track["median_bbox_xyxy"], "selection_reason": reason, "ground_truth_status": "pending_independent_visual_review"})
            manifest.append({"scene_id": scene_id, "video_id": sample, "frame_indices": frame_ids, "timestamps_ms": [round(i/fps*1000,1) for i in frame_ids], "candidate_ids": [x[0] for x in local], "track_ids": [x[1]["track_id"] for x in local], "selection_reason": reason, "full_frame_paths": frame_paths, "candidate_crop_paths": crop_paths, "ground_truth_status": "pending_independent_visual_review"})
        cap.release()
        # compact contact sheet, review source only
        if sheet:
            thumbs=[cv2.resize(x,(180,320)) for x in sheet]
            rows=[]
            for i in range(0,len(thumbs),5):
                row=thumbs[i:i+5]
                row += [np.zeros_like(thumbs[0])]*(5-len(row))
                rows.append(np.hstack(row))
            cv2.imwrite(str(OUT / "review_contact_sheets" / f"{sample}_semantic_review_contact_sheet.jpg"), np.vstack(rows))
    (OUT / "semantic_subset_manifest.json").write_text(json.dumps({"selection_method":"local-only stratification; no semantic model decisions used", "scene_count":len(manifest), "scenes":manifest}, indent=2), encoding="utf-8")
    with (OUT / "semantic_subset_candidates.jsonl").open("w", encoding="utf-8") as handle:
        for item in candidates: handle.write(json.dumps(item)+"\n")
    print(json.dumps({"scenes":len(manifest), "candidates":len(candidates), "output":str(OUT)}))

if __name__ == "__main__": main()
