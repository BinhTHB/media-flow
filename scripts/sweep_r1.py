"""STEP 2: Fast optimized downscaled R1 motion coupling parameter sweep.
Downscales frames by 4x for 16x speedup on Farneback flow calculation.
"""
import json
import sys
import numpy as np
from pathlib import Path
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.infrastructure.benchmark.hybrid_subtitle.r1_motion import R1MotionCoupling
from src.application.dto.proposal_bundle import ProposalBundle

VIDEO = Path("test_videos/real_mini_benchmark/sample5.mp4")
OUT = Path("artifacts/round26")
GT = OUT / "gt_v2_2.json"
TRACKS = Path("artifacts/round23/baseline_tracks_conf_0.30.json")
MAP_PATH = Path("artifacts/round23/gt_track_mapping.json")

MIN_BG_SPEED_GRID = [0.5, 1.0, 2.0, 3.0]
K_CONSECUTIVE_GRID = [1, 2, 3]


def main():
    if not VIDEO.exists():
        print(json.dumps({"status": "BLOCKED", "reason": "Video not found"}, indent=2))
        return
    if not GT.exists():
        print(json.dumps({"status": "BLOCKED", "reason": "GT not found"}, indent=2))
        return
    if not TRACKS.exists():
        print(json.dumps({"status": "BLOCKED", "reason": "Tracks file not found"}, indent=2))
        return

    gt = json.loads(GT.read_text(encoding="utf-8"))
    positives = [e for e in gt["entries"] if e["is_subtitle"]]
    tracks = json.loads(TRACKS.read_text(encoding="utf-8"))
    tracks_by_id = {t["track_id"]: t for t in tracks}

    track_dicts = {}
    if MAP_PATH.exists():
        map_data = json.loads(MAP_PATH.read_text(encoding="utf-8"))
        for match in map_data.get("matches", []):
            gt_id = match["gt_id"]
            base_id = match["base_id"]
            pt = tracks_by_id.get(base_id)
            if pt and gt_id in {e["id"] for e in positives}:
                track_dicts[gt_id] = pt

    if not track_dicts:
        print(json.dumps({"status": "COMPLETED", "results": []}, indent=2))
        return

    needed_frames = set()
    max_pairs = max(K_CONSECUTIVE_GRID) + 2  # 5 pair max per track for K_CONSECUTIVE=3
    for pt in track_dicts.values():
        groups = pt.get("groups", [])
        for i in range(min(len(groups), max_pairs + 1)):
            needed_frames.add(int(groups[i]["frame_index"]))

    cap = cv2.VideoCapture(str(VIDEO))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    bundle = ProposalBundle(str(VIDEO), fps, 1, width, height, total_frames)
    fidx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if fidx in needed_frames:
            bundle.add_frame(frame_index=fidx, timestamp_sec=fidx/fps, image=frame, image_path=None, detections=[])
        fidx += 1
    cap.release()
    print("Decoded %d frames." % len(bundle.frames))

    track_motion_data = {}
    r1_evaluator = R1MotionCoupling(frame_height=height, frame_width=width)

    for eid, td in track_dicts.items():
        groups = td.get("groups", [])
        pairs = []
        for i in range(min(len(groups) - 1, max_pairs)):
            g0 = groups[i]
            g1 = groups[i+1]
            fi0 = g0["frame_index"]
            fi1 = g1["frame_index"]
            try:
                frame0 = bundle.frame_for_index(fi0)
                frame1 = bundle.frame_for_index(fi1)
            except KeyError:
                continue
            img0 = frame0.image
            img1 = frame1.image
            if img0 is None or img1 is None:
                continue
            
            gray0 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY) if len(img0.shape) == 3 else img0
            gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) if len(img1.shape) == 3 else img1

            bbox0 = g0["union_bbox_xyxy"]
            mask = np.ones_like(gray0, dtype=np.uint8) * 255
            x1 = max(0, int(round(bbox0[0])) - r1_evaluator.TEXT_MASK_MARGIN)
            y1 = max(0, int(round(bbox0[1])) - r1_evaluator.TEXT_MASK_MARGIN)
            x2 = min(gray0.shape[1], int(round(bbox0[2])) + r1_evaluator.TEXT_MASK_MARGIN)
            y2 = min(gray0.shape[0], int(round(bbox0[3])) + r1_evaluator.TEXT_MASK_MARGIN)
            mask[y1:y2, x1:x2] = 0

            # Resize 4x down for flow computation
            small_gray0 = cv2.resize(gray0, (0, 0), fx=0.25, fy=0.25)
            small_gray1 = cv2.resize(gray1, (0, 0), fx=0.25, fy=0.25)
            small_mask = cv2.resize(mask, (0,0), fx=0.25, fy=0.25, interpolation=cv2.INTER_NEAREST)

            visible = small_mask > 0
            if visible.sum() < 20:  # Adjust threshold for smaller resolution
                continue
            try:
                # Dense Farneback flow on smaller resolution
                flow = cv2.calcOpticalFlowFarneback(small_gray0, small_gray1, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                y_idxs, x_idxs = np.where(visible)
                pts_src = np.column_stack([x_idxs, y_idxs]).astype(np.float32)
                flow_vectors = flow[y_idxs, x_idxs]
                pts_dst = pts_src + flow_vectors
                _, inliers = cv2.estimateAffinePartial2D(pts_src, pts_dst, method=cv2.RANSAC, ransacReprojThreshold=3.0, maxIters=1000, confidence=0.99)
                if inliers is not None and inliers.sum() >= 5:
                    bg_vx = float(np.median(flow_vectors[inliers.ravel() > 0, 0])) * 4.0
                    bg_vy = float(np.median(flow_vectors[inliers.ravel() > 0, 1])) * 4.0
                else:
                    bg_vx, bg_vy = 0.0, 0.0
            except Exception as e:
                print("Flow estimation error: ", e)
                bg_vx, bg_vy = 0.0, 0.0

            bg_speed = np.sqrt(bg_vx**2 + bg_vy**2)

            bbox1 = g1["union_bbox_xyxy"]
            c0_x = (bbox0[0] + bbox0[2]) / 2.0
            c0_y = (bbox0[1] + bbox0[3]) / 2.0
            c1_x = (bbox1[0] + bbox1[2]) / 2.0
            c1_y = (bbox1[1] + bbox1[3]) / 2.0
            track_speed = np.sqrt((c1_x - c0_x)**2 + (c1_y - c0_y)**2)

            track_angle = np.degrees(np.arctan2(c1_y - c0_y, c1_x - c0_x))
            bg_angle = np.degrees(np.arctan2(bg_vy, bg_vx))
            angle_diff = abs(track_angle - bg_angle)
            angle_diff = min(angle_diff, 360 - angle_diff)
            speed_ratio = track_speed / bg_speed if bg_speed > 0 else 0.0

            # Let's print flow results for diagnostics
            # print("eid=%s, bg_speed=%.2f, track_speed=%.2f, ratio=%.2f, angle_diff=%.1f" % (eid, bg_speed, track_speed, speed_ratio, angle_diff))

            pairs.append({
                "bg_speed": bg_speed,
                "track_speed": track_speed,
                "angle_diff": angle_diff,
                "speed_ratio": speed_ratio
            })
        track_motion_data[eid] = pairs
    print("Precomputed motion for %d tracks." % len(track_motion_data))

    results = []
    for mp in MIN_BG_SPEED_GRID:
        for kc in K_CONSECUTIVE_GRID:
            rejected = 0
            details = []
            for eid, pairs in track_motion_data.items():
                if len(pairs) < kc:
                    continue
                bg_speeds = [p["bg_speed"] for p in pairs[:kc]]
                avg_bg_speed = float(np.mean(bg_speeds)) if bg_speeds else 0.0

                if avg_bg_speed < mp:
                    continue

                coupled_count = 0
                frame_pairs_verified = 0
                for p in pairs[:kc]:
                    if p["bg_speed"] < mp:
                        continue
                    is_coupled = (
                        p["angle_diff"] <= r1_evaluator.MATCH_ANGLE_TOLERANCE_DEG
                        and r1_evaluator.MATCH_SPEED_RATIO[0] <= p["speed_ratio"] <= r1_evaluator.MATCH_SPEED_RATIO[1]
                    )
                    if is_coupled:
                        coupled_count += 1
                    frame_pairs_verified += 1

                if frame_pairs_verified >= kc and coupled_count >= kc:
                    rejected += 1
                    details.append(eid)

            results.append({
                "min_bg_speed": mp,
                "k_consecutive": kc,
                "rejected": rejected,
                "total": len(positives),
                "rejected_ids": details,
            })
            print("  mp=%.1f kc=%d: rejected=%d/%d" % (mp, kc, rejected, len(positives)))

    payload = {
        "status": "COMPLETED",
        "kind": "v2_2",
        "grid": {"MIN_BG_SPEED_PX": MIN_BG_SPEED_GRID, "K_CONSECUTIVE": K_CONSECUTIVE_GRID},
        "results": results,
    }
    (OUT / "sweep_r1.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (OUT / "r1_motion_sweep.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
