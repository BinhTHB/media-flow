#!/usr/bin/env python3
"""Generate sweep parameter space for candidate fix proposals on V2.1.
Each fix is scored on: negatives_killed (more is better), positives_killed (0 is must), net_score.

Run: python scripts/sweep_step7.py
"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(".")
OUT = ROOT / "artifacts" / "round28"
GT_V23 = OUT / "gt_v2_3.json"

# Load GT V2.3
gt = json.loads(GT_V23.read_text(encoding="utf-8"))
positives = {e["id"] for e in gt["entries"] if e["is_subtitle"]}
negatives = {e["id"] for e in gt["entries"] if not e["is_subtitle"]}

# Load the candidate decisions from Gemini cache for all 8 FPs
# We need to know which 14 positives + which negatives to evaluate against
print(f"Positives ({len(positives)}): {sorted(positives)}")
print(f"Negatives ({len(negatives)}): {sorted(negatives)}")
print()

# The 8 FPs under V2.3: T030, T041, T044, T047, T055, T061, T071, T105
# All 8 confirmed by Gemini with is_subtitle=true but they are FP.

# Load the subtitle_regions.json data for these 8 tracks to get their metrics
regions = json.load((ROOT / "output" / "round27-sample5" / "debug" / "subtitle_regions.json").open(encoding="utf-8"))
fp_tracks = {t["track_id"]: t for t in regions if t["track_id"] in {"T030","T041","T044","T047","T055","T061","T071","T105"}}

print("=== FP Track Metrics ===")
for tid in sorted(fp_tracks):
    t = fp_tracks[tid]
    w = t["width"]
    h = t["height"]
    xy_ratio = w / h if h > 0 else 0
    cy = t["y"] + h / 2
    print(f"{tid}: x={round(t['x'],3)} y={round(t['y'],3)} w={round(w,3)} h={round(h,3)} aspect={round(xy_ratio,2)} cy_norm={round(cy,3)}")

print()

# Get same metrics for the 14 positive tracks
pos_tracks = {t["track_id"]: t for t in regions if t["track_id"] in positives}
print("=== Positive Track Metrics ===")
for tid in sorted(pos_tracks):
    t = pos_tracks[tid]
    w = t["width"]
    h = t["height"]
    xy_ratio = w / h if h > 0 else 0
    cy = t["y"] + h / 2
    print(f"{tid}: x={round(t['x'],3)} y={round(t['y'],3)} w={round(w,3)} h={round(h,3)} aspect={round(xy_ratio,2)} cy_norm={round(cy,3)}")

# Compute stats
print()
fp_aspects = [fp_tracks[t]["width"] / max(fp_tracks[t]["height"], 0.0001) for t in fp_tracks]
pos_aspects = [pos_tracks[t]["width"] / max(pos_tracks[t]["height"], 0.0001) for t in pos_tracks]

print(f"FP aspect ratios: {[round(a,2) for a in sorted(fp_aspects)]}")
print(f"Pos aspect ratios: {[round(a,2) for a in sorted(pos_aspects)]}")

# Check center_y
fp_cy = [fp_tracks[t]["y"] + fp_tracks[t]["height"]/2 for t in fp_tracks]
pos_cy = [pos_tracks[t]["y"] + pos_tracks[t]["height"]/2 for t in pos_tracks]
print(f"FP cy_norm: {[round(c,3) for c in sorted(fp_cy)]}")
print(f"Pos cy_norm: {[round(c,3) for c in sorted(pos_cy)]}")
