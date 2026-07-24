# Fix sweep_r1.py
content = '''\"\"\"STEP 2: R1 motion coupling parameter sweep.
Sweeps MIN_BG_SPEED_PX and K_CONSECUTIVE over the GT V2.2 positives.
Reports BLOCKED when video or tracks are unavailable.
Requires: sample5 video + round23 cached tracks for proposal_bundle.\"\"\"
from __future__ import annotations
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    from src.infrastructure.benchmark.hybrid_subtitle.r1_motion import R1MotionCoupling, R1Result
    R1_AVAILABLE = True
except ImportError:
    R1_AVAILABLE = False
VIDEO = Path(\"test_videos/real_mini_benchmark/sample5.mp4\")
OUT = Path(\"artifacts/round26\")
GT = OUT / \"gt_v2_2.json\"
TRACKS = Path(\"artifacts/round23/baseline_tracks_conf_0.30.json\")

# Sweep grid
MIN_BG_SPEED_GRID = [0.5, 1.0, 2.0, 3.0]
K_CONSECUTIVE_GRID = [1, 2, 3]

def main():
    if not R1_AVAILABLE:
        print(json.dumps({\"status\": \"BLOCKED\", \"reason\": \"R1MotionCoupling not importable\", \"result\": {\"rejected\": -1, \"total\": 19}}, indent=2))
        return
    if not VIDEO.exists():
        print(json.dumps({\"status\": \"BLOCKED\", \"reason\": \"Video not found at %s\" %% VIDEO, \"result\": {\"rejected\": -1, \"total\": 19}}, indent=2))
        return
    gt = json.loads(GT.read_text(encoding=\"utf-8\"))
    positives = [e for e in gt[\"entries\"] if e[\"is_subtitle\"]]
    try:
        tracks = json.loads(TRACKS.read_text(encoding=\"utf-8\"))
    except FileNotFoundError:
        print(json.dumps({\"status\": \"BLOCKED\", \"reason\": \"baseline_tracks_conf_0.30.json not found\", \"result\": {\"rejected\": -1, \"total\": len(positives)}}, indent=2))
        return
    # Map tracks by ID
    tracks_by_id = {t[\"track_id\"]: t for t in tracks}
    # Build track dicts for each GT entry (using matching pipeline tracks)
    track_dicts = {}
    import hashlib
    map_data_path = Path(\"artifacts/round23/gt_track_mapping.json\")
    if map_data_path.exists():
        map_data = json.loads(map_data_path.read_text(encoding=\"utf-8\"))
        for match in map_data.get(\"matches\", []):
            gt_id = match[\"gt_id\"]
            base_id = match[\"base_id\"]
            pt = tracks_by_id.get(base_id)
            if pt and gt_id in {e[\"id\"] for e in positives}:
                track_dicts[gt_id] = pt
    results = []
    for mp in MIN_BG_SPEED_GRID:
        for kc in K_CONSECUTIVE_GRID:
            r1 = R1MotionCoupling(frame_height=1280, frame_width=720)
            r1.MIN_BG_SPEED_PX = mp
            r1.K_CONSECUTIVE = kc
            rejected = 0
            total = len(positives)
            details = []
            for entry in positives:
                eid = entry[\"id\"]
                td = track_dicts.get(eid)
                if td is None:
                    continue
                # evaluate() needs (track, proposal_bundle).
                # Without frame cache, use a dummy bundle that returns None frames
                class _DummyBundle:
                    def frame_for_index(self, idx):
                        return None
                result = r1.evaluate(td, _DummyBundle())
                if result.rejected:
                    rejected += 1
                    details.append(eid)
            results.append({
                \"min_bg_speed\": mp,
                \"k_consecutive\": kc,
                \"rejected\": rejected,
                \"total\": total,
                \"rejected_ids\": details,
            })
            print(\"  mp=%%s kc=%%s: rejected=%%d/%%d\" %% (mp, kc, rejected, total))
    payload = {
        \"status\": \"COMPLETED\" if results else \"NO_DATA\",
        \"kind\": \"v2_2\",
        \"grid\": {\"MIN_BG_SPEED_PX\": MIN_BG_SPEED_GRID, \"K_CONSECUTIVE\": K_CONSECUTIVE_GRID},
        \"results\": results,
    }
    (OUT / \"sweep_r1.json\").write_text(json.dumps(payload, indent=2), encoding=\"utf-8\")
    print(json.dumps(payload, indent=2))
if __name__ == \"__main__\":
    main()
'''
with open('scripts/sweep_r1.py', 'w', encoding='utf-8') as f:
    f.write(content)
print('sweep_r1.py rewritten')
