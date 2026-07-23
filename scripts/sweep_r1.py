"""STEP 2: R1 motion coupling sweep.
Requires: sample5 video + round23 cached detection data.
Run after STEP 0 caches crops (CODE-ONLY without E2E run, reports DATA_NEEDED)."""
from __future__ import annotations
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    from src.infrastructure.benchmark.hybrid_subtitle.r1_motion import R1MotionCoupling
    R1_AVAILABLE = True
except ImportError:
    R1_AVAILABLE = False
VIDEO = Path("test_videos/real_mini_benchmark/sample5.mp4")
OUT = Path("artifacts/round26")
GT = OUT / "gt_v2_2.json"
TRACKS = Path("artifacts/round23/baseline_tracks.json")
def main():
    if not R1_AVAILABLE:
        print(json.dumps({"status": "BLOCKED", "reason": "R1MotionCoupling not importable", "result": {"rejected": -1, "total": 19}}, indent=2))
        return
    if not VIDEO.exists():
        print(json.dumps({"status": "BLOCKED", "reason": f"Video not found at {VIDEO}", "result": {"rejected": -1, "total": 19}}, indent=2))
        return
    gt = json.loads(GT.read_text(encoding="utf-8"))
    positives = [e for e in gt["entries"] if e["is_subtitle"]]
    try:
        tracks = json.loads(TRACKS.read_text(encoding="utf-8"))
    except FileNotFoundError:
        print(json.dumps({"status": "BLOCKED", "reason": "baseline_tracks.json not found", "result": {"rejected": -1, "total": len(positives)}}, indent=2))
        return
    r1 = R1MotionCoupling(video_width=720, video_height=1280)
    total = len(positives)
    rejected = 0
    for entry in positives:
        bbox = entry["bbox_xyxy"]
        ts_ms = entry["timestamp_s"] * 1000.0
        track_id = entry["id"]
        related = [t for t in tracks if t.get("track_id") == track_id]
        result = r1.evaluate(bbox, ts_ms, related)
        if result.reject_subtitle:
            rejected += 1
    payload = {"status": "COMPLETED" if total > 0 else "NO_DATA", "kind": "v2_2", "result": {"rejected": rejected, "total": total, "pass": rejected == 0}}
    (OUT / "sweep_r1.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
if __name__ == "__main__":
    main()
