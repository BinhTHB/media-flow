"""STEP 4: R3 static discriminator sweep.
Sweeps MIN_DURATION_SECONDS and STATIC_THRESHOLD_PX over GT V2.2.
Requires: video frames + scene-cut data. Reports BLOCKED without frames."""
from __future__ import annotations
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    from src.infrastructure.benchmark.hybrid_subtitle.scene_cut_detector import SceneCutDetector
    from src.infrastructure.benchmark.hybrid_subtitle.behavior_discriminator import R3StaticDiscriminator
    R3_AVAILABLE = True
except ImportError:
    R3_AVAILABLE = False
OUT = Path("artifacts/round26")
GT = OUT / "gt_v2_2.json"
FRAMES = Path("output/round25-sample5/frames")
MIN_DUR_GRID = [0.5, 1.0, 2.0]
STATIC_THRESH_GRID = [3.0, 5.0, 10.0]

def main():
    if not R3_AVAILABLE:
        out = {"status": "BLOCKED", "reason": "R3 components not importable", "result": {"rejected": -1, "total": 19}}
        print(json.dumps(out, indent=2))
        (OUT / "sweep_r3.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
        return
    if not FRAMES.exists():
        out = {"status": "BLOCKED", "reason": "Frames dir not found at %s" % FRAMES, "result": {"rejected": -1, "total": 19}}
        print(json.dumps(out, indent=2))
        (OUT / "sweep_r3.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
        return
    gt = json.loads(GT.read_text(encoding="utf-8"))
    positives = [e for e in gt["entries"] if e["is_subtitle"]]
    negatives = [e for e in gt["entries"] if not e["is_subtitle"]]
    entries_total = len(positives) + len(negatives)
    results = []
    for md in MIN_DUR_GRID:
        for st in STATIC_THRESH_GRID:
            scene = SceneCutDetector()
            r3 = R3StaticDiscriminator(scene)
            r3.MIN_DURATION_SECONDS = md
            r3.STATIC_THRESHOLD_PX = st
            evaluated = 0
            errors = 0
            pos_rejected = 0
            neg_rejected = 0
            for entry in positives + negatives:
                is_positive = entry in positives
                # Build minimal track dict needed by R3 evaluate
                groups = [{
                    "frame_index": 0,
                    "timestamp_ms": int(entry["timestamp_s"] * 1000),
                    "union_bbox_xyxy": entry.get("bbox_xyxy", [0, 0, 0, 0]),
                    "member_candidate_ids": [],
                }]
                track = {
                    "track_id": entry["id"],
                    "groups": groups,
                    "median_bbox_xyxy": entry.get("bbox_xyxy", [0, 0, 0, 0]),
                }
                try:
                    result = r3.evaluate(track, None, [])
                    evaluated += 1
                except Exception as exc:
                    errors += 1
                    print("  ERROR: md=%s st=%s id=%s %s" % (md, st, entry["id"], exc))
                    continue
                if result.rule_fired:
                    if is_positive:
                        pos_rejected += 1
                    else:
                        neg_rejected += 1
            status = "BLOCKED" if errors > 0 else ("COMPLETED" if evaluated == entries_total else "PARTIAL")
            results.append({
                "min_duration": md,
                "static_threshold": st,
                "positives_rejected": pos_rejected,
                "total_positives": len(positives),
                "negatives_rejected": neg_rejected,
                "total_negatives": len(negatives),
                "evaluated": evaluated,
                "errors": errors,
            })
            print("  md=%s st=%s: pos_rej=%d/%d neg_rej=%d/%d (eval=%d err=%d)" % (md, st, pos_rejected, len(positives), neg_rejected, len(negatives), evaluated, errors))
    total_errors = sum(r["errors"] for r in results)
    all_evaluated = all(r["evaluated"] == entries_total for r in results)
    status = "COMPLETED" if all_evaluated and total_errors == 0 else ("BLOCKED" if total_errors > 0 else "PARTIAL")
    payload = {
        "status": status,
        "kind": "v2_2",
        "grid": {"MIN_DURATION_SECONDS": MIN_DUR_GRID, "STATIC_THRESHOLD_PX": STATIC_THRESH_GRID},
        "results": results,
    }
    (OUT / "sweep_r3.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
if __name__ == "__main__":
    main()
