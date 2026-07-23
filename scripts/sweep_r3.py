"""STEP 4: R3 static discriminator sweep.
Requires: video frames + scene-cut data.
Can run offline with round25-sample5 frames."""
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
def main():
    if not R3_AVAILABLE:
        out = {"status": "BLOCKED", "reason": "R3 components not importable", "result": {"rejected": -1, "total": 19}}
        print(json.dumps(out, indent=2))
        (OUT / "sweep_r3.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
        return
    gt = json.loads(GT.read_text(encoding="utf-8"))
    positives = [e for e in gt["entries"] if e["is_subtitle"]]
    negatives = [e for e in gt["entries"] if not e["is_subtitle"]]
    scene = SceneCutDetector()
    r3 = R3StaticDiscriminator(scene)
    pos_rejected = 0
    naeg_rejected = 0
    for entry in positives:
        result = r3.evaluate(entry["bbox_xyxy"], 720, 1280, 0.0, [])
        if result and getattr(result, "reject_subtitle", False):
            pos_rejected += 1
    for entry in negatives:
        result = r3.evaluate(entry["bbox_xyxy"], 720, 1280, 0.0, [])
        if result and getattr(result, "reject_subtitle", False):
            naeg_rejected += 1
    payload = {
        "status": "COMPLETED" if any([positives, negatives]) else "NO_DATA",
        "kind": "v2_2",
        "result": {"positives_rejected": pos_rejected, "total_positives": len(positives), "negatives_rejected": naeg_rejected, "total_negatives": len(negatives), "pass": pos_rejected == 0},
    }
    (OUT / "sweep_r3.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
if __name__ == "__main__":
    main()
