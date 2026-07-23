"""STEP 3: R2 language detector sweep.
Requires: PaddleOCR + cached crops (needs E2E pipeline run first).
Without cached crops, reports DATA_NEEDED."""
from __future__ import annotations
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    from src.infrastructure.benchmark.hybrid_subtitle.r2_language import R2LanguageDetector
    R2_AVAILABLE = True
except ImportError:
    R2_AVAILABLE = False
OUT = Path("artifacts/round26")
GT = OUT / "gt_v2_2.json"
def main():
    if not R2_AVAILABLE:
        out = {"status": "BLOCKED", "reason": "R2LanguageDetector not importable", "result": {"rejected": -1, "total": 19}}
        print(json.dumps(out, indent=2))
        (OUT / "sweep_r2.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
        return
    gt = json.loads(GT.read_text(encoding="utf-8"))
    positives = [e for e in gt["entries"] if e["is_subtitle"]]
    negatives = [e for e in gt["entries"] if not e["is_subtitle"]]
    r2 = R2LanguageDetector()
    pos_rejected = 0
    naeg_rejected = 0
    for entry in positives:
        result = r2.detect(entry["bbox_xyxy"], None)
        if result and result.reject_subtitle:
            pos_rejected += 1
    for entry in negatives:
        result = r2.detect(entry["bbox_xyxy"], None)
        if result and result.reject_subtitle:
            naeg_rejected += 1
    payload = {
        "status": "COMPLETED",
        "kind": "v2_2",
        "result": {"positives_rejected": pos_rejected, "total_positives": len(positives), "negatives_rejected": naeg_rejected, "total_negatives": len(negatives), "pass": pos_rejected == 0},
    }
    (OUT / "sweep_r2.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
if __name__ == "__main__":
    main()
