"""STEP 7: Combined V2.1 stack sweep (Tier → R1 → R2 → R3 → CropBatchClassifierV21).
Requires: E2E pipeline data with crops. Reports DATA_NEEDED without E2E run."""
from __future__ import annotations
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
OUT = Path("artifacts/round26")
GT = OUT / "gt_v2_2.json"
def main():
    gt = json.loads(GT.read_text(encoding="utf-8"))
    positives = [e for e in gt["entries"] if e["is_subtitle"]]
    payload = {
        "status": "DATA_NEEDED",
        "kind": "v2_2",
        "reason": "STEP 7 combined stack requires E2E pipeline run to produce cached crops + 720p context frames. Run E2E first: python main.py --stt --ocr-provider hybrid_gemini --subtitle-region-detector hybrid_yolo --ocr-device gpu --job-id round26-sample5",
        "result": {"rejected": -1, "total": len(positives)},
    }
    (OUT / "sweep_step7.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
if __name__ == "__main__":
    main()
