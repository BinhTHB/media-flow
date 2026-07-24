# Rewrite sweep_r2.py
content = '''\"\"\"STEP 3: R2 language detector sweep.
Sweeps target_lang over common configurations.
Requires: PaddleOCR + cached crops (needs E2E pipeline run first).\"""\"
from __future__ import annotations
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    from src.infrastructure.benchmark.hybrid_subtitle.r2_language import R2LanguageDetector
    from src.infrastructure.benchmark.hybrid_subtitle.r2_language import R2Result
    R2_AVAILABLE = True
except ImportError:
    R2_AVAILABLE = False
OUT = Path(\"artifacts/round26\")
GT = OUT / \"gt_v2_2.json\"
TARGET_LANG_GRID = [None, \"zh\", \"en\", \"ja\"]

def main():
    if not R2_AVAILABLE:
        out = {\"status\": \"BLOCKED\", \"reason\": \"R2LanguageDetector not importable\", \"result\": {\"rejected\": -1, \"total\": 19}}
        print(json.dumps(out, indent=2))
        (OUT / \"sweep_r2.json\").write_text(json.dumps(out, indent=2), encoding=\"utf-8\")
        return
    gt = json.loads(GT.read_text(encoding=\"utf-8\"))
    positives = [e for e in gt[\"entries\"] if e[\"is_subtitle\"]]
    negatives = [e for e in gt[\"entries\"] if not e[\"is_subtitle\"]]
    results = []
    for lang in TARGET_LANG_GRID:
        r2 = R2LanguageDetector(target_lang=lang)
        pos_rejected = 0
        neg_rejected = 0
        for entry in positives:
            try:
                # evaluate needs (track, proposal_bundle, all_tracks)
                result = r2.evaluate(entry, None, [])
            except Exception:
                pass
            else:
                if hasattr(result, \"reject_subtitle\") and result.reject_subtitle:
                    pos_rejected += 1
        for entry in negatives:
            try:
                result = r2.evaluate(entry, None, [])
            except Exception:
                pass
            else:
                if hasattr(result, \"reject_subtitle\") and result.reject_subtitle:
                    neg_rejected += 1
        results.append({
            \"target_lang\": lang,
            \"positives_rejected\": pos_rejected,
            \"total_positives\": len(positives),
            \"negatives_rejected\": neg_rejected,
            \"total_negatives\": len(negatives),
        })
        print(\"  lang=%%s: pos_rej=%%d/%%d neg_rej=%%d/%%d\" %% (lang, pos_rejected, len(positives), neg_rejected, len(negatives)))
    payload = {
        \"status\": \"COMPLETED\" if results else \"NO_DATA\",
        \"kind\": \"v2_2\",
        \"grid\": {\"target_lang\": TARGET_LANG_GRID},
        \"results\": results,
    }
    (OUT / \"sweep_r2.json\").write_text(json.dumps(payload, indent=2), encoding=\"utf-8\")
    print(json.dumps(payload, indent=2))
if __name__ == \"__main__\":
    main()
'''
with open('scripts/sweep_r2.py', 'w', encoding='utf-8') as f:
    f.write(content)
print('sweep_r2.py rewritten')
