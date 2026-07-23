"""STEP 3: R2 language detector sweep.
Sweeps target_lang over common configurations.
Requires: PaddleOCR + cached crops (needs E2E pipeline run first)."""
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
OUT = Path("artifacts/round26")
GT = OUT / "gt_v2_2.json"
TARGET_LANG_GRID = [None, "zh", "en", "ja"]

def main():
    if not R2_AVAILABLE:
        out = {"status": "BLOCKED", "reason": "R2LanguageDetector not importable", "result": {"rejected": -1, "total": 19}}
        print(json.dumps(out, indent=2))
        (OUT / "sweep_r2.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
        return
    gt = json.loads(GT.read_text(encoding="utf-8"))
    positives = [e for e in gt["entries"] if e["is_subtitle"]]
    negatives = [e for e in gt["entries"] if not e["is_subtitle"]]
    entries_total = len(positives) + len(negatives)
    results = []
    for lang in TARGET_LANG_GRID:
        r2 = R2LanguageDetector(target_lang=lang)
        evaluated = 0
        errors = 0
        pos_rejected = 0
        neg_rejected = 0
        for entry in positives + negatives:
            is_positive = entry in positives
            # Build minimal track dict needed by R2 evaluate
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
                result = r2.evaluate(track, None, [], [])
                evaluated += 1
            except Exception as exc:
                errors += 1
                print("  ERROR: lang=%s id=%s %s" % (lang, entry["id"], exc))
                continue
            if result.rejected:
                if is_positive:
                    pos_rejected += 1
                else:
                    neg_rejected += 1
        status = "BLOCKED" if errors > 0 else ("COMPLETED" if evaluated == entries_total else "PARTIAL")
        results.append({
            "target_lang": lang,
            "positives_rejected": pos_rejected,
            "total_positives": len(positives),
            "negatives_rejected": neg_rejected,
            "total_negatives": len(negatives),
            "evaluated": evaluated,
            "errors": errors,
        })
        print("  lang=%s: pos_rej=%d/%d neg_rej=%d/%d (eval=%d err=%d)" % (lang, pos_rejected, len(positives), neg_rejected, len(negatives), evaluated, errors))
        if errors > 0:
            print("  BLOCKED due to %d error(s) in lang=%s" % (errors, lang))
    total_errors = sum(r["errors"] for r in results)
    all_evaluated = all(r["evaluated"] == entries_total for r in results)
    status = "COMPLETED" if all_evaluated and total_errors == 0 else ("BLOCKED" if total_errors > 0 else "PARTIAL")
    payload = {
        "status": status,
        "kind": "v2_2",
        "grid": {"target_lang": TARGET_LANG_GRID},
        "results": results,
    }
    (OUT / "sweep_r2.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
if __name__ == "__main__":
    main()
