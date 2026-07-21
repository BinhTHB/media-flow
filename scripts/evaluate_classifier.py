"""STEP B evaluation: compute confusion matrix from frozen GT + v19 classifier output.

Usage:
  .venv/Scripts/python scripts/evaluate_classifier.py
"""
import json
from pathlib import Path

# Load frozen GT
GT_PATH = Path("scripts/frozen_gt/sample5")
sha_path = GT_PATH / "frozen_gt.sha256"
gt_path = GT_PATH / "frozen_gt.json"
import hashlib
expected_sha = sha_path.read_text(encoding="ascii").strip().split()[0]
actual_sha = hashlib.sha256(gt_path.read_bytes()).hexdigest()
if actual_sha != expected_sha:
    raise ValueError(f"frozen GT checksum mismatch: {actual_sha} != {expected_sha}")
print(f"GT frozen at SHA256: {expected_sha}")

gt_entries = json.loads(gt_path.read_text(encoding="utf-8"))
gt_by_id = {e["id"]: e["expected_is_subtitle"] for e in gt_entries}
total_tracks = len(gt_by_id)
n_subtitle_gt = sum(1 for v in gt_by_id.values() if v)
n_non_subtitle_gt = total_tracks - n_subtitle_gt

# Load classifier output: tracks that got mask events = classifier predicted subtitle
# Source: provenance=classification:Txxx in output/round19-gpu/subtitles/subtitle.ass
# All 147 tracks from frozen GT were found in ASS = classifier predicted ALL as subtitle
import re
ass_tracks = set()
for line in Path("output/round19-gpu/subtitles/subtitle.ass").read_text(encoding="utf-8", errors="replace").splitlines():
    m = re.search(r"provenance=classification:(T\d+)", line)
    if m:
        ass_tracks.add(m.group(1))
classified_subtitle_ids = ass_tracks
classified_non_subtitle_ids = set(gt_by_id.keys()) - ass_tracks

# Confusion matrix
tp = sum(1 for tid in classified_subtitle_ids if gt_by_id.get(tid) is True)
fp = sum(1 for tid in classified_subtitle_ids if gt_by_id.get(tid) is False)
tn = sum(1 for tid in classified_non_subtitle_ids if gt_by_id.get(tid) is False)
fn = sum(1 for tid in classified_non_subtitle_ids if gt_by_id.get(tid) is True)

precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

print(f"\n=== STEP B: Classifier Confusion Matrix on Frozen GT ===")
print(f"GT: {total_tracks} tracks ({n_subtitle_gt} subtitle, {n_non_subtitle_gt} non-subtitle)")
print(f"Classifier predicted subtitle: {len(classified_subtitle_ids)} (from ASS)")
print(f"Classifier predicted non-subtitle: {len(classified_non_subtitle_ids)}")
print(f"\nConfusion Matrix:")
print(f"  TP={tp} FP={fp}")
print(f"  FN={fn} TN={tn}")
print(f"\nMetrics:")
print(f"  Precision: {precision:.4f} ({tp}/{tp+fp})")
print(f"  Recall:    {recall:.4f} ({tp}/{tp+fn})")
print(f"  F1:        {f1:.4f}")
print(f"\n  TP rate (sensitivity): {tp/n_subtitle_gt:.4f} ({tp}/{n_subtitle_gt})")
print(f"  Specificity: {tn/n_non_subtitle_gt:.4f} ({tn}/{n_non_subtitle_gt})")

# Save result
result = {
    "step": "B",
    "ground_truth_sha256": expected_sha,
    "ground_truth_total_tracks": total_tracks,
    "ground_truth_subtitle": n_subtitle_gt,
    "ground_truth_non_subtitle": n_non_subtitle_gt,
    "classifier_predicted_subtitle": len(classified_subtitle_ids),
    "classifier_predicted_non_subtitle": len(classified_non_subtitle_ids),
    "confusion_matrix": {"TP": tp, "FP": fp, "TN": tn, "FN": fn},
    "precision": round(precision, 4),
    "recall": round(recall, 4),
    "f1": round(f1, 4),
    "note": "Classifier predicted all 147 GT tracks as subtitle (from ASS mask events). "
            "No tracks classified as non-subtitle in available v19 artifacts. "
            "Real verification would require running verifier on non-mask tracks."
}
out_dir = Path("artifacts/round20/gt")
out_dir.mkdir(parents=True, exist_ok=True)
(out_dir / "classifier_evaluation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
print(f"\nResult saved to artifacts/round20/gt/classifier_evaluation.json")