#!/usr/bin/env python3
"""Sweep bottom-band y1_norm threshold on all 42 GT V2.3 entries.

Goal: choose a threshold that keeps 14/14 positives while rejecting as many
of the 28 negatives as possible.

Usage:
    .venv/Scripts/python scripts/sweep_band_gate_r30.py

Outputs:
    artifacts/round30/band_sweep.json  — full sweep table
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

GT_PATH = ROOT / "artifacts" / "round28" / "gt_v2_3.json"
OUTPUT_PATH = ROOT / "artifacts" / "round30" / "band_sweep.json"

FRAME_HEIGHT = 1280  # sample5 is 720x1280 portrait

# ---------------------------------------------------------------------------
# Load GT V2.3 (42 entries with bbox_xyxy)
# ---------------------------------------------------------------------------
def load_gt() -> dict:
    raw = json.loads(GT_PATH.read_text(encoding="utf-8"))
    entries = raw["entries"]
    print(f"[GT] Loaded {len(entries)} entries from GT V2.3")

    positive_ids: set[str] = set()
    negative_ids: set[str] = set()

    by_id: dict[str, dict] = {}
    for e in entries:
        eid = e["id"]
        by_id[eid] = e
        if e["is_subtitle"]:
            positive_ids.add(eid)
        else:
            negative_ids.add(eid)

    print(f"  Positives: {len(positive_ids)} — {sorted(positive_ids)}")
    print(f"  Negatives: {len(negative_ids)}")
    return by_id, positive_ids, negative_ids


# ---------------------------------------------------------------------------
# Compute y1_norm for all entries
# ---------------------------------------------------------------------------
def compute_y1_norm(by_id):
    result = {}
    for eid, entry in by_id.items():
        bbox = entry.get("bbox_xyxy")
        if not bbox or len(bbox) < 4:
            result[eid] = None
            continue
        y1 = bbox[1]
        result[eid] = y1 / FRAME_HEIGHT
    return result


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("BAND GATE SWEEP — y1_norm on 42 GT V2.3 entries")
    print(f"  Frame height: {FRAME_HEIGHT} px")
    print(f"  GT:           {GT_PATH.name}")
    print(f"  Sweep range:  0.60 → 0.80  step 0.01")
    print("=" * 60)

    by_id, pos_ids, neg_ids = load_gt()
    y1n = compute_y1_norm(by_id)

    # Print per-track y1_norm
    print(f"\n{'='*60}")
    print("Per-track y1_norm (top of bounding box / frame height):")
    print(f"{'='*60}")
    for eid in sorted(by_id.keys()):
        label = "POSITIVE" if eid in pos_ids else "NEGATIVE"
        val = y1n.get(eid)
        if val is not None:
            print(f"  {eid:<20} ({label:<8}): y1_norm = {val:.4f}  (y1={by_id[eid]['bbox_xyxy'][1]:.0f})")
        else:
            print(f"  {eid:<20} ({label:<8}): NO BBOX")

    # Sweep thresholds
    thresholds = [round(t * 0.01, 2) for t in range(60, 81)]  # 0.60 to 0.80
    rows = []

    for th in thresholds:
        pos_kept = 0
        pos_lost = 0
        neg_kept = 0   # negatives that PASS through band (not rejected)
        neg_rejected = 0

        pos_lost_ids = []
        neg_kept_ids = []

        for eid in sorted(by_id.keys()):
            val = y1n.get(eid)
            if val is None:
                continue
            passes_band = val >= th  # bottom band = y1_norm >= threshold
            is_pos = eid in pos_ids

            if is_pos:
                if passes_band:
                    pos_kept += 1
                else:
                    pos_lost += 1
                    pos_lost_ids.append(eid)
            else:
                if passes_band:
                    neg_kept += 1
                    neg_kept_ids.append(eid)
                else:
                    neg_rejected += 1

        rows.append({
            "threshold": th,
            "positives_kept": pos_kept,
            "positives_lost": pos_lost,
            "positives_lost_ids": pos_lost_ids,
            "negatives_kept": neg_kept,
            "negatives_rejected": neg_rejected,
            "negatives_kept_ids": neg_kept_ids,
            "fn_zero": pos_lost == 0,
        })

    # Print sweep table
    print(f"\n{'='*100}")
    print(f"{'Thresh':<8} {'Pos kept':<10} {'Pos lost':<10} {'Neg kept':<10} {'Neg rej':<10} {'FN=0':<8}  {'Negatives kept (ids)'}")
    print(f"{'='*100}")
    best_th = None
    best_neg_rejected = -1
    best_row = None

    for r in rows:
        fn_ok = "YES" if r["fn_zero"] else "NO"
        neg_kept_short = ",".join(r["negatives_kept_ids"][:5])
        if len(r["negatives_kept_ids"]) > 5:
            neg_kept_short += f"...(+{len(r['negatives_kept_ids'])-5})"
        print(f"  {r['threshold']:<7.2f} {r['positives_kept']:<10} {r['positives_lost']:<10} {r['negatives_kept']:<10} {r['negatives_rejected']:<10} {fn_ok:<8}  {neg_kept_short}")

        # Best threshold: FN=0 and max negatives rejected
        if r["fn_zero"] and r["negatives_rejected"] > best_neg_rejected:
            best_neg_rejected = r["negatives_rejected"]
            best_th = r["threshold"]
            best_row = r

    # Print recommendation
    print(f"\n{'='*60}")
    print("RECOMMENDATION")
    print(f"{'='*60}")
    if best_row:
        # Find the lowest positive y1_norm to compute safe margin
        pos_y1n = {eid: v for eid, v in y1n.items() if eid in pos_ids and v is not None}
        min_pos_y1n = min(pos_y1n.values())
        print(f"  Lowest positive y1_norm: {min_pos_y1n:.4f}")
        print(f"  Selected threshold:      {best_th:.2f}")
        print(f"  Safe margin:             {min_pos_y1n - best_th:.4f} (lùi {min_pos_y1n - best_th:.4f} so với positive thấp nhất)")
        print(f"  Positives kept:           {best_row['positives_kept']}/14")
        print(f"  Negatives rejected:       {best_row['negatives_rejected']}/28")
        print(f"  Negatives still passing:  {best_row['negatives_kept']}/28")
        if best_row['negatives_kept'] > 0:
            print(f"  Negatives kept (ids):     {best_row['negatives_kept_ids']}")

        # If best threshold is the same as many, pick one with safe margin >= 0.02
        safe_candidates = [r for r in rows if r["fn_zero"] and (min_pos_y1n - r["threshold"]) >= 0.02]
        if safe_candidates:
            pick = max(safe_candidates, key=lambda r: r["negatives_rejected"])
            print(f"\n  --- Safe margin >= 0.02 filtered ---")
            print(f"  Selected threshold:      {pick['threshold']:.2f}")
            print(f"  Safe margin:             {min_pos_y1n - pick['threshold']:.4f}")
            print(f"  Negatives rejected:       {pick['negatives_rejected']}/28")
            best_th = pick["threshold"]
            best_row = pick
    else:
        print("  [FAIL] No threshold with FN=0 found.")

    # Write full sweep table
    output = {
        "gt_source": str(GT_PATH),
        "frame_height": FRAME_HEIGHT,
        "sweep_range": {"min": 0.60, "max": 0.80, "step": 0.01},
        "per_track_y1_norm": {eid: round(v, 4) if v is not None else None for eid, v in sorted(y1n.items())},
        "sweep_table": rows,
        "recommendation": {
            "threshold": best_th,
            "safe_margin": round(min_pos_y1n - best_th, 4) if best_row else None,
            "positives_kept": best_row["positives_kept"] if best_row else 0,
            "negatives_rejected": best_row["negatives_rejected"] if best_row else 0,
            "negatives_kept_ids": best_row["negatives_kept_ids"] if best_row else [],
        },
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[OUTPUT] Wrote sweep table to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
