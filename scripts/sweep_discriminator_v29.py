#!/usr/bin/env python3
"""Sweep static discriminator metrics on 22 evaluation tracks to find optimal thresholds.
Requires video file test_videos/real_mini_benchmark/sample5.mp4 and baseline tracks.
"""
from __future__ import annotations
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.application.dto.proposal_bundle import ProposalBundle
from scripts.offline_gate_v29 import (
    load_gt,
    load_tracks,
    collect_frame_indices,
    build_proposal_bundle,
    POSITIVE_IDS,
    FP_IDS,
)

SEGMENTED_TRANSCRIPT = ROOT / "output" / "round27-sample5" / "transcripts" / "segmented.json"
METRICS_OUTPUT = ROOT / "artifacts" / "round29" / "discriminator_metrics.json"

# ---------------------------------------------------------------------------
# Feature Extraction
# ---------------------------------------------------------------------------
def load_speech_segments() -> list[dict]:
    if not SEGMENTED_TRANSCRIPT.exists():
        print(f"[WARN] segmented transcript not found at {SEGMENTED_TRANSCRIPT}. Running blank speech.")
        return []
    with open(SEGMENTED_TRANSCRIPT, encoding="utf-8") as f:
        data = json.load(f)
    speech = [item for item in data if item.get("source") == "speech"]
    print(f"[OK] Loaded {len(speech)} speech segments.")
    return speech


def extract_track_features(
    track: dict,
    proposal_bundle: ProposalBundle,
    speech_segments: list[dict],
) -> dict:
    """Extract (duration, speech_overlap_ratio, average_rgb_variance) for a track."""
    groups = track.get("groups", [])
    if not groups:
        return {"duration": 0.0, "speech_overlap": 0.0, "variance": 0.0}

    # 1. Duration
    first_ts = groups[0].get("timestamp_ms", 0) / 1000.0
    last_ts = groups[-1].get("timestamp_ms", 0) / 1000.0
    duration = max(0.0, last_ts - first_ts)

    # 2. Speech overlap
    overlap = 0.0
    for s in speech_segments:
        s_start = float(s.get("start", 0.0))
        s_end = float(s.get("end", 0.0))
        overlap += max(0.0, min(last_ts, s_end) - max(first_ts, s_start))
    overlap_ratio = min(1.0, overlap / duration) if duration > 0 else 0.0

    # 3. Variance
    variances = []
    for g in groups:
        fi = g.get("frame_index")
        bbox = g.get("union_bbox_xyxy")
        if fi is None or not bbox or len(bbox) != 4:
            continue
        try:
            decoded = proposal_bundle.frame_for_index(fi)
        except (KeyError, AttributeError):
            continue
        img = decoded.image
        if img is None:
            continue

        H, W = img.shape[:2]
        x1, y1, x2, y2 = [int(round(coord)) for coord in bbox]
        # Clamp coordinates
        x1 = max(0, min(W - 1, x1))
        y1 = max(0, min(H - 1, y1))
        x2 = max(0, min(W, x2))
        y2 = max(0, min(H, y2))

        if (x2 - x1) < 2 or (y2 - y1) < 2:
            continue

        crop = img[y1:y2, x1:x2]
        # RGB variance
        color_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        var = float(np.var(color_crop))
        variances.append(var)

    avg_variance = float(np.mean(variances)) if variances else 0.0

    return {
        "duration": round(duration, 4),
        "speech_overlap": round(overlap_ratio, 4),
        "variance": round(avg_variance, 4),
    }


# ---------------------------------------------------------------------------
# Sweep / Evaluation
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("TASK 3 — Static Discriminator Threshold Sweeper")
    print("=" * 60)

    # Load tracks and GT
    gt = load_gt()
    all_tracks = load_tracks()
    ev_tracks = [t for t in all_tracks if t["track_id"] in set(POSITIVE_IDS + FP_IDS)]

    needed_indices = collect_frame_indices(ev_tracks)
    proposal_bundle = build_proposal_bundle(needed_indices)
    speech = load_speech_segments()

    # Extract features per-track
    metrics_ledger = {}
    positives = []
    fps = []

    for t in ev_tracks:
        tid = t["track_id"]
        expected = gt.get(tid, False)
        feats = extract_track_features(t, proposal_bundle, speech)
        feats["expected_is_subtitle"] = expected
        logger_name = "POSITIVE" if expected else "FALSE_POSITIVE"
        print(f"  Track {tid:<6} ({logger_name}): dur={feats['duration']:>6}s | speech={feats['speech_overlap']:>6.2%} | var={feats['variance']:>10.2f}")
        metrics_ledger[tid] = feats
        if expected:
            positives.append(feats)
        else:
            fps.append(feats)

    # Save metrics thô
    METRICS_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(METRICS_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(metrics_ledger, f, indent=2, ensure_ascii=False)
    print(f"\n[DUMP] Wrote metrics ledger to {METRICS_OUTPUT}")

    # Analyze distributions
    print(f"\n{'-'*60}")
    print("DISTRIBUTION STUDY")
    print(f"{'-'*60}")

    pos_var = [p["variance"] for p in positives]
    fp_var = [f["variance"] for f in fps]
    pos_speech = [p["speech_overlap"] for p in positives]
    fp_speech = [f["speech_overlap"] for f in fps]
    pos_dur = [p["duration"] for p in positives]
    fp_dur = [f["duration"] for f in fps]

    print(f"Variance check:")
    print(f"  Pos: min={np.min(pos_var):.2f}, mean={np.mean(pos_var):.2f}, max={np.max(pos_var):.2f}")
    print(f"  FP:  min={np.min(fp_var):.2f}, mean={np.mean(fp_var):.2f}, max={np.max(fp_var):.2f}")
    print(f"Speech overlap check:")
    print(f"  Pos: min={np.min(pos_speech):.2%}, mean={np.mean(pos_speech):.2%}, max={np.max(pos_speech):.2%}")
    print(f"  FP:  min={np.min(fp_speech):.2%}, mean={np.mean(fp_speech):.2%}, max={np.max(fp_speech):.2%}")
    print(f"Duration check:")
    print(f"  Pos: min={np.min(pos_dur):.2f}s, mean={np.mean(pos_dur):.2f}s, max={np.max(pos_dur):.2f}s")
    print(f"  FP:  min={np.min(fp_dur):.2f}s, mean={np.mean(fp_dur):.2f}s, max={np.max(fp_dur):.2f}s")

    # Let's search for discriminant criteria
    # A track is predicted as Subtitle if it passes static check (e.g. variance belongs to sub class)
    # Subtitles must have variance below a threshold, or speech overlap above a threshold?
    # Actually, subtitles typically represent visual text tracks. Watermarks are static.
    # If a track has displacement <= 12, duration >= 3, AND speech overlap <= 0.05 -> it is classically a static watermark (should be rejected).
    # Can we improve the discriminator?
    # Let's sweep: if displacement <= 12 and duration >= min_dur and speech_overlap <= max_speech and variance <= max_variance_for_clean_text.
    # Let's perform a sweep to look for 100% separation on these 22 tracks.
    best_tp = 0
    best_tn = 0
    best_param = None

    # Sweep variance-only threshold (simple, since gap is very clean)
    for min_var in np.linspace(1000, 10000, 19):
        for min_dur in np.linspace(0.0, 4.0, 9):
            tp = 0
            fn = 0
            tn = 0
            fp_wrong = 0

            for tid, feats in metrics_ledger.items():
                dur = feats["duration"]
                var = feats["variance"]
                expected = feats["expected_is_subtitle"]

                # Rule: reject if variance is low (subtitle has high variance) OR duration too short
                # Subtitles (positive) = high variance + long; Watermarks (FP) = low variance
                # Prediction: is_subtitle = (var >= min_var AND dur >= min_dur)
                predicted = var >= min_var and dur >= min_dur

                if expected:
                    if predicted:
                        tp += 1
                    else:
                        fn += 1
                else:
                    if not predicted:
                        tn += 1
                    else:
                        fp_wrong += 1

            if tp == 14 and tn >= best_tn:
                best_tp = tp
                best_tn = tn
                best_param = (min_dur, round(min_var))

    print(f"\n{'-'*60}")
    print("SWEEP RESULTS")
    print(f"{'-'*60}")
    if best_param:
        min_dur, min_var = best_param
        print(f"Optimal parameters found:")
        print(f"  min_variance:    {min_var:.2f}")
        print(f"  min_duration:    {min_dur:.2f}s")
        print(f"Resulting separation:")
        print(f"  Positives Kept:  {best_tp}/14")
        print(f"  FPs Rejected:    {best_tn}/8")

        # Output confusion matrix
        print(f"\nCONFUSION MATRIX:")
        print(f"                 Expected Subtitle    Expected FP")
        print(f"Predicted Sub    {best_tp:<17}    {8 - best_tn:<11}")
        print(f"Predicted FP     {14 - best_tp:<17}    {best_tn:<11}")

        # Gap verification:
        # Let's check if the threshold sits in a clean gap
        print(f"\nChecking class gap for variance (min_var={min_var:.2f}):")
        match_candidates = []
        for tid, feats in metrics_ledger.items():
            match_candidates.append((tid, feats["variance"], feats["expected_is_subtitle"]))

        match_candidates.sort(key=lambda x: x[1])
        print("  Variance distribution across all 22 tracks:")
        for tid, var_val, exp in match_candidates:
            label = "Subtitle" if exp else "FP_Watermark"
            print(f"    Track {tid:<5} ({label:<12}): variance = {var_val:.2f}")

    else:
        print("[FAIL] No parameter set yielded 0 False Negatives and significant FP rejections.")

if __name__ == "__main__":
    main()
