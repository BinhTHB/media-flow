#!/usr/bin/env python3
"""Offline gate V29 — single-crop vs separator-batch Gemini classification on 22 real tracks.

Usage:
    cd /e/projects/media-flow && ./.venv/Scripts/python scripts/offline_gate_v29.py

Requirements:
    - .env with GEMINI_API_KEY
    - test_videos/real_mini_benchmark/sample5.mp4
    - artifacts/round23/baseline_tracks_conf_0.30.json
    - scripts/frozen_gt/sample5/frozen_gt.json
"""
from __future__ import annotations
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Add project root
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Manual .env load
env_path = ROOT / ".env"
if env_path.exists():
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, val = line.split("=", 1)
            val = val.strip().strip("'").strip('"')
            # Support comma-separated keys (take first)
            if "," in val:
                val = val.split(",")[0].strip()
            os.environ[key.strip()] = val

from src.application.dto.proposal_bundle import ProposalBundle, DecodedFrame
from src.infrastructure.benchmark.hybrid_subtitle.crop_batch_classifier import CropBatchClassifierV21, BUDGET_LIMIT
from src.infrastructure.benchmark.hybrid_subtitle.crop_extractor import TrackCropExtractor
from src.infrastructure.gemini.client import GeminiClient

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
VIDEO_PATH = ROOT / "test_videos" / "real_mini_benchmark" / "sample5.mp4"
BASELINE_PATH = ROOT / "artifacts" / "round23" / "baseline_tracks_conf_0.30.json"
GT_PATH = ROOT / "scripts" / "frozen_gt" / "sample5" / "frozen_gt.json"
ARTIFACT_DIR = ROOT / "artifacts" / "round29"
GEMINI_MODEL = "gemini-3.5-flash-lite"

POSITIVE_IDS = ['T003', 'T006', 'T010', 'T014', 'T016', 'T019',
                'T116', 'T120', 'T127', 'T133', 'T135', 'T137', 'T138', 'T146']
FP_IDS = ['T030', 'T041', 'T044', 'T047', 'T055', 'T061', 'T071', 'T105']
CANARY_ID = "T001"  # real negative GT track

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _resize_720p(image, max_dim=720):
    H, W = image.shape[:2]
    if max(H, W) <= max_dim:
        return image.copy()
    scale = max_dim / max(H, W)
    new_w = int(round(W * scale))
    new_h = int(round(H * scale))
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _request_hash(track_ids: list[str], variant: str) -> str:
    """Deterministic hash of track IDs for ledger traceability."""
    raw = variant + "|" + "|".join(sorted(track_ids))
    return hashlib.sha256(raw.encode()).hexdigest()[:12]


def _extract_text_from_decision(dec: dict) -> str:
    """Extract text_echo or reason from a Gemini decision dict."""
    return str(dec.get("text_echo", "") or dec.get("reason", "") or "")


def _t003_yes(mapped: dict) -> bool:
    """Check T003 was classified as subtitle (expected positive)."""
    d = mapped.get("T003", {})
    return d.get("is_subtitle") is True


# ---------------------------------------------------------------------------
# 1. Load GT and tracks
# ---------------------------------------------------------------------------
def load_gt() -> dict:
    with open(GT_PATH, encoding="utf-8") as f:
        gt_list = json.load(f)
    return {item["id"]: item["expected_is_subtitle"] for item in gt_list}


def load_tracks() -> list[dict]:
    with open(BASELINE_PATH, encoding="utf-8") as f:
        all_tracks = json.load(f)
    target_ids = set(POSITIVE_IDS + FP_IDS + [CANARY_ID])
    selected = [t for t in all_tracks if t["track_id"] in target_ids]
    found = {t["track_id"] for t in selected}
    missing = target_ids - found
    if missing:
        print(f"[FATAL] Missing tracks in baseline: {missing}")
        sys.exit(1)
    print(f"[OK] Loaded {len(selected)} tracks from baseline ({len(POSITIVE_IDS)} pos + {len(FP_IDS)} fp + canary)")
    return selected


def collect_frame_indices(tracks: list[dict]) -> set[int]:
    """Collect all unique frame indices needed from the tracks."""
    indices: set[int] = set()
    for t in tracks:
        for g in t.get("groups", []):
            fi = g.get("frame_index")
            if fi is not None:
                indices.add(fi)
    return indices


# ---------------------------------------------------------------------------
# 2. Decode video → ProposalBundle (video frames only for needed indices)
# ---------------------------------------------------------------------------
def build_proposal_bundle(needed_indices: set[int]) -> ProposalBundle:
    """Decode needed frames from video and build a ProposalBundle."""
    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {VIDEO_PATH}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    vw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    bundle = ProposalBundle(
        video_path=str(VIDEO_PATH),
        fps=fps,
        stride=1,
        video_width=vw,
        video_height=vh,
        total_frames=total_frames,
    )

    frame_index = 0
    while True:
        ret, img = cap.read()
        if not ret:
            break
        if frame_index in needed_indices:
            ts = frame_index / fps
            image_720p = _resize_720p(img)
            bundle.add_frame(
                frame_index=frame_index,
                timestamp_sec=round(ts, 4),
                image=img,
                image_path=None,
                detections=[],  # no YOLO — crops come from video frames directly
            )
            bundle.frames[-1].image_720p = image_720p
            bundle.frames[-1].crops = []  # no pre-extracted crops
        frame_index += 1

    cap.release()
    print(f"[VIDEO] Decoded {len(bundle.frames)} frames (needed {len(needed_indices)}) from {VIDEO_PATH.name}")
    return bundle


# ---------------------------------------------------------------------------
# 3. Build canary evidence from a real negative GT track
# ---------------------------------------------------------------------------
def build_canary_track(tracks: list[dict], canary_id: str) -> dict | None:
    for t in tracks:
        if t["track_id"] == canary_id:
            return t
    return None


# ---------------------------------------------------------------------------
# 4. Run single-variant gate and return results
# ---------------------------------------------------------------------------
def run_variant(
    variant_name: str,
    batch_size: int,
    ev_tracks: list[dict],
    canary_track: dict | None,
    proposal_bundle: ProposalBundle,
    gemini_client: GeminiClient,
) -> dict:
    """Run one variant, return results dict with ledger, confusion matrix, etc."""
    extractor = TrackCropExtractor()
    artifact_dir = ARTIFACT_DIR / variant_name
    artifact_dir.mkdir(parents=True, exist_ok=True)
    ledger_path = artifact_dir / "crop_classifier_v29_ledger.jsonl"

    classifier = CropBatchClassifierV21(
        gemini_client=gemini_client,
        model=GEMINI_MODEL,
        budget_limit=BUDGET_LIMIT,
        artifact_dir=artifact_dir,
        job_id=f"v29_{variant_name}",
        run_id=variant_name,
    )

    # Build canary evidence from the real negative track
    canary_evidence = None
    if canary_track:
        # We must pass canary as a list of tracks to the classifier's classify_batch
        # But classify_batch accepts canary_tracks list directly
        canary_tracks_list = [canary_track]
    else:
        canary_tracks_list = []

    # Run classification
    start_ts = time.time()
    try:
        mapped = classifier.classify_batch(
            ev_tracks,
            proposal_bundle,
            batch_size=batch_size,
            canary_tracks=canary_tracks_list,
        )
    except RuntimeError as exc:
        print(f"[{variant_name}] RUNTIME ERROR: {exc}")
        return {"variant": variant_name, "status": "ERROR", "error": str(exc)}
    elapsed = time.time() - start_ts

    # Build results
    gt = load_gt()
    results: dict[str, bool] = {}
    ledger_entries: list[dict] = []

    for t in ev_tracks:
        tid = t["track_id"]
        dec = mapped.get(tid, {})
        predicted = dec.get("is_subtitle") is True
        expected = gt.get(tid, False)
        results[tid] = predicted

        echo = _extract_text_from_decision(dec)
        entry = {
            "job_id": classifier.job_id,
            "request_hash": _request_hash([tid], variant_name),
            "track_id": tid,
            "variant": variant_name,
            "batch_size": batch_size,
            "expected": expected,
            "predicted": predicted,
            "is_subtitle": dec.get("is_subtitle"),
            "burn_in": dec.get("burn_in"),
            "category": dec.get("category", ""),
            "confidence": dec.get("confidence"),
            "text_echo": echo,
            "reason": dec.get("reason", ""),
        }
        ledger_entries.append(entry)

    # Read classifier's internal ledger (written by _log_ledger)
    per_request: list[dict] = []
    if ledger_path.exists():
        with open(ledger_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    per_request.append(json.loads(line))

    # Compute statistics first
    pos_results = {tid: results[tid] for tid in POSITIVE_IDS if tid in results}
    fp_results = {tid: results[tid] for tid in FP_IDS if tid in results}
    tp = sum(1 for v in pos_results.values() if v is True)
    fn = sum(1 for v in pos_results.values() if v is False)
    fp_rejected = sum(1 for v in fp_results.values() if v is False)
    fp_wrong = sum(1 for v in fp_results.values() if v is True)

    recall = tp / (tp + fn) if (tp + fn) else 0.0
    fp_reject_rate = fp_rejected / (fp_rejected + fp_wrong) if (fp_rejected + fp_wrong) else 0.0
    pred_pos = sum(1 for v in results.values() if v is True)
    precision = tp / pred_pos if pred_pos > 0 else 0.0
    t003_ok = _t003_yes(mapped)

    # Build per-request ledger with required fields
    per_request_entries: list[dict] = []
    request_counter = 0
    for batch_start in range(0, len(ev_tracks), batch_size):
        batch_tracks = ev_tracks[batch_start : batch_start + batch_size]
        batch_tids = [t["track_id"] for t in batch_tracks]
        rhash = _request_hash(batch_tids, variant_name)
        # Get the corresponding ledger entry from classifier
        req_entry = None
        if request_counter < len(per_request):
            req_entry = per_request[request_counter]
        # Aggregate decisions for this batch
        batch_echoes = []
        batch_decisions = []
        for tid in batch_tids:
            dec = mapped.get(tid, {})
            batch_echoes.append(_extract_text_from_decision(dec))
            batch_decisions.append({
                "track_id": tid,
                "is_subtitle": dec.get("is_subtitle"),
                "confidence": dec.get("confidence"),
            })
        canary_ok = True
        if canary_track:
            canary_tid = f"canary_{canary_track['track_id']}"
            canary_dec = mapped.get(canary_tid, {})
            canary_ok = canary_dec.get("is_subtitle") is not True
        entry = {
            "job_id": classifier.job_id,
            "request_hash": rhash,
            "variant": variant_name,
            "batch_track_ids": batch_tids,
            "track_count": len(batch_tids),
            "echo_summary": batch_echoes[0] if batch_echoes else "",
            "decisions": batch_decisions,
            "canary_pass": canary_ok,
            "canary_track": canary_track["track_id"] if canary_track else None,
            "tokens": req_entry.get("usage", {}) if req_entry else {},
            "cache_hit": req_entry.get("live_or_cache") == "cache" if req_entry else False,
            "model": GEMINI_MODEL,
            "gate_status": "PASS" if (t003_ok and recall >= 0.85 and fp_reject_rate >= 0.75) else "FAIL",
        }
        per_request_entries.append(entry)
        request_counter += 1

    # Write per-request ledger
    request_ledger_path = artifact_dir / "request_ledger.jsonl"
    with open(request_ledger_path, "w", encoding="utf-8") as f:
        for entry in per_request_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"[{variant_name}] Wrote per-request ledger ({len(per_request_entries)} entries) to {request_ledger_path}")

    summary = {
        "variant": variant_name,
        "batch_size": batch_size,
        "status": "OK",
        "total_tracks": len(ev_tracks),
        "pos_count": len(pos_results),
        "fp_count": len(fp_results),
        "true_positive": tp,
        "false_negative": fn,
        "fp_wrongly_accepted": fp_wrong,
        "fp_correctly_rejected": fp_rejected,
        "recall": round(recall, 4),
        "precision": round(precision, 4),
        "fp_reject_rate": round(fp_reject_rate, 4),
        "t003_is_subtitle": t003_ok,
        "requests_sent": classifier.requests_sent,
        "retry_count": classifier.retry_count,
        "budget_used": classifier.requests_sent,
        "budget_limit": classifier.budget_limit,
        "budget_ok": classifier.requests_sent <= classifier.budget_limit,
        "elapsed_seconds": round(elapsed, 2),
        "canary_track": CANARY_ID,
        "per_request_ledger_count": len(per_request),
        "per_track_ledger_count": len(ledger_entries),
    }

    # Write detailed per-track results
    detail_path = artifact_dir / "per_track_results.json"
    with open(detail_path, "w", encoding="utf-8") as f:
        json.dump(ledger_entries, f, indent=2, ensure_ascii=False)
    print(f"[{variant_name}] Wrote per-track results to {detail_path}")

    # Write summary
    summary_path = artifact_dir / "gate_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[{variant_name}] Wrote gate summary to {summary_path}")

    # Print human-readable gate report
    gate_pass = (
        recall >= 0.85
        and fp_reject_rate >= 0.75  # at least 6/8
        and t003_ok
        and classifier.requests_sent <= classifier.budget_limit
    )
    print(f"\n{'='*60}")
    print(f"GATE REPORT — {variant_name} (batch_size={batch_size})")
    print(f"{'='*60}")
    print(f"  Recall (TP/Pos):        {tp}/{len(pos_results)} = {recall:.2%}")
    print(f"  FP rejection:           {fp_rejected}/{len(fp_results)} ({fp_wrong} wrong accepts)")
    print(f"  Precision:              {precision:.2%}")
    print(f"  T003 is_subtitle:       {'YES' if t003_ok else 'NO'}")
    print(f"  Requests used:          {classifier.requests_sent} (limit: {classifier.budget_limit})")
    print(f"  Retries:                {classifier.retry_count}")
    print(f"  Budget OK:              {'YES' if summary['budget_ok'] else 'NO'}")
    print(f"  Gate:                   {'PASS' if gate_pass else 'FAIL'}")
    print(f"  Ledger entries:         {len(per_request)} per-request, {len(ledger_entries)} per-track")
    print(f"{'='*60}\n")

    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("OFFLINE GATE V29 — Single-crop vs Separator-batch")
    print(f"GT SHA: 99d1d75e04226c5b7946d3b24532167f6875da73be9ff69f40fbb2a8f3940bd5")
    print(f"Video:  {VIDEO_PATH.name}")
    print(f"Tracks: {len(POSITIVE_IDS)} positive + {len(FP_IDS)} FP = {len(POSITIVE_IDS) + len(FP_IDS)}")
    print(f"Canary: {CANARY_ID} (real negative GT)")
    print(f"Model:  {GEMINI_MODEL}")
    print("=" * 60)

    # 1. Load tracks
    all_tracks = load_tracks()
    ev_tracks = [t for t in all_tracks if t["track_id"] in set(POSITIVE_IDS + FP_IDS)]
    canary_track_data = build_canary_track(all_tracks, CANARY_ID)

    # 2. Collect frame indices and build ProposalBundle
    needed_indices = collect_frame_indices(ev_tracks)
    if canary_track_data:
        needed_indices |= collect_frame_indices([canary_track_data])
    print(f"[REQUIRED] {len(needed_indices)} unique frame indices across 22 tracks + canary")
    proposal_bundle = build_proposal_bundle(needed_indices)

    # 3. Create Gemini client
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("[FATAL] GEMINI_API_KEY not found in environment or .env")
        sys.exit(1)

    client = GeminiClient(api_key=api_key, model=GEMINI_MODEL)

    # 4. Run both variants
    print(f"\n{'#'*60}")
    print(f"# VARIANT 1: Single-crop (batch_size=1)")
    print(f"{'#'*60}")
    r1 = run_variant("single_crop", 1, ev_tracks, canary_track_data, proposal_bundle, client)

    print(f"\n{'#'*60}")
    print(f"# VARIANT 2: Separator-batch (batch_size=5)")
    print(f"{'#'*60}")
    r2 = run_variant("separator_batch", 5, ev_tracks, canary_track_data, proposal_bundle, client)

    # 5. Decision
    print("\n" + "=" * 60)
    print("FINAL DECISION — V29 E2E")
    print("=" * 60)
    single_pass = r1["status"] == "OK" and r1.get("recall", 0) >= 0.85 and r1["fp_reject_rate"] >= 0.75 and r1["t003_is_subtitle"] and r1["budget_ok"]
    batch_pass = r2["status"] == "OK" and r2.get("recall", 0) >= 0.85 and r2["fp_reject_rate"] >= 0.75 and r2["t003_is_subtitle"] and r2["budget_ok"]

    print(f"  Single-crop (batch_size=1):  {'PASS' if single_pass else 'FAIL'}")
    print(f"  Separator-batch (batch=5):   {'PASS' if batch_pass else 'FAIL'}")

    if single_pass and batch_pass:
        print("\n  ==> CHON: separator-batch (batch_size=5) for E2E — budget efficient")
        print("  ==> Will use <= 60 requests for full pipeline")
    elif single_pass and not batch_pass:
        print("\n  ==> CHON: single-crop (batch_size=1) for E2E — report true budget")
    elif not single_pass and batch_pass:
        print("\n  ==> CHON: separator-batch (batch_size=5) — only variant that passed")
    else:
        print("\n  ==> BLOCKED: NEITHER variant passed offline gate. Cannot proceed to E2E.")
        print("  ==> User must analyze per-track results to fix precision or recall")

    # Write final decision
    decision_path = ARTIFACT_DIR / "gate_decision_v29.json"
    decision = {
        "timestamp": time.time(),
        "single_crop_pass": single_pass,
        "separator_batch_pass": batch_pass,
        "recommendation": (
            "separator_batch" if batch_pass else ("single_crop" if single_pass else "BLOCKED")
        ),
        "single_crop": r1,
        "separator_batch": r2,
    }
    with open(decision_path, "w", encoding="utf-8") as f:
        json.dump(decision, f, indent=2, ensure_ascii=False)
    print(f"\n  Decision saved to {decision_path}")


if __name__ == "__main__":
    main()
