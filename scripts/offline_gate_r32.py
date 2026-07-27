#!/usr/bin/env python3
"""Offline gate v31 — send ALL 42 GT V2.3 entries through Gemini classifier.

Unlike r30 which only sent 14 positives, this script bypasses the band gate
and discriminator for the gate evaluation, sending every GT track to Gemini.

Changes from r30:
  - Three-tier reporting: A (local rules), B (Gemini on 42), C (combined).
  - ALL 42 tracks sent to Gemini regardless of band gate result.
  - Transparent retry logging with attempts_total vs tracks_total.
  - ONE-SHOT vs FINAL reporting.
  - Named results for T013, T036, T053, T126, T041, T105, T055.
  - Verifies bleed tracks from v29: T041, T105, T055.

Usage:
    .venv/Scripts/python scripts/offline_gate_r31.py

Exit:
    0 if PASS, 2 if FAIL (non-zero for CI integration).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.application.dto.proposal_bundle import ProposalBundle
from src.infrastructure.benchmark.hybrid_subtitle.crop_batch_classifier import CropBatchClassifierV21
from src.infrastructure.benchmark.hybrid_subtitle.crop_extractor import TrackCropExtractor
from src.infrastructure.gemini.client import GeminiClient
from src.infrastructure.gemini.contracts import GeminiRequest
from src.infrastructure.gemini.structured_output import parse_json
from src.infrastructure.benchmark.hybrid_subtitle.bottom_band_gate import passes_bottom_band, track_y1_norm
from src.infrastructure.benchmark.hybrid_subtitle.pixel_variance_discriminator import PixelVarianceDiscriminator

ARTIFACT_DIR = ROOT / "artifacts" / "round32"
GT_PATH = ROOT / "artifacts" / "round28" / "gt_v2_3.json"
BASELINE_PATH = ROOT / "artifacts" / "round23" / "baseline_tracks_conf_0.30.json"
VIDEO_PATH = ROOT / "test_videos" / "real_mini_benchmark" / "sample5.mp4"
GEMINI_MODEL = "gemini-3.5-flash-lite"
FRAME_HEIGHT = 1280

BAND_THRESHOLD = 0.68
VARIANCE_THRESHOLD = 827.12
POSITIVE_IDS = ["T003", "T006", "T010", "T014", "T016", "T019",
                "T116", "T120", "T127", "T133", "T135", "T137", "T138", "T146"]
BLEED_IDS = ["T041", "T105", "T055"]  # v29 bleed negatives
CANARY_ID = "T013"  # negative track from GT v2.3 (named negative:T013 in entries)


def load_key() -> str:
    for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
        if line.strip().startswith("GEMINI_API_KEY="):
            return line.split("=", 1)[1].strip().strip("'\"").split(",")[0].strip()
    raise RuntimeError("GEMINI_API_KEY unavailable")


def load_gt_entries() -> list[dict]:
    raw = json.loads(GT_PATH.read_text(encoding="utf-8"))
    return raw["entries"]


def load_baseline_tracks() -> dict[str, dict]:
    """Return {track_id: track} from baseline."""
    raw = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    return {t["track_id"]: t for t in raw}


def build_synthetic_track_from_gt(entry: dict, fps: float) -> dict:
    """Build a minimal track dict with groups for a GT entry.

    Creates ±3 frames around the GT timestamp with the GT bbox so the
    classifier can extract temporal crops.
    """
    eid = entry["id"]
    ts = entry["timestamp_s"]
    bbox = entry["bbox_xyxy"]
    center_fi = int(round(ts * fps))

    groups = []
    for offset in range(-3, 4):
        fi = max(0, center_fi + offset)
        ts_ms = int(round(fi / fps * 1000))
        groups.append({
            "frame_index": fi,
            "union_bbox_xyxy": [int(round(v)) for v in bbox],
            "timestamp_ms": ts_ms,
        })

    return {
        "track_id": eid,
        "groups": groups,
        "median_bbox_xyxy": [int(round(v)) for v in bbox],
        "temporal_union_bbox_xyxy": [int(round(v)) for v in bbox],
    }


def build_proposal_bundle(all_entries: list[dict], fps: float) -> ProposalBundle:
    """Decode all frames referenced by GT entries into a ProposalBundle."""
    needed_indices: set[int] = set()
    for entry in all_entries:
        ts = entry["timestamp_s"]
        center_fi = int(round(ts * fps))
        for offset in range(-6, 7):
            needed_indices.add(max(0, center_fi + offset))

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {VIDEO_PATH}")

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

    fi = 0
    while True:
        ret, img = cap.read()
        if not ret:
            break
        if fi in needed_indices:
            ts_sec = fi / fps
            bundle.add_frame(
                frame_index=fi,
                timestamp_sec=round(ts_sec, 4),
                image=img,
                image_path=None,
                detections=[],
            )
        fi += 1
    cap.release()
    print(f"[VIDEO] Decoded {len(bundle.frames)} frames (needed {len(needed_indices)}) from {VIDEO_PATH.name}")
    return bundle, needed_indices


def temporal_variance_from_gt(entry: dict, bundle: ProposalBundle, fps: float) -> float | None:
    """Match the r30 sweep metric using the immutable GT bbox and ±6 frames."""
    x1, y1, x2, y2 = [int(round(value)) for value in entry["bbox_xyxy"]]
    center = int(round(float(entry["timestamp_s"]) * fps))
    crops: list[np.ndarray] = []
    for frame_index in range(center - 6, center + 7):
        try:
            frame = bundle.frame_for_index(frame_index).image
        except KeyError:
            continue
        if frame is None:
            continue
        height, width = frame.shape[:2]
        cx1, cy1 = max(0, min(width - 1, x1)), max(0, min(height - 1, y1))
        cx2, cy2 = max(0, min(width, x2)), max(0, min(height, y2))
        if cx2 - cx1 < 2 or cy2 - cy1 < 2:
            continue
        crop = frame[cy1:cy2, cx1:cx2]
        crops.append(cv2.resize(crop, (96, 48), interpolation=cv2.INTER_AREA))
    if len(crops) < 2:
        return None
    return float(np.mean(np.var(np.stack(crops).astype(np.float32), axis=0)))


def evaluate_local_rules(entry: dict, track: dict) -> dict:
    """Apply band gate and discriminator, return pass/reject decisions."""
    result = {
        "id": entry["id"],
        "is_subtitle_gt": entry["is_subtitle"],
        "y1_norm": None,
        "band_pass": None,
        "temporal_variance": None,
        "discriminator_pass": None,
        "local_verdict_reject": None,
    }
    # Band gate
    bbox = entry.get("bbox_xyxy", [])
    if len(bbox) == 4:
        y1n = bbox[1] / FRAME_HEIGHT
        result["y1_norm"] = round(y1n, 4)
        result["band_pass"] = y1n >= BAND_THRESHOLD
    else:
        result["band_pass"] = None

    # Discriminator temporal variance - use the sweep approach
    # Actually for the gate report we already have discriminator_42.json data
    # But we compute it here to be self-contained
    result["temporal_variance"] = None
    result["discriminator_pass"] = None

    # Combined local verdict
    if result["band_pass"] is False:
        result["local_verdict_reject"] = True  # killed by band
    elif result["discriminator_pass"] is False:
        result["local_verdict_reject"] = True  # killed by discriminator
    else:
        result["local_verdict_reject"] = False  # passes both local tiers

    return result


def run_gemini_classifier(
    tracks: list[dict],
    proposal_bundle: ProposalBundle,
    canary_tracks: list[dict],
    classifier: CropBatchClassifierV21,
) -> tuple[dict[str, dict], int, int, list[str]]:
    """Run all primary requests first, then explicit retries with 4s pacing."""
    events_path = ARTIFACT_DIR / "gate_attempt_events_r32.jsonl"
    responses_dir = ARTIFACT_DIR / "gate_responses_r32"
    responses_dir.mkdir(parents=True, exist_ok=True)
    one_shot: dict[str, dict] = {}
    final: dict[str, dict] = {}
    attempts_total = 0
    retries = 0
    last_sent_at = 0.0
    pacing_seconds = 5.0
    primary_attempts = 0

    def attempt(track: dict, phase: str, attempt_no: int) -> dict | None:
        nonlocal attempts_total, last_sent_at
        tid = track["track_id"]
        if attempts_total >= 60:
            print(f"[BUDGET-BLOCKED] {tid}")
            return None
        evidence, _, _ = classifier._build_packet([track], proposal_bundle)
        prompt_parts = classifier._build_v21_prompt(evidence)
        if not prompt_parts:
            print(f"[{phase}] {tid}: no evidence packet")
            return None
        wait = pacing_seconds - (time.monotonic() - last_sent_at)
        if wait > 0:
            print(f"[PACING] waiting {wait:.1f}s before {tid}")
            time.sleep(wait)
        attempt_id = f"{tid.replace(':', '__')}_{phase.lower()}_{attempt_no}"
        started = {
            "event": "attempt_started", "attempt_id": attempt_id, "track_id": tid,
            "phase": phase, "attempt_no": attempt_no, "cache_policy": "disabled",
            "attempts_total_before": attempts_total,
        }
        with events_path.open("a", encoding="utf-8") as out:
            out.write(json.dumps(started) + "\n")
        attempts_total += 1
        last_sent_at = time.monotonic()
        try:
            response = classifier.client.generate_content(
                GeminiRequest(
                    contents=[{"parts": prompt_parts}], model=GEMINI_MODEL,
                    generation_config={"temperature": 0.1},
                    system_instruction="Classify on-screen dialogue subtitles. Default is NOT subtitle.",
                    request_id=f"r31_{attempt_id}", source_kind="round31_gate",
                    api_surface="benchmark",
                ),
                use_cache=False,
            )
            raw_path = responses_dir / f"{attempt_id}.txt"
            raw_path.write_text(response.text, encoding="utf-8")
            data = parse_json(response.text)
            if not isinstance(data, list):
                raise ValueError("response is not a JSON array")
            decision = next(
                (item for item in data if isinstance(item, dict)
                 and re.sub(
                     r"^Candidate\s+",
                     "",
                     item.get("candidate_id") or item.get("track_id") or "",
                 ) == tid),
                None,
            )
            if not isinstance(decision, dict) or not isinstance(decision.get("is_subtitle"), bool):
                raise ValueError("missing or invalid target decision")
            finished = {
                "event": "attempt_finished", "attempt_id": attempt_id, "track_id": tid,
                "phase": phase, "outcome": "accepted", "cache_hit": bool(response.cache_hit),
                "tokens": response.usage.to_dict(), "response_path": str(raw_path),
                "decision": decision,
            }
            with events_path.open("a", encoding="utf-8") as out:
                out.write(json.dumps(finished, ensure_ascii=False) + "\n")
            return decision
        except Exception as exc:
            retry_after = getattr(exc, "retry_after", None)
            finished = {
                "event": "attempt_finished", "attempt_id": attempt_id, "track_id": tid,
                "phase": phase, "outcome": "invalid_or_missing_decision",
                "error_type": type(exc).__name__, "error": str(exc),
                "retry_after": retry_after,
            }
            with events_path.open("a", encoding="utf-8") as out:
                out.write(json.dumps(finished, ensure_ascii=False) + "\n")
            print(f"[{phase}] {tid}: {type(exc).__name__}: {exc}")
            if retry_after is not None:
                backoff = min(float(retry_after) * (1.5 ** (attempt_no - 1)), 60.0)
                print(f"[BACKOFF] {tid}: waiting {backoff:.1f}s")
                time.sleep(backoff)
            return None

    print(f"PHASE1 primary_sent=0/{len(tracks)} pacing_seconds={pacing_seconds}")
    for track in tracks:
        decision = attempt(track, "ONE_SHOT", 1)
        primary_attempts += 1
        if decision is not None:
            one_shot[track["track_id"]] = decision
            final[track["track_id"]] = decision
        print(f"PHASE1 primary_sent={primary_attempts}/{len(tracks)} resolved={len(one_shot)}")

    missing_ids = [track["track_id"] for track in tracks if track["track_id"] not in one_shot]
    print(f"PHASE1 complete primary_sent={len(tracks)}/{len(tracks)} unresolved={len(missing_ids)}")
    print(f"PHASE2 retries={len(missing_ids)}")
    for tid in missing_ids:
        track = next(track for track in tracks if track["track_id"] == tid)
        retries += 1
        print(f"[RETRY] {tid}")
        decision = attempt(track, "FINAL", 2)
        if decision is not None:
            final[tid] = decision
    unresolved = [tid for tid in missing_ids if tid not in final]
    return final, attempts_total, retries, missing_ids


def main() -> None:
    artifact_dir = ARTIFACT_DIR
    artifact_dir.mkdir(parents=True, exist_ok=True)

    # ── Git SHA ──────────────────────────────────────────────────
    git_sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=str(ROOT), text=True
    ).strip()

    print("=" * 70)
    print("OFFLINE GATE R32 — ALL 42 GT V2.3 tracks → Gemini")
    print(f"GIT_SHA:        {git_sha}")
    print(f"GT V2.3 SHA:    99d1d75e04226c5b7946d3b24532167f6875da73be9ff69f40fbb2a8f3940bd5")
    print(f"Video:          {VIDEO_PATH.name}")
    print(f"Baseline:       {BASELINE_PATH.name}")
    print(f"Band threshold: {BAND_THRESHOLD}")
    print(f"Var threshold:  {VARIANCE_THRESHOLD}")
    print(f"Gemini model:   {GEMINI_MODEL}")
    print(f"Budget limit:   60")
    print("=" * 70)

    # ── Log start time ───────────────────────────────────────────
    smi = subprocess.run(["nvidia-smi"], capture_output=True, text=True, check=False)
    print(smi.stdout, end="")

    # ── Load GT ──────────────────────────────────────────────────
    entries = load_gt_entries()
    print(f"[GT] Loaded {len(entries)} entries")

    # Build ID → GT entry map
    gt_map: dict[str, dict] = {}
    for e in entries:
        gt_map[e["id"]] = e

    # ── Load baseline tracks ─────────────────────────────────────
    baseline_tracks = load_baseline_tracks()
    print(f"[BASELINE] Loaded {len(baseline_tracks)} baseline tracks")

    # ── Build tracks for ALL 42 GT entries ───────────────────────
    fps = 30.0
    all_tracks: list[dict] = []
    missing_tracks: list[str] = []
    for entry in entries:
        eid = entry["id"]
        if eid in baseline_tracks:
            all_tracks.append(baseline_tracks[eid])
        else:
            # Build synthetic track from GT data
            all_tracks.append(build_synthetic_track_from_gt(entry, fps))
            missing_tracks.append(eid)

    if missing_tracks:
        print(f"[WARN] {len(missing_tracks)} tracks not in baseline, built synthetic: {missing_tracks}")

    # ── Build proposal bundle ────────────────────────────────────
    bundle, _ = build_proposal_bundle(entries, fps)

    # ── LAYER A: Local rules (band gate + discriminator) ───────────
    print("\n" + "=" * 70)
    print("LAYER A — Local Rule Results (band gate + discriminator)")
    print("=" * 70)

    local_results: dict[str, dict] = {}
    for entry in entries:
        eid = entry["id"]
        bbox = entry.get("bbox_xyxy", [])
        y1n = bbox[1] / FRAME_HEIGHT if len(bbox) == 4 else None
        band_pass = y1n is not None and y1n >= BAND_THRESHOLD if y1n is not None else None
        local_results[eid] = {
            "y1_norm": round(y1n, 4) if y1n is not None else None,
            "band_pass": band_pass,
        }

    # Layer A uses immutable GT crops. It must not depend on baseline track IDs.
    for entry in entries:
        eid = entry["id"]
        variance = temporal_variance_from_gt(entry, bundle, fps)
        discriminator_rejected = variance is None or variance < VARIANCE_THRESHOLD
        local_results[eid].update({
            "temporal_variance": round(variance, 4) if variance is not None else None,
            "discriminator_rejected": discriminator_rejected,
            "discriminator_reason": (
                "insufficient_temporal_crops" if variance is None
                else "low_temporal_pixel_variance" if discriminator_rejected
                else "passed"
            ),
        })
        if discriminator_rejected:
            local_results[eid]["local_verdict"] = "REJECT"
        elif local_results[eid].get("band_pass") is False:
            local_results[eid]["local_verdict"] = "REJECT"
        else:
            local_results[eid]["local_verdict"] = "PASS"

    # Print Layer A table
    print(f"{'ID':<22} {'GT_sub':<10} {'y1_norm':<10} {'band':<8} {'var':<10} {'disc':<10} {'verdict'}")
    print("-" * 70)
    a_reject = 0
    for eid in sorted(local_results.keys()):
        r = local_results[eid]
        gt_lbl = "POS" if gt_map[eid]["is_subtitle"] else "NEG"
        b = "PASS" if r.get("band_pass") else ("REJ" if r.get("band_pass") is False else "N/A")
        d = "REJ" if r.get("discriminator_rejected") else "PASS"
        v = r.get("temporal_variance", "N/A")
        v_str = f"{v:.1f}" if isinstance(v, (int, float)) else "N/A"
        vrd = r.get("local_verdict", "?")
        if vrd == "REJECT":
            a_reject += 1
        print(f"  {eid:<20} {gt_lbl:<10} {r['y1_norm'] or 'N/A':<10} {b:<8} {v_str:<10} {d:<10} {vrd}")

    print(f"\nLayer A summary: {a_reject} rejected locally, "
          f"{sum(1 for e in entries if not e['is_subtitle']) - a_reject} false negatives still alive")

    # ── LAYER B: Gemini on ALL 42 ───────────────────────────────────
    print("\n" + "=" * 70)
    print("LAYER B — Gemini Classifier on ALL 42 GT V2.3 tracks")
    print("=" * 70)

    client = GeminiClient(api_key=load_key(), model=GEMINI_MODEL)
    classifier = CropBatchClassifierV21(
        client, model=GEMINI_MODEL, budget_limit=60,
        artifact_dir=artifact_dir, job_id="round32-offline", run_id="all42",
    )

    # Build canary track for T013
    canary_tracks_list = []
    canary_id_canonical = CANARY_ID  # GT entry uses "T013" with is_subtitle=false
    if canary_id_canonical in baseline_tracks:
        canary_tracks_list = [baseline_tracks[canary_id_canonical]]
    else:
        entry = gt_map.get(canary_id_canonical)
        if entry:
            canary_tracks_list = [build_synthetic_track_from_gt(entry, fps)]
    print(f"[CANARY] Negative track for canary: {CANARY_ID}")

    # Send ALL 42 tracks to Gemini
    start_b = time.monotonic()
    final_decisions, attempts_total, retry_count, missing_ids = run_gemini_classifier(
        all_tracks, bundle, canary_tracks_list, classifier
    )
    elapsed_b = time.monotonic() - start_b

    print(f"\nLayer B timing: {elapsed_b:.1f}s total")
    print(f"attempts_total={attempts_total}, tracks_total={len(all_tracks)}, retries={retry_count}")

    # ── Layer B confusion matrix ─────────────────────────────────
    b_tp = 0
    b_fp = 0
    b_tn = 0
    b_fn = 0
    b_unknown = 0
    for entry in entries:
        eid = entry["id"]
        is_sub_gt = entry["is_subtitle"]
        dec = final_decisions.get(eid, {})
        if not dec or dec.get("is_subtitle") is None:
            b_unknown += 1
            continue
        is_sub_pred = dec.get("is_subtitle") is True
        if is_sub_gt and is_sub_pred:
            b_tp += 1
        elif is_sub_gt and not is_sub_pred:
            b_fn += 1
        elif not is_sub_gt and is_sub_pred:
            b_fp += 1
        elif not is_sub_gt and not is_sub_pred:
            b_tn += 1

    print(f"\nLayer B confusion matrix:")
    print(f"  TP={b_tp}  FP={b_fp}")
    print(f"  FN={b_fn}  TN={b_tn}  unknown={b_unknown}")
    precision_b = b_tp / (b_tp + b_fp) if (b_tp + b_fp) > 0 else 0.0
    recall_b = b_tp / (b_tp + b_fn) if (b_tp + b_fn) > 0 else 0.0
    print(f"  precision={precision_b:.4f}  recall={recall_b:.4f}")

    # ── Named tracks ─────────────────────────────────────────────
    named_ids = [CANARY_ID, "T036", "T053", "T126"] + BLEED_IDS
    print(f"\nNamed track results:")
    print(f"{'ID':<10} {'GT':<8} {'Gemini':<8} {'Band':<8} {'Disc':<10} {'echo'}")
    for nid in named_ids:
        if nid in gt_map:
            gt_lbl = "sub" if gt_map[nid]["is_subtitle"] else "NOSUB"
            dec = final_decisions.get(nid, {})
            gem_lbl = "YES" if dec.get("is_subtitle") is True else "NO"
            echo = str(dec.get("text_echo", ""))[:30]
            lr = local_results.get(nid, {})
            b = "P" if lr.get("band_pass") else "R"
            d = "P" if not lr.get("discriminator_rejected") else "R"
            print(f"  {nid:<8} {gt_lbl:<8} {gem_lbl:<8} {b:<8} {d:<10} {echo}")
        elif nid in baseline_tracks:
            # Some BLEED_IDS like T041 might be standard negative IDs
            gt_lbl = "NOSUB"  # known negative
            dec = final_decisions.get(nid, {})
            gem_lbl = "YES" if dec.get("is_subtitle") is True else "NO"
            echo = str(dec.get("text_echo", ""))[:30]
            lr = local_results.get(nid, {})
            b = "P" if lr.get("band_pass") else "R"
            d = "P" if not lr.get("discriminator_rejected") else "R"
            print(f"  {nid:<8} {gt_lbl:<8} {gem_lbl:<8} {b:<8} {d:<10} {echo}")
        else:
            print(f"  {nid:<8} — not found in GT v2.3")

    # ── LAYER C: Combined pipeline ────────────────────────────────
    print("\n" + "=" * 70)
    print("LAYER C — Combined Pipeline (local rules + Gemini)")
    print("=" * 70)

    c_tp = 0
    c_fp = 0
    c_tn = 0
    c_fn = 0
    for entry in entries:
        eid = entry["id"]
        is_sub_gt = entry["is_subtitle"]
        lr = local_results.get(eid, {})
        band_pass = lr.get("band_pass")
        disc_reject = lr.get("discriminator_rejected", True)

        # Local kills first
        if band_pass is False or disc_reject:
            combined_is_sub = False
        else:
            dec = final_decisions.get(eid, {})
            combined_is_sub = dec.get("is_subtitle") is True if dec else False

        if is_sub_gt and combined_is_sub:
            c_tp += 1
        elif is_sub_gt and not combined_is_sub:
            c_fn += 1
        elif not is_sub_gt and combined_is_sub:
            c_fp += 1
        elif not is_sub_gt and not combined_is_sub:
            c_tn += 1

    print(f"  TP={c_tp}  FP={c_fp}")
    print(f"  FN={c_fn}  TN={c_tn}")
    precision_c = c_tp / (c_tp + c_fp) if (c_tp + c_fp) > 0 else 0.0
    recall_c = c_tp / (c_tp + c_fn) if (c_tp + c_fn) > 0 else 0.0
    print(f"  precision={precision_c:.4f}  recall={recall_c:.4f}")

    # ── ONE-SHOT gate status ──────────────────────────────────────
    one_shot_fail = bool(missing_ids)
    final_fail = (
        bool(b_unknown)
        or c_fn > 0
        or attempts_total > 60
    )

    print(f"\n{'='*70}")
    print(f"ONE-SHOT: {'FAIL' if one_shot_fail else 'PASS'} "
          f"(missing decisions: {missing_ids})")
    print(f"FINAL:    {'FAIL' if final_fail else 'PASS'} "
          f"(FN={c_fn}, FP={c_fp})")
    print(f"attempts_total={attempts_total}  tracks_total={len(all_tracks)}  retries={retry_count}")

    # ── Write report ─────────────────────────────────────────────
    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": git_sha,
        "gt_sha256": "99d1d75e04226c5b7946d3b24532167f6875da73be9ff69f40fbb2a8f3940bd5",
        "layer_a": {
            "local_results": local_results,
            "rejected_local": a_reject,
        },
        "layer_b": {
            "confusion_matrix": {"tp": b_tp, "fp": b_fp, "tn": b_tn, "fn": b_fn, "unknown": b_unknown},
            "precision": round(precision_b, 4),
            "recall": round(recall_b, 4),
            "total_tracks": len(all_tracks),
            "attempts_total": attempts_total,
            "retry_count": retry_count,
        },
        "layer_c": {
            "confusion_matrix": {"tp": c_tp, "fp": c_fp, "tn": c_tn, "fn": c_fn},
            "precision": round(precision_c, 4),
            "recall": round(recall_c, 4),
        },
        "named_tracks": {nid: {
            "gt_is_subtitle": gt_map.get(nid, {}).get("is_subtitle"),
            "gemini_is_subtitle": final_decisions.get(nid, {}).get("is_subtitle"),
            "text_echo": str(final_decisions.get(nid, {}).get("text_echo", "")),
            "band_pass": local_results.get(nid, {}).get("band_pass"),
            "discriminator_rejected": local_results.get(nid, {}).get("discriminator_rejected"),
        } for nid in named_ids},
        "one_shot": {
            "missing_ids": missing_ids,
            "status": "FAIL" if one_shot_fail else "PASS",
        },
        "final_status": "FAIL" if final_fail else "PASS",
    }

    report_path = artifact_dir / "offline_gate_r32.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[REPORT] Written to {report_path}")
    print("GATE:", report["final_status"])
    raise SystemExit(0 if report["final_status"] == "PASS" else 2)


if __name__ == "__main__":
    main()
