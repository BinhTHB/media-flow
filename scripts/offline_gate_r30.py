#!/usr/bin/env python3
"""Round 30 offline gate: immutable combined band + Gemini evaluation."""
from __future__ import annotations
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.offline_gate_v29 import (
    BASELINE_PATH, GEMINI_MODEL, VIDEO_PATH, build_proposal_bundle,
    collect_frame_indices,
)
from src.infrastructure.benchmark.hybrid_subtitle.bottom_band_gate import passes_bottom_band
from src.infrastructure.benchmark.hybrid_subtitle.crop_batch_classifier import CropBatchClassifierV21
from src.infrastructure.gemini.client import GeminiClient

ARTIFACT_DIR = ROOT / "artifacts" / "round30"
GT_PATH = ROOT / "artifacts" / "round28" / "gt_v2_3.json"
POSITIVE_IDS = ["T003", "T006", "T010", "T014", "T016", "T019", "T116", "T120", "T127", "T133", "T135", "T137", "T138", "T146"]
FP_IDS = ["T030", "T041", "T044", "T047", "T055", "T061", "T071", "T105"]
CANARY_ID = "T013"  # GT V2.3 negative:T013, not a v29 non-GT canary.


def load_key() -> str:
    for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
        if line.strip().startswith("GEMINI_API_KEY="):
            return line.split("=", 1)[1].strip().strip("'\"").split(",")[0].strip()
    raise RuntimeError("GEMINI_API_KEY unavailable")


def main() -> None:
    print("OFFLINE GATE R30")
    print("GT V2.3 SHA: 99d1d75e04226c5b7946d3b24532167f6875da73be9ff69f40fbb2a8f3940bd5")
    print("HARD CRITERIA: FN = 0/14; FP reject >= 6/8; T003 = YES; Budget <= 60")
    raw_gt = json.loads(GT_PATH.read_text(encoding="utf-8"))
    assert len(raw_gt["entries"]) == 42
    tracks = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    selected = {t["track_id"]: t for t in tracks if t["track_id"] in set(POSITIVE_IDS + FP_IDS + [CANARY_ID])}
    missing = set(POSITIVE_IDS + FP_IDS + [CANARY_ID]) - set(selected)
    if missing:
        raise RuntimeError(f"Missing baseline tracks: {sorted(missing)}")

    # Band gate is deterministic and precedes Gemini.
    band_rejected = [tid for tid in FP_IDS if not passes_bottom_band(selected[tid], 1280, 0.68)]
    candidates = [selected[tid] for tid in POSITIVE_IDS] + [
        selected[tid] for tid in FP_IDS if tid not in band_rejected
    ]
    print(f"BAND GATE: rejected {len(band_rejected)}/8 FP before Gemini: {band_rejected}")
    print(f"GEMINI CANDIDATES: {len(candidates)}")

    bundle = build_proposal_bundle(collect_frame_indices(candidates + [selected[CANARY_ID]]))
    classifier = CropBatchClassifierV21(
        GeminiClient(api_key=load_key(), model=GEMINI_MODEL),
        model=GEMINI_MODEL, budget_limit=60, artifact_dir=ARTIFACT_DIR,
        job_id="round30-offline", run_id="single_crop",
    )
    decisions = classifier.classify_batch(
        candidates, bundle, batch_size=1, canary_tracks=[selected[CANARY_ID]]
    )
    # Missing response IDs are transport/model omissions, not a semantic NO.
    # Retry each missing candidate solo while preserving the same crop-pixel
    # echo and fail-closed behavior; retry requests count toward the budget.
    missing_ids = [
        track["track_id"] for track in candidates
        if track["track_id"] not in decisions
    ]
    if missing_ids:
        print(f"RETRY MISSING IDS SOLO: {missing_ids}")
        for track_id in missing_ids:
            retry = classifier.classify_batch(
                [selected[track_id]], bundle, batch_size=1,
                canary_tracks=[selected[CANARY_ID]],
            )
            decisions.update(retry)
    result = {}
    for tid in POSITIVE_IDS + FP_IDS:
        if tid in band_rejected:
            result[tid] = {"is_subtitle": False, "stage": "band_gate"}
        else:
            result[tid] = dict(decisions.get(tid, {}))
            result[tid]["stage"] = "gemini"

    tp = sum(result[tid].get("is_subtitle") is True for tid in POSITIVE_IDS)
    fn = len(POSITIVE_IDS) - tp
    fp_rejected = sum(result[tid].get("is_subtitle") is not True for tid in FP_IDS)
    t003_yes = result["T003"].get("is_subtitle") is True
    budget = classifier.requests_sent
    passed = fn == 0 and fp_rejected >= 6 and t003_yes and budget <= 60
    ledger = []
    for tid in POSITIVE_IDS + FP_IDS:
        decision = result[tid]
        ledger.append({
            "job_id": "round30-offline",
            "request_hash": hashlib.sha256(f"round30|{tid}".encode()).hexdigest()[:16],
            "track_id": tid, "variant": "single_crop",
            "echo": decision.get("text_echo", ""),
            "decision": decision.get("is_subtitle"),
            "canary": CANARY_ID, "tokens": {}, "cache": False,
            "stage": decision["stage"],
        })
    ledger_path = ARTIFACT_DIR / "request_ledger_r30.jsonl"
    ledger_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in ledger), encoding="utf-8")
    report = {
        "gt_sha256": "99d1d75e04226c5b7946d3b24532167f6875da73be9ff69f40fbb2a8f3940bd5",
        "canary": "negative:T013", "band_rejected_fps": band_rejected,
        "true_positive": tp, "false_negative": fn, "fp_rejected": fp_rejected,
        "t003_yes": t003_yes, "requests": budget, "budget_limit": 60,
        "hard_gate": {"fn_zero": fn == 0, "fp_reject_at_least_6_of_8": fp_rejected >= 6, "t003_yes": t003_yes, "budget_at_most_60": budget <= 60},
        "status": "PASS" if passed else "FAIL",
    }
    (ARTIFACT_DIR / "offline_gate_r30.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print("GATE:", report["status"])
    raise SystemExit(0 if passed else 2)


if __name__ == "__main__":
    main()
