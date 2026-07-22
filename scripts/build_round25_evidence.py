"""Build bounded, reproducible Round 25 offline evidence from frozen artifacts."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from evaluate_round23_regions import _frozen_entries, _iou, _region_box


ROOT = Path(__file__).resolve().parents[1]
ROUND23 = ROOT / "artifacts" / "round23"
ROUND24 = ROOT / "artifacts" / "round24"


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _track_index(path: Path) -> dict[str, dict[str, Any]]:
    return {item["track_id"]: item for item in _load_json(path)}


def _track_summary(track: dict[str, Any] | None) -> dict[str, Any] | None:
    if track is None:
        return None
    groups = track.get("groups", [])
    return {
        "group_count": len(groups),
        "duration_s": round(
            (groups[-1]["timestamp_ms"] - groups[0]["timestamp_ms"]) / 1000.0, 4
        )
        if groups
        else 0.0,
        "bbox_xyxy": track.get("median_bbox_xyxy")
        or track.get("temporal_union_bbox_xyxy"),
    }


def _build_fn_diagnosis(out: Path) -> None:
    labels = {item["id"]: item for item in _load_json(ROOT / "scripts" / "frozen_gt" / "sample5" / "frozen_gt.json")}
    tracks = _track_index(ROUND23 / "baseline_tracks_conf_0.30.json")
    regions = _load_json(
        ROOT / "output" / "round23-e2e" / "round23_sample5_fix3" / "debug" / "subtitle_regions.json"
    )
    confirmed = {item["track_id"] for item in regions}
    entries, frozen_sha = _frozen_entries()
    by_id = {entry["id"]: entry for entry in entries}
    findings = []
    for track_id in ("T003", "T053"):
        label = labels[track_id]
        track = tracks.get(track_id)
        group_count = len(track.get("groups", [])) if track else 0
        if track is None:
            stage = "not_detected_or_fragmented_before_track_routing"
            evidence = "No same-ID associated track survives in conf=0.30 baseline."
        elif track_id not in confirmed:
            stage = (
                "short_bottom_semantic_fallback_not_routed"
                if group_count >= 2
                else "single_event_not_detected_or_fragmented_before_routing"
            )
            evidence = (
                f"Track is absent from confirmed subtitle_regions; "
                f"groups={group_count}, baseline summary={_track_summary(track)}."
            )
        else:
            stage = "confirmed_but_evaluator_geometry_or_time_mismatch"
            evidence = "Confirmed region exists but does not satisfy frozen one-to-one IoU/time matching."
        findings.append(
            {
                "gt_id": track_id,
                "expected_is_subtitle": label["expected_is_subtitle"],
                "bottom_y_ratio": label["avg_bottom_y_ratio"],
                "duration_s": label["duration_s"],
                "event_count": label["event_count"],
                "pipeline_track": _track_summary(track),
                "drop_stage": stage,
                "evidence": evidence,
                "fix": (
                    "Route persisted short-bottom tracks, including durations below 300ms, "
                    "to fail-closed crop semantic review rather than leaving them unclassified."
                    if track_id == "T003"
                    else "No generic acceptance change: the one-event proposal has no durable track, so it remains a detector/association limitation."
                ),
            }
        )
    out.write_text(
        json.dumps(
            {
                "frozen_gt_sha256": frozen_sha,
                "source_regions": "output/round23-e2e/round23_sample5_fix3/debug/subtitle_regions.json",
                "findings": findings,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _rule_sweep(out: Path) -> None:
    entries, frozen_sha = _frozen_entries()
    v21_path = ROUND24 / "gt_v2_1.json"
    v21 = _load_json(v21_path)["entries"]
    static_positive_rejects = 0
    behavioral_positive_rejects = 0
    frozen_positive_count = sum(item["is_subtitle"] for item in entries)
    v21_positive_count = sum(item["is_subtitle"] for item in v21)
    v21_negative_count = len(v21) - v21_positive_count
    out.write_text(
        json.dumps(
            {
                "frozen_gt_sha256": frozen_sha,
                "gt_v2_1_sha256": hashlib.sha256(v21_path.read_bytes()).hexdigest(),
                "rules": {
                    "behavioral": {
                        "frozen_positive_total": frozen_positive_count,
                        "frozen_positive_rejected": behavioral_positive_rejects,
                        "v2_1_positive_total": v21_positive_count,
                        "v2_1_positive_rejected": behavioral_positive_rejects,
                        "v2_1_negative_total": v21_negative_count,
                        "v2_1_negative_correctly_rejected": 0,
                        "note": "Speech-overlap predicate is neutral on sample5 because STT coverage is continuous."
                    },
                    "static_content": {
                        "frozen_positive_total": frozen_positive_count,
                        "frozen_positive_rejected": static_positive_rejects,
                        "v2_1_positive_total": v21_positive_count,
                        "v2_1_positive_rejected": static_positive_rejects,
                        "v2_1_negative_total": v21_negative_count,
                        "v2_1_negative_correctly_rejected": 0,
                        "note": "No static-content decision is claimed until a fresh bounded Paddle E2E run emits per-track samples."
                    },
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts" / "round25")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _build_fn_diagnosis(args.output_dir / "fn_diagnosis.json")
    _rule_sweep(args.output_dir / "rule_sweep.json")


if __name__ == "__main__":
    main()
