"""Evaluate Round 23 confirmed subtitle regions against frozen GT or GT V2."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "round23"


def _iou(left: list[float], right: list[float]) -> float:
    x1, y1 = max(left[0], right[0]), max(left[1], right[1])
    x2, y2 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_left = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    area_right = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = area_left + area_right - intersection
    return intersection / union if union else 0.0


def _region_box(region: dict[str, Any], width: int, height: int) -> list[float]:
    return [
        region["x"] * width,
        region["y"] * height,
        (region["x"] + region["width"]) * width,
        (region["y"] + region["height"]) * height,
    ]


def _frozen_entries() -> tuple[list[dict[str, Any]], str]:
    sidecar = json.loads((OUT / "gt_mask_geometry.json").read_text(encoding="utf-8"))
    labels = {
        item["id"]: item["expected_is_subtitle"]
        for item in json.loads(
            (ROOT / "scripts" / "frozen_gt" / "sample5" / "frozen_gt.json").read_text(
                encoding="utf-8"
            )
        )
    }
    entries = [
        {
            "id": item["track_id"],
            "is_subtitle": labels[item["track_id"]],
            "start_time": item["start_time_s"],
            "end_time": item["end_time_s"],
            "bbox_xyxy": [item["bbox_xyxy"][0] + 8, item["bbox_xyxy"][1] + 8,
                          item["bbox_xyxy"][2] - 8, item["bbox_xyxy"][3] - 8],
        }
        for item in sidecar["tracks"]
    ]
    return entries, sidecar["frozen_gt_sha256"]


def _v2_entries(kind: str = "v2") -> tuple[list[dict[str, Any]], str]:
    target = ROOT / "artifacts" / ("round24" if kind == "v2_1" else "round23") / (
        "gt_v2_1.json" if kind == "v2_1" else "gt_v2.json"
    )
    payload = json.loads(target.read_text(encoding="utf-8"))
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    return [
        {
            "id": item["id"],
            "is_subtitle": item["is_subtitle"],
            "start_time": item["timestamp_s"] - 0.25,
            "end_time": item["timestamp_s"] + 0.25,
            "bbox_xyxy": item["bbox_xyxy"],
            "set": item["set"],
        }
        for item in payload["entries"]
    ], digest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--regions", required=True, type=Path)
    parser.add_argument("--kind", choices=("frozen", "v2", "v2_1"), required=True)
    parser.add_argument("--width", required=True, type=int)
    parser.add_argument("--height", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    regions = json.loads(args.regions.read_text(encoding="utf-8"))
    entries, gt_sha256 = (
        _frozen_entries()
        if args.kind == "frozen"
        else _v2_entries(args.kind)
    )
    by_track: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for region in regions:
        by_track[region["track_id"]].append(region)

    edges: list[tuple[float, str, str]] = []
    for track_id, track_regions in by_track.items():
        for entry in entries:
            for region in track_regions:
                if region["end_time"] < entry["start_time"] or region["start_time"] > entry["end_time"]:
                    continue
                score = _iou(_region_box(region, args.width, args.height), entry["bbox_xyxy"])
                if score >= 0.5:
                    edges.append((score, entry["id"], track_id))

    matched_gt: dict[str, str] = {}
    matched_predictions: set[str] = set()
    for _, entry_id, track_id in sorted(edges, reverse=True):
        if entry_id in matched_gt or track_id in matched_predictions:
            continue
        matched_gt[entry_id] = track_id
        matched_predictions.add(track_id)
    unmatched_predictions = set(by_track) - matched_predictions

    positives = {entry["id"] for entry in entries if entry["is_subtitle"]}
    true_positive_ids = positives & set(matched_gt)
    false_positive_ids = {
        track_id
        for entry_id, track_id in matched_gt.items()
        if entry_id not in positives
    } | unmatched_predictions
    true_positives = len(true_positive_ids)
    false_positives = len(false_positive_ids)
    false_negatives = len(positives - true_positive_ids)
    precision = true_positives / (true_positives + false_positives) if true_positives + false_positives else 0.0
    recall = true_positives / len(positives) if positives else 0.0

    result = {
        "kind": args.kind,
        "gt_sha256": gt_sha256,
        "iou_threshold": 0.5,
        "confirmed_prediction_tracks": len(by_track),
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "matched_gt": matched_gt,
        "unmatched_prediction_tracks": sorted(unmatched_predictions),
    }
    if args.kind in ("v2", "v2_1"):
        moving_ids = {entry["id"] for entry in entries if entry.get("set") == "moving"}
        result["moving_hit_rate"] = round(
            len(moving_ids & true_positive_ids) / len(moving_ids), 4
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
