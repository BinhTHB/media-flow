"""Diagnose frozen-GT false negatives and false positives from Round 23 E2E."""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.infrastructure.benchmark.hybrid_subtitle.core import build_tracks, group_frame_candidates

OUT = ROOT / "artifacts" / "round23"

def iou(left: list[float], right: list[float]) -> float:
    x1, y1 = max(left[0], right[0]), max(left[1], right[1])
    x2, y2 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (
        (left[2] - left[0]) * (left[3] - left[1])
        + (right[2] - right[0]) * (right[3] - right[1])
        - intersection
    )
    return intersection / union if union else 0.0


def unpad(mask_bbox: list[float]) -> list[float]:
    return [mask_bbox[0] + 8, mask_bbox[1] + 8, mask_bbox[2] - 8, mask_bbox[3] - 8]


def build_e2e_tracks() -> list[dict[str, Any]]:
    detections = [
        json.loads(line)
        for line in (OUT / "baseline_raw_yolo_detections.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
        if line
    ]
    by_frame: dict[int, list[dict[str, Any]]] = {}
    for detection in detections:
        if float(detection["confidence"]) >= 0.45:
            by_frame.setdefault(int(detection["frame_index"]), []).append(detection)
    groups = [
        group
        for frame_index in sorted(by_frame)
        for group in group_frame_candidates(by_frame[frame_index])
    ]
    return build_tracks(groups, iou_threshold=0.25, max_gap_frames=5)


def best_track_for_gt(gt: dict[str, Any], tracks: list[dict[str, Any]]) -> dict[str, Any] | None:
    target = unpad(gt["bbox_xyxy"])
    best: tuple[float, dict[str, Any] | None] = (0.0, None)
    duration = max(gt["end_time_s"] - gt["start_time_s"], 0.001)
    for track in tracks:
        start = max(0.0, track["start_ms"] / 1000.0 - 0.08)
        end = track["end_ms"] / 1000.0 + 0.08
        overlap = max(0.0, min(end, gt["end_time_s"]) - max(start, gt["start_time_s"]))
        if overlap / duration < 0.5:
            continue
        scores = [
            iou(target, group["union_bbox_xyxy"])
            for group in track["groups"]
            if gt["start_time_s"] <= group["timestamp_ms"] / 1000.0 <= gt["end_time_s"]
        ]
        if scores and max(scores) > best[0]:
            best = (max(scores), track)
    return best[1] if best[0] >= 0.5 else None


def track_summary(track: dict[str, Any] | None) -> dict[str, Any] | None:
    if track is None:
        return None
    return {
        "track_id": track["track_id"],
        "time_range_s": [round(track["start_ms"] / 1000.0, 4), round(track["end_ms"] / 1000.0, 4)],
        "bbox_xyxy": [round(value, 2) for value in track["median_bbox_xyxy"]],
        "groups": len(track["groups"]),
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--regions",
        type=Path,
        default=ROOT / "output" / "round23-e2e" / "round23_sample5" / "debug" / "subtitle_regions.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUT / "e2e_frozen_gt_diagnosis.json",
    )
    args = parser.parse_args()
    labels = {
        item["id"]: item["expected_is_subtitle"]
        for item in json.loads(
            (ROOT / "scripts" / "frozen_gt" / "sample5" / "frozen_gt.json").read_text(
                encoding="utf-8"
            )
        )
    }
    gt_tracks = json.loads((OUT / "gt_mask_geometry.json").read_text(encoding="utf-8"))["tracks"]
    tracks = build_e2e_tracks()
    regions = json.loads(args.regions.read_text(encoding="utf-8"))
    confirmed = {region["track_id"] for region in regions}
    gt_to_track = {gt["track_id"]: best_track_for_gt(gt, tracks) for gt in gt_tracks}
    track_to_gt = {
        track["track_id"]: gt
        for gt, track in ((item, gt_to_track[item["track_id"]]) for item in gt_tracks)
        if track is not None
    }

    false_negatives = []
    false_positives = []
    for gt in gt_tracks:
        track = gt_to_track[gt["track_id"]]
        is_confirmed = track is not None and track["track_id"] in confirmed
        row = {
            "gt_id": gt["track_id"],
            "gt_bbox_xyxy": unpad(gt["bbox_xyxy"]),
            "gt_time_range_s": [gt["start_time_s"], gt["end_time_s"]],
            "pipeline_track": track_summary(track),
            "confirmed": is_confirmed,
        }
        if labels[gt["track_id"]] and not is_confirmed:
            false_negatives.append(row)
        if not labels[gt["track_id"]] and is_confirmed:
            false_positives.append(row)

    grouped_regions: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for region in regions:
        grouped_regions[region["track_id"]].append(region)
    unmatched_confirmed = [
        {
            "track_id": track_id,
            "time_range_s": [
                round(min(item["start_time"] for item in group), 4),
                round(max(item["end_time"] for item in group), 4),
            ],
            "bbox_xyxy": [
                round(group[0]["x"] * 720, 2),
                round(group[0]["y"] * 1280, 2),
                round((group[0]["x"] + group[0]["width"]) * 720, 2),
                round((group[0]["y"] + group[0]["height"]) * 1280, 2),
            ],
            "model": group[0]["classifier_model"],
        }
        for track_id, group in grouped_regions.items()
        if track_id not in track_to_gt
    ]

    pairing = []
    for fp in false_positives:
        fp_track = gt_to_track[fp["gt_id"]]
        for fn in false_negatives:
            fn_track = gt_to_track[fn["gt_id"]]
            if fp_track is None or fn_track is None:
                continue
            time_overlap = max(
                0.0,
                min(fp_track["end_ms"], fn_track["end_ms"])
                - max(fp_track["start_ms"], fn_track["start_ms"]),
            ) / 1000.0
            pairing.append(
                {
                    "fp_gt_id": fp["gt_id"],
                    "fn_gt_id": fn["gt_id"],
                    "track_iou": round(
                        iou(fp_track["median_bbox_xyxy"], fn_track["median_bbox_xyxy"]), 4
                    ),
                    "time_overlap_s": round(time_overlap, 4),
                }
            )

    result = {
        "frozen_gt_sha256": json.loads(
            (OUT / "gt_mask_geometry.json").read_text(encoding="utf-8")
        )["frozen_gt_sha256"],
        "threshold": 0.45,
        "geometry": "parse_mask_coverage.parse_ass_dialogues, mask pad removed by 8px",
        "false_negatives": false_negatives,
        "false_positives": false_positives,
        "confirmed_tracks_without_frozen_gt_match": unmatched_confirmed,
        "fp_fn_pairing": sorted(pairing, key=lambda item: (item["track_iou"], item["time_overlap_s"]), reverse=True),
    }
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
