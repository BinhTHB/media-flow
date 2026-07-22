"""Create a zero-request Round 24 evidence pack from the frozen Round 23 E2E."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import cv2

from evaluate_round23_regions import _frozen_entries, _iou, _region_box

ROOT = Path(__file__).resolve().parents[1]
REGIONS = (
    ROOT / "output" / "round23-e2e" / "round23_sample5_fix3" / "debug" / "subtitle_regions.json"
)
FRAMES = ROOT / "output" / "round23-e2e" / "round23_sample5_fix3" / "frames"
OUT = ROOT / "artifacts" / "round24" / "fp_taxonomy"


def _taxonomy(region: dict) -> str:
    duration = region["end_time"] - region["start_time"]
    if duration >= 3.0 and region["height"] <= 0.12:
        return "watermark/logo"
    if region["y"] < 0.55:
        return "scene-text"
    if region["y"] < 0.72:
        return "ui/sticker"
    return "other"


def _frame_path(timestamp: float) -> Path:
    return FRAMES / f"frame_{round(timestamp * 30):06d}.jpg"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    regions = json.loads(REGIONS.read_text(encoding="utf-8"))
    entries, frozen_sha = _frozen_entries()
    by_track: dict[str, list[dict]] = defaultdict(list)
    for region in regions:
        by_track[region["track_id"]].append(region)

    edges: list[tuple[float, str, str]] = []
    for track_id, track_regions in by_track.items():
        for entry in entries:
            for region in track_regions:
                if region["end_time"] < entry["start_time"] or region["start_time"] > entry["end_time"]:
                    continue
                score = _iou(_region_box(region, 720, 1280), entry["bbox_xyxy"])
                if score >= 0.5:
                    edges.append((score, entry["id"], track_id))
    matched_gt: dict[str, str] = {}
    matched_predictions: set[str] = set()
    for _, entry_id, track_id in sorted(edges, reverse=True):
        if entry_id not in matched_gt and track_id not in matched_predictions:
            matched_gt[entry_id] = track_id
            matched_predictions.add(track_id)

    entry_by_id = {entry["id"]: entry for entry in entries}
    label_by_id = {
        item["id"]: item
        for item in json.loads(
            (ROOT / "scripts" / "frozen_gt" / "sample5" / "frozen_gt.json").read_text(
                encoding="utf-8"
            )
        )
    }
    false_positive_tracks = {
        track_id
        for entry_id, track_id in matched_gt.items()
        if not entry_by_id[entry_id]["is_subtitle"]
    } | (set(by_track) - matched_predictions)

    gt_by_track = {track_id: entry_id for entry_id, track_id in matched_gt.items()}
    rows = []
    for index, track_id in enumerate(sorted(false_positive_tracks), start=1):
        track_regions = by_track[track_id]
        region = track_regions[0]
        gt_id = gt_by_track.get(track_id)
        frame_path = _frame_path((region["start_time"] + region["end_time"]) / 2)
        crop_name = None
        if index <= 25 and frame_path.exists():
            image = cv2.imread(str(frame_path))
            x1 = max(0, round(region["x"] * image.shape[1] - 8))
            y1 = max(0, round(region["y"] * image.shape[0] - 8))
            x2 = min(image.shape[1], round((region["x"] + region["width"]) * image.shape[1] + 8))
            y2 = min(image.shape[0], round((region["y"] + region["height"]) * image.shape[0] + 8))
            crop_name = f"{index:02d}_{track_id}.png"
            cv2.imwrite(str(OUT / crop_name), image[y1:y2, x1:x2])
        rows.append(
            {
                "track_id": track_id,
                "gt_id": gt_id,
                "gt_label_rationale": label_by_id.get(gt_id, {}).get("label_rationale", ""),
                "bbox_xyxy": _region_box(region, 720, 1280),
                "time_range_s": [region["start_time"], region["end_time"]],
                "duration_s": round(region["end_time"] - region["start_time"], 4),
                "route": (
                    "crop_batch"
                    if region["classifier_model"] == "crop_classifier"
                    else "full_frame_verifier"
                ),
                "gemini_model": region["classifier_model"],
                "gemini_category": region["category"],
                "gemini_confidence": region["classifier_confidence"],
                "taxonomy": _taxonomy(region),
                "crop_file": crop_name,
            }
        )
    payload = {
        "source_e2e": str(REGIONS.relative_to(ROOT)),
        "frozen_gt_sha256": frozen_sha,
        "zero_live_requests": True,
        "false_positive_count": len(rows),
        "rows": rows,
    }
    (OUT / "fp_taxonomy.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    route_counts: dict[str, int] = defaultdict(int)
    category_counts: dict[str, int] = defaultdict(int)
    for row in rows:
        route_counts[row["route"]] += 1
        category_counts[row["taxonomy"]] += 1
    report = [
        "# Round 24 FP taxonomy",
        "",
        f"Frozen GT SHA-256: `{frozen_sha}`",
        f"FP tracks: `{len(rows)}`",
        "",
        "## Route distribution",
        *[f"- {key}: {value}" for key, value in sorted(route_counts.items())],
        "",
        "## Taxonomy distribution",
        *[f"- {key}: {value}" for key, value in sorted(category_counts.items())],
        "",
        "| Track | Route | Taxonomy | GT | Crop |",
        "|---|---|---|---|---|",
        *[
            f"| {row['track_id']} | {row['route']} | {row['taxonomy']} | "
            f"{row['gt_id'] or 'unmatched'} | {row['crop_file'] or 'not-exported'} |"
            for row in rows
        ],
    ]
    (OUT / "fp_taxonomy.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"fp_count": len(rows), "routes": route_counts, "taxonomy": category_counts}))


if __name__ == "__main__":
    main()
