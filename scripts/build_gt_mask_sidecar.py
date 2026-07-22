"""Build round23 GT geometry/timestamp sidecars from the verified round19 ASS.

The frozen GT remains immutable. Geometry is parsed only through
parse_mask_coverage.parse_ass_dialogues, which is the invariant parser.
"""
from __future__ import annotations

import hashlib
import json
import re
import tempfile
from pathlib import Path

from parse_mask_coverage import parse_ass_dialogues

ASS_PATH = Path("output/round19-gpu/subtitles/subtitle.ass")
GT_PATH = Path("scripts/frozen_gt/sample5/frozen_gt.json")
OUT_DIR = Path("artifacts/round23")


def canonical_event(line: str) -> dict:
    """Parse one provenance-tagged ASS geometry event with the invariant parser."""
    with tempfile.NamedTemporaryFile(
        "w", suffix=".ass", encoding="utf-8", delete=False
    ) as temp:
        temp.write(line + "\n")
        temp_path = Path(temp.name)
    try:
        events = parse_ass_dialogues(temp_path)
    finally:
        temp_path.unlink(missing_ok=True)
    if len(events) != 1:
        raise ValueError("Invariant parser did not produce exactly one mask event.")
    return events[0]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frozen_sha = hashlib.sha256(GT_PATH.read_bytes()).hexdigest()
    if frozen_sha != "c60196edaf3d418f89073b05649f68cf37168f4f316d261d09cf6f5d1f27b103":
        raise RuntimeError(f"Frozen GT SHA mismatch: {frozen_sha}")

    gt_by_id = {item["id"]: item for item in json.loads(GT_PATH.read_text(encoding="utf-8"))}
    events_by_track: dict[str, list[dict]] = {}
    for line in ASS_PATH.read_text(encoding="utf-8", errors="replace").splitlines():
        match = re.search(r"provenance=classification:(T\d+)", line)
        if not match:
            continue
        track_id = match.group(1)
        events_by_track.setdefault(track_id, []).append(canonical_event(line))

    if set(events_by_track) != set(gt_by_id):
        missing = sorted(set(gt_by_id) - set(events_by_track))
        extra = sorted(set(events_by_track) - set(gt_by_id))
        raise RuntimeError(f"ASS/GT track mismatch. missing={missing}, extra={extra}")

    geometry = []
    timestamps = {}
    signature = []
    for track_id in sorted(events_by_track):
        events = events_by_track[track_id]
        x_values = [event["x"] for event in events]
        y_values = [event["y"] for event in events]
        widths = [event["width"] for event in events]
        heights = [event["height"] for event in events]
        starts = [event["start"] for event in events]
        ends = [event["end"] for event in events]
        representative = events[len(events) // 2]
        bbox = [
            representative["x"],
            representative["y"],
            representative["x"] + representative["width"],
            representative["y"] + representative["height"],
        ]
        item = {
            "track_id": track_id,
            "event_count": len(events),
            "bbox_xyxy": bbox,
            "avg_top_y_ratio": sum(y_values) / len(y_values) / 1280,
            "avg_width_px": sum(widths) / len(widths),
            "avg_height_px": sum(heights) / len(heights),
            "start_time_s": min(starts),
            "end_time_s": max(ends),
            "duration_s": max(ends) - min(starts),
        }
        geometry.append(item)
        timestamps[track_id] = {
            "start_time_s": item["start_time_s"],
            "end_time_s": item["end_time_s"],
            "duration_s": item["duration_s"],
        }
        frozen = gt_by_id[track_id]
        signature.append({
            "track_id": track_id,
            "canonical_y": representative["y"],
            "legacy_parsed_y": representative["y"] - representative["height"],
            "height": representative["height"],
            "delta_y": representative["height"],
            "frozen_avg_bottom_y_ratio": frozen["avg_bottom_y_ratio"],
            "canonical_avg_top_y_ratio": item["avg_top_y_ratio"],
        })

    (OUT_DIR / "gt_mask_geometry.json").write_text(
        json.dumps({
            "source_ass": str(ASS_PATH),
            "frozen_gt_sha256": frozen_sha,
            "parser": "scripts.parse_mask_coverage.parse_ass_dialogues",
            "anchor": "ASS \\an7 + \\pos is top-left for p1 drawing",
            "tracks": geometry,
        }, indent=2),
        encoding="utf-8",
    )
    (OUT_DIR / "gt_track_timestamps.json").write_text(
        json.dumps(timestamps, indent=2), encoding="utf-8"
    )
    (OUT_DIR / "gt_parser_signature.json").write_text(
        json.dumps(signature, indent=2), encoding="utf-8"
    )
    exact = sum(1 for item in signature if item["delta_y"] == item["height"])
    print(f"frozen_sha={frozen_sha}")
    print(f"tracks={len(geometry)} signature_delta_equals_height={exact}/{len(signature)}")
    print("T010", next(item for item in geometry if item["track_id"] == "T010"))


if __name__ == "__main__":
    main()
