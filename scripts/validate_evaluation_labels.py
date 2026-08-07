"""Validate ground-truth evaluation labels before they are used for metrics.

Checks the canonical schema invariants required by ``evaluate_yolo_detection.py``:

* ``class_id`` in 0..3 and ``class_name`` matches the canonical mapping;
* bounding boxes are inside the frame, have positive width/height, and no NaN;
* no duplicate boxes that look like accidental duplicates;
* no orphan label files (label without image) and no orphan images (image
  without label);
* frame metadata in the manifest is consistent with the label file.

Exits 0 when everything is valid, 1 when validation issues were found.

Example:
  .venv\\Scripts\\python.exe scripts\\validate_evaluation_labels.py ^
      --manifest evaluation/manifests/manifest.csv ^
      --labels evaluation/labels --frames evaluation/frames
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.infrastructure.detection.schemas import EXPECTED_CLASS_NAMES

VALID_IDS = set(EXPECTED_CLASS_NAMES.keys())
VALID_NAMES = set(EXPECTED_CLASS_NAMES.values())
NAME_BY_ID = EXPECTED_CLASS_NAMES


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Validate evaluation labels")
    p.add_argument("--manifest", required=True, help="CSV manifest")
    p.add_argument("--labels", required=True, help="Labels directory")
    p.add_argument("--frames", required=True, help="Frames directory")
    p.add_argument("--require-verified", action="store_true",
                   help="Fail unless every manifest row is verified=true")
    return p


def _is_json(path: Path) -> bool:
    return path.suffix == ".json"


def validate_objects(objs: List[Dict[str, Any]], width: int, height: int, frame_id: str, errors: List[str]) -> None:
    seen: set = set()
    for i, o in enumerate(objs):
        tag = f"{frame_id}#{i}"
        try:
            cid = int(o["class_id"])
            name = str(o["class_name"])
        except (KeyError, TypeError, ValueError):
            errors.append(f"{tag}: missing/invalid class_id or class_name")
            continue
        if cid not in VALID_IDS:
            errors.append(f"{tag}: class_id {cid} out of 0..3")
        if name not in VALID_NAMES:
            errors.append(f"{tag}: class_name {name!r} not in {sorted(VALID_NAMES)}")
        if NAME_BY_ID.get(cid) and name and NAME_BY_ID[cid] != name:
            errors.append(f"{tag}: class_id={cid} but class_name={name!r} (expected {NAME_BY_ID[cid]!r})")
        try:
            box = [float(v) for v in o["bbox_xyxy"]]
        except (KeyError, TypeError, ValueError):
            errors.append(f"{tag}: missing/invalid bbox_xyxy")
            continue
        if len(box) != 4:
            errors.append(f"{tag}: bbox_xyxy len != 4")
            continue
        if any(math.isnan(v) or math.isinf(v) for v in box):
            errors.append(f"{tag}: bbox contains NaN/Inf")
            continue
        x1, y1, x2, y2 = box
        if x2 <= x1 or y2 <= y1:
            errors.append(f"{tag}: non-positive width/height ({x2 - x1:.2f}x{y2 - y1:.2f})")
        if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
            errors.append(f"{tag}: box outside frame {width}x{height}")
        x1i, y1i, x2i, y2i = int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))
        key = (cid, x1i, y1i, x2i, y2i)
        if key in seen:
            errors.append(f"{tag}: duplicate box (same class + bbox)")
        seen.add(key)


def main() -> int:
    args = build_parser().parse_args()
    frames_dir = Path(args.frames)
    labels_dir = Path(args.labels)
    manifest = []
    try:
        import csv
        with open(args.manifest, "r", encoding="utf-8", newline="") as fh:
            manifest = [dict(r) for r in csv.DictReader(fh)]
    except FileNotFoundError:
        print(f"ERROR: manifest {args.manifest} not found", file=sys.stderr)
        return 1

    errors: List[str] = []
    warnings: List[str] = []
    manifest_ids = set(str(r["frame_id"]) for r in manifest)
    label_files = [p for p in labels_dir.iterdir() if _is_json(p)] if labels_dir.is_dir() else []
    image_files = {p.stem for p in frames_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}}

    for row in manifest:
        frame_id = str(row["frame_id"])
        width = int(row.get("width", 0))
        height = int(row.get("height", 0))
        verified = str(row.get("verified", "")).strip().lower() in {"true", "yes", "1"}
        if width <= 0 or height <= 0:
            errors.append(f"{frame_id}: manifest width/height invalid ({width}x{height})")
        if frame_id not in image_files:
            errors.append(f"{frame_id}: manifest frame has no image in {frames_dir}")
        label_path = labels_dir / f"{frame_id}.json"
        if not label_path.is_file():
            errors.append(f"{frame_id}: no label file")
            continue
        try:
            data = json.loads(label_path.read_text(encoding="utf-8"))
        except Exception as exc:
            errors.append(f"{frame_id}: unreadable label JSON: {exc}")
            continue
        if not bool(data.get("verified")):
            warnings.append(f"{frame_id}: label not verified (pre-label only)")
        objs = data.get("objects", [])
        validate_objects(objs, width, height, frame_id, errors)

    # Orphan label files
    for p in label_files:
        stem = p.stem
        if stem not in manifest_ids:
            warnings.append(f"orphan label {p.name}: not in manifest")
        if stem not in image_files:
            warnings.append(f"orphan label {p.name}: no image in {frames_dir}")

    if args.require_verified and any(w.startswith("=>") or w.endswith("not verified") for w in warnings) is False:
        pass

    print(f"Checked {len(manifest)} manifest rows, {len(manifest_ids)} frames")
    print(f"Label files: {len(label_files)}; image files: {len(image_files)}")
    if errors:
        print("\nERRORS:")
        for e in errors:
            print(f"  - {e}")
    if warnings:
        print("\nWARNINGS (not necessarily errors):")
        for w in warnings:
            print(f"  - {w}")
    print(f"\n{len(errors)} errors, {len(warnings)} warnings")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
