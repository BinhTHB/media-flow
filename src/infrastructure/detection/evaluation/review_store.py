"""Data-layer helpers for the human evaluation-review workflow.

This module owns the reading / writing of the evaluation manifest and label
files that the local review server and the evaluator both use. It provides the
atomic save + validation guarantees required by the review workflow:

* labels are written atomically (temp file + ``os.replace``) so an interrupted
  write never leaves a corrupt label;
* manifest rows are updated atomically;
* a per-day backup of ``manifest.csv`` + ``labels/`` is created once before the
  first edit of a session;
* saves reject invalid boxes (bad class id/name, out-of-frame box, zero area,
  NaNs) with an HTTP-agnostic error list so the server returns 400.

The review state model supports at least three states: ``unverified``,
``verified`` and ``needs_review``, plus independent ``needs_second_review`` and
``evaluation_role`` / ``leakage_risk`` fields for hard-case double-checking and
held-out-bookkeeping.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.infrastructure.detection.schemas import EXPECTED_CLASS_NAMES, verify_class_mapping

VALID_IDS = set(EXPECTED_CLASS_NAMES.keys())
VALID_NAMES = set(EXPECTED_CLASS_NAMES.values())

#: Canonical manifest columns, including the review workflow fields.
MANIFEST_COLUMNS = [
    "frame_id",
    "video_id",
    "frame_index",
    "timestamp_seconds",
    "width",
    "height",
    "verified",
    "needs_review",
    "needs_second_review",
    "evaluation_role",
    "leakage_risk",
]


def default_row() -> Dict[str, Any]:
    return {
        "verified": "false",
        "needs_review": "false",
        "needs_second_review": "false",
        "evaluation_role": "unknown",
        "leakage_risk": "false",
        "label_source": "prelabel_unverified",
    }


FRAME_EXTENSIONS = (".jpg", ".jpeg", ".png")

#: Allowed extension -> MIME type for serving frame images.
FRAME_MIMES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
}


def resolve_frame_path(frames_dir: Path, frame_id: str) -> Optional[Path]:
    """Resolve the actual frame file for ``frame_id`` inside ``frames_dir``.

    Walks the allowed extensions rather than assuming ``.jpg``. Returns
    ``None`` when no matching file exists. The returned path is guaranteed to
    be within ``frames_dir`` (no traversal possible because ``frame_id`` is used
    verbatim as the stem with a controlled extension).
    """
    frames_dir = frames_dir.resolve()
    for ext in FRAME_EXTENSIONS:
        cand = frames_dir / f"{frame_id}{ext}"
        if cand.is_file():
            return cand
    return None


# ---------------------------------------------------------------------------
# Manifest IO
# ---------------------------------------------------------------------------
def load_manifest(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []
    with open(path, "r", encoding="utf-8") as fh:
        import csv

        return [dict(r) for r in csv.DictReader(fh)]


def migrate_manifest(path: Path, *, force_write: bool = False) -> List[Dict[str, Any]]:
    """Ensure every manifest row has all canonical columns.

    Adds missing review fields with neutral defaults (nothing is marked
    verified; roles default to ``unknown``). Rewrites the CSV only when the
    file is missing columns.
    """
    import csv

    existing = load_manifest(path)
    if not existing:
        return existing
    missing = [c for c in MANIFEST_COLUMNS if c not in existing[0]]
    if not missing and not force_write:
        return existing
    defaults = _row_defaults()
    rows = []
    for r in existing:
        merged = dict(r)
        for c in MANIFEST_COLUMNS:
            if c not in merged or merged[c] == "":
                merged[c] = defaults.get(c, "")
        rows.append(merged)
    _write_manifest(path, rows)
    return rows


def _row_defaults() -> Dict[str, str]:
    return {
        "verified": "false",
        "needs_review": "false",
        "needs_second_review": "false",
        "evaluation_role": "unknown",
        "leakage_risk": "false",
        "label_source": "prelabel_unverified",
    }


def _atomic_write_manifest(path: Path, rows: List[Dict[str, Any]]) -> None:
    import csv

    cols = MANIFEST_COLUMNS + [c for c in rows[0] if c not in MANIFEST_COLUMNS]
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".csv.tmp")
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


def _write_manifest(path: Path, rows: List[Dict[str, Any]]) -> None:
    _atomic_write_manifest(path, rows)


def update_manifest_row(
    manifest_path: Path, frame_id: str, updates: Dict[str, Any]
) -> Dict[str, Any]:
    """Overwrite one manifest row atomically.

    ``updates`` may be a full row or a partial dict that gets merged over the
    existing row. Returns the final row (or the merged row if the frame_id is
    not found, in which case the manifest is left untouched).
    """
    rows = load_manifest(manifest_path)
    for i, r in enumerate(rows):
        if str(r.get("frame_id")) == frame_id:
            merged = {**r, **updates}
            rows[i] = merged
            _atomic_write_manifest(manifest_path, rows)
            return merged
    return dict(updates)


def set_row_fields(
    manifest_path: Path, frame_id: str, fields: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    return update_manifest_row(manifest_path, frame_id, fields)


def get_row(manifest_path: Path, frame_id: str) -> Optional[Dict[str, Any]]:
    for r in load_manifest(manifest_path):
        if str(r.get("frame_id")) == frame_id:
            return r
    return None


# ---------------------------------------------------------------------------
# Label IO
# ---------------------------------------------------------------------------
def load_label(labels_dir: Path, frame_id: str) -> Dict[str, Any]:
    """Return the label dict for a frame, defaulting fields to neutral values."""
    path = labels_dir / f"{frame_id}.json"
    if path.is_file():
        data = json.loads(path.read_text(encoding="utf-8"))
    else:
        data = {}
    data.setdefault("frame_id", frame_id)
    data.setdefault("verified", False)
    data.setdefault("needs_review", False)
    data.setdefault("needs_second_review", False)
    data.setdefault("evaluation_role", "unknown")
    data.setdefault("leakage_risk", False)
    data.setdefault("label_source", "prelabel_unverified")
    data.setdefault("objects", [])
    return data


def _atomic_write_json(path: Path, data: Any) -> None:
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".json.tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


def save_label(labels_dir: Path, frame_id: str, data: Dict[str, Any]) -> Path:
    """Atomic write of a label file. Returns the written path."""
    path = labels_dir / f"{frame_id}.json"
    _atomic_write_json(path, data)
    return path


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def validate_objects(
    objects: List[Dict[str, Any]], width: float, height: float, frame_id: str
) -> List[str]:
    """Validate a label's object list; returns a list of human-readable errors."""
    import math

    errors: List[str] = []
    seen: set = set()
    for i, obj in enumerate(objects or []):
        tag = f"{frame_id}#{i}"
        cid_raw = obj.get("class_id")
        name = obj.get("class_name")
        if cid_raw is None or not isinstance(cid_raw, int) or cid_raw not in VALID_IDS:
            errors.append(f"{tag}: class_id must be an int in {{0,1,2,3}}")
            continue
        if name != EXPECTED_CLASS_NAMES.get(cid_raw):
            errors.append(f"{tag}: class_name {name!r} != expected {EXPECTED_CLASS_NAMES.get(cid_raw)!r}")
            continue
        bb = obj.get("bbox_xyxy")
        if not isinstance(bb, list) or len(bb) != 4:
            errors.append(f"{tag}: bbox_xyxy must be a list of 4 numbers")
            continue
        vals = []
        bad = False
        for v in bb:
            try:
                f = float(v)
            except (TypeError, ValueError):
                errors.append(f"{tag}: bbox value not numeric: {v!r}")
                bad = True
                continue
            if math.isnan(f) or math.isinf(f):
                errors.append(f"{tag}: bbox contains NaN/Inf")
                bad = True
            vals.append(f)
        if bad:
            continue
        x1, y1, x2, y2 = vals
        if x2 <= x1 or y2 <= y1:
            errors.append(f"{tag}: non-positive width/height ({x2 - x1:.2f}x{y2 - y1:.2f})")
        if x1 < -1e-6 or y1 < -1e-6 or x2 > width + 1e-6 or y2 > height + 1e-6:
            errors.append(f"{tag}: box outside frame {width:.0f}x{height:.0f}")
        key = (cid_raw, int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2)))
        if key in seen:
            errors.append(f"{tag}: duplicate box (same class + bbox)")
        seen.add(key)
    return errors


# ---------------------------------------------------------------------------
# Backup
# ---------------------------------------------------------------------------
def create_backup_once(eval_root: Path, *available: Path) -> Optional[Path]:
    """Create one dated backup of manifest + labels, once per UTC date.

    Returns the backup directory, or ``None`` if a backup for today already
    exists. Backups are never committed (see ``.gitignore``).
    """
    backups_dir = eval_root / "backups"
    day = datetime.now().strftime("%Y-%m-%d")
    today_dir = backups_dir / day
    if today_dir.exists():
        # A backup already exists for today -> don't multiply copies.
        return None
    today_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = eval_root / "manifests" / "manifest.csv"
    if manifest_path.is_file():
        shutil.copy2(manifest_path, today_dir / "manifest.csv")
    labels_dir = eval_root / "labels"
    if labels_dir.is_dir():
        dest = today_dir / "labels"
        shutil.copytree(labels_dir, dest, ignore=shutil.ignore_patterns("__pycache__"))
    (today_dir / "README.txt").write_text(
        "Auto-created by the evaluation review server before the first edit of this day.\n"
        "Not ground truth; a rollback snapshot.\n",
        encoding="utf-8",
    )
    return today_dir


# ---------------------------------------------------------------------------
# Summary (progress dashboard)
# ---------------------------------------------------------------------------
@dataclass
class FrameSummary:
    frame_id: str
    video_id: str
    frame_index: int
    timestamp_seconds: float
    width: int
    height: int
    verified: bool
    needs_review: bool
    needs_second_review: bool
    evaluation_role: str
    leakage_risk: bool
    label_source: str
    n_objects: int
    classes: List[str] = field(default_factory=list)
    min_confidence: Optional[float] = None
    has_image: bool = False


def build_summary(
    manifest: List[Dict[str, Any]], labels_dir: Path, frames_dir: Path
) -> Dict[str, Any]:
    verified = 0
    needs_review = 0
    second = 0
    class_dist: Dict[str, int] = {}
    roles: Dict[str, int] = {}
    total = len(manifest)
    for r in manifest:
        if str(r.get("verified", "")).lower() in {"true", "1", "yes"}:
            verified += 1
        if str(r.get("needs_review", "")).lower() in {"true", "1", "yes"}:
            needs_review += 1
        if str(r.get("needs_second_review", "")).lower() in {"true", "1", "yes"}:
            second += 1
        role = str(r.get("evaluation_role", "unknown"))
        roles[role] = roles.get(role, 0) + 1
        frame_id = str(r.get("frame_id"))
        label = load_label(labels_dir, frame_id)
        for obj in label.get("objects", []):
            name = str(obj.get("class_name", "unknown"))
            class_dist[name] = class_dist.get(name, 0) + 1
    return {
        "total": total,
        "verified": verified,
        "needs_review": needs_review,
        "needs_second_review": second,
        "remaining": max(0, total - verified),
        "progress_pct": round(100.0 * verified / total, 1) if total else 0.0,
        "class_distribution": class_dist,
        "roles": roles,
    }
