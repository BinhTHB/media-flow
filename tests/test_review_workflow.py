"""Regression tests for the human review workflow (review_store + gating)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.infrastructure.detection.evaluation.review_store import (
    build_summary,
    create_backup_once,
    get_row,
    load_label,
    load_manifest,
    migrate_manifest,
    save_label,
    update_manifest_row,
    validate_objects,
)
from scripts.evaluate_yolo_detection import select_eval_rows


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _make_set(tmp_path: Path) -> None:
    (tmp_path / "manifests").mkdir(parents=True)
    (tmp_path / "labels").mkdir()
    (tmp_path / "frames").mkdir()
    manifest = tmp_path / "manifests" / "manifest.csv"
    manifest.write_text(
        "frame_id,video_id,frame_index,timestamp_seconds,width,height,verified,"
        "label_source\n"
        "f01,v01,0,0.0,720,1280,false,prelabel_unverified\n"
        "f02,v01,30,1.0,720,1280,false,prelabel_unverified\n",
        encoding="utf-8",
    )


def _box(cid: int = 0, name: str = "subtitle", bb=(10.0, 10.0, 100.0, 60.0)) -> dict:
    return {"class_id": cid, "class_name": name, "bbox_xyxy": list(bb)}


# ---------------------------------------------------------------------------
# Manifest migration
# ---------------------------------------------------------------------------
def test_migrate_manifest_adds_review_columns(tmp_path: Path):
    _make_set(tmp_path)
    manifest = tmp_path / "manifests" / "manifest.csv"
    rows = migrate_manifest(manifest)
    assert "needs_review" in rows[0]
    assert "evaluation_role" in rows[0]
    assert "leakage_risk" in rows[0]
    assert rows[0]["verified"] == "false"
    assert rows[0]["evaluation_role"] == "unknown"
    # file rewritten
    assert "evaluation_role" in manifest.read_text(encoding="utf-8")


# 2. Save + load label atomic
def test_save_and_load_label(tmp_path: Path):
    labels = tmp_path / "labels"
    labels.mkdir()
    data = {"frame_id": "f01", "verified": True, "objects": [_box()]}
    save_label(labels, "f01", data)
    loaded = load_label(labels, "f01")
    assert loaded["verified"] is True
    assert loaded["evaluation_role"] == "unknown"  # defaulted
    assert len(loaded["objects"]) == 1


# 3. Empty frame GT valid
def test_empty_frame_is_valid(tmp_path: Path):
    labels = tmp_path / "labels"
    labels.mkdir()
    data = {"frame_id": "f02", "verified": True, "objects": []}
    save_label(labels, "f02", data)
    loaded = load_label(labels, "f02")
    assert loaded["objects"] == []
    # validation of an empty object list produces no errors
    assert validate_objects([], 720, 1280, "f02") == []


# 4. Invalid box rejection
def test_validate_objects_rejects_bad_boxes():
    dup1 = _box(bb=(0, 0, 10, 10))
    dup2 = _box(bb=(0, 0, 10, 10))
    frames = [
        _box(bb=(0, 0, 5000, 5000)),  # out of frame
        {"class_id": 0, "class_name": "wrong", "bbox_xyxy": [0, 0, 10, 10]},  # class mismatch
        {"class_id": 5, "class_name": "subtitle", "bbox_xyxy": [0, 0, 10, 10]},  # bad id
        {"class_id": 0, "class_name": "subtitle", "bbox_xyxy": [10, 10, 5, 50]},  # <=0 width
        dup1,
        dup2,
    ]
    errors = validate_objects(frames, 720, 1280, "f")
    assert any("outside frame" in e for e in errors)
    assert any("class_name" in e and "wrong" in e for e in errors)
    assert any("class_id must be an int" in e for e in errors)
    assert any("non-positive width/height" in e for e in errors)
    assert any("duplicate box" in e for e in errors)


# 5. Manifest row update + reading back
def test_update_manifest_row(tmp_path: Path):
    _make_set(tmp_path)
    manifest = tmp_path / "manifests" / "manifest.csv"
    migrate_manifest(manifest)
    update_manifest_row(manifest, "f01", {"verified": "true", "evaluation_role": "held_out"})
    rows = load_manifest(manifest)
    f01 = next(r for r in rows if r["frame_id"] == "f01")
    assert f01["verified"] == "true"
    assert f01["evaluation_role"] == "held_out"
    # other rows untouched
    f02 = next(r for r in rows if r["frame_id"] == "f02")
    assert f02["verified"] == "false"


# 6. Summary counts
def test_build_summary_counts(tmp_path: Path):
    _make_set(tmp_path)
    manifest = tmp_path / "manifests" / "manifest.csv"
    migrate_manifest(manifest)
    update_manifest_row(manifest, "f01", {"verified": "true"})
    s = build_summary(load_manifest(manifest), tmp_path / "labels", tmp_path / "frames")
    assert s["total"] == 2
    assert s["verified"] == 1
    assert s["remaining"] == 1


# 7. Backup once per day
def test_backup_once_per_day(tmp_path: Path):
    b1 = create_backup_once(tmp_path)
    assert b1 is not None
    b2 = create_backup_once(tmp_path)
    assert b2 is None  # already backed-up today


# ---------------------------------------------------------------------------
# Gating (evaluator)
# ---------------------------------------------------------------------------
def _row(fid: str, verified=True, role="unknown", needs_review=False, leak=False) -> dict:
    return {
        "frame_id": fid,
        "verified": "true" if verified else "false",
        "evaluation_role": role,
        "needs_review": "true" if needs_review else "false",
        "leakage_risk": "true" if leak else "false",
    }


def test_gating_verified_only():
    rows = [_row("a", True), _row("b", False)]
    picked, g = select_eval_rows(rows, total_manifest=2)
    assert [r["frame_id"] for r in picked] == ["a"]
    assert g["verified_used"] == 1
    assert g["unverified_excluded"] == 1


def test_gating_needs_review_excluded():
    rows = [_row("a", True, needs_review=True), _row("b", True)]
    picked, g = select_eval_rows(rows, total_manifest=2)
    assert [r["frame_id"] for r in picked] == ["b"]
    assert g["needs_review_excluded"] == 1


def test_gating_include_needs_review():
    rows = [_row("a", True, needs_review=True), _row("b", True)]
    picked, _ = select_eval_rows(rows, include_needs_review=True, total_manifest=2)
    assert len(picked) == 2


def test_gating_held_out_filter():
    rows = [_row("a", True, role="dev"), _row("b", True, role="held_out"),
            _row("c", True, role="excluded")]
    picked, g = select_eval_rows(rows, roles="held_out", total_manifest=3)
    assert [r["frame_id"] for r in picked] == ["b"]
    assert g["role_excluded"] >= 1


def test_gating_leakage_excluded_default():
    rows = [_row("a", True, leak=True), _row("b", True)]
    picked, g = select_eval_rows(rows, total_manifest=2)
    assert [r["frame_id"] for r in picked] == ["b"]
    assert g["leakage_excluded"] == 1


def test_gating_partial_flag():
    rows = [_row("a", True), _row("b", True)]
    picked, g = select_eval_rows(rows, roles="nothing_matches", total_manifest=2)
    assert picked == []
    assert g["partial"] is True


def test_gating_no_verified_yields_partial():
    # select_eval_rows receives the already-verified subset; none verified -> empty.
    picked, g = select_eval_rows([], total_manifest=1)
    assert picked == []
    assert g["partial"] is True
    assert g["verified_total"] == 0
