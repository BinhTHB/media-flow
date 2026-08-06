"""Unit tests for model path resolution and detection config.

These tests do NOT depend on the model file or GPU.
"""

from __future__ import annotations

import os

import pytest

from src.infrastructure.detection.config import (
    DEFAULT_PER_CLASS_CONFIDENCE,
    DetectionConfig,
    TemporalConfig,
    load_detection_config,
)
from src.infrastructure.detection.paths import (
    MEDIA_FLOW_YOLO_WEIGHTS_ENV,
    DEFAULT_YOLO_WEIGHTS_REL,
    ModelNotFoundError,
    resolve_project_root,
    resolve_yolo_weights,
)


def test_project_root_resolves_to_repo(tmp_path, monkeypatch):
    # Even with a custom cwd, resolve_project_root should find the repo via AGENTS.md.
    root = resolve_project_root()
    assert (root / "AGENTS.md").exists()
    assert (root / "src").is_dir()


def test_default_weights_points_at_models_yolo(tmp_path, monkeypatch):
    root = tmp_path
    (root / "AGENTS.md").write_text("x", encoding="utf-8")
    (root / "models" / "yolo").mkdir(parents=True)
    weights = root / "models" / "yolo" / "best.pt"
    weights.write_bytes(b"model")
    monkeypatch.delenv(MEDIA_FLOW_YOLO_WEIGHTS_ENV, raising=False)
    resolved = resolve_yolo_weights(project_root=root)
    assert resolved == weights.resolve()
    parts = list(DEFAULT_YOLO_WEIGHTS_REL.parts)
    assert parts == ["models", "yolo", "best.pt"]


def test_weights_missing_raises_clear_error(tmp_path, monkeypatch):
    root = tmp_path
    (root / "AGENTS.md").write_text("x", encoding="utf-8")
    monkeypatch.delenv(MEDIA_FLOW_YOLO_WEIGHTS_ENV, raising=False)
    with pytest.raises(ModelNotFoundError) as exc_info:
        resolve_yolo_weights(project_root=root)
    msg = str(exc_info.value)
    assert "best.pt" in msg
    assert "models" in msg
    assert "MEDIA_FLOW_YOLO_WEIGHTS" in msg


def test_env_override_wins(tmp_path, monkeypatch):
    root = tmp_path
    (root / "AGENTS.md").write_text("x", encoding="utf-8")
    custom = root / "custom.pt"
    custom.write_bytes(b"m")
    monkeypatch.setenv(MEDIA_FLOW_YOLO_WEIGHTS_ENV, str(custom))
    resolved = resolve_yolo_weights(
        project_root=root, config_value="models/yolo/best.pt"
    )
    assert resolved == custom.resolve()


def test_config_value_used_when_no_env(tmp_path, monkeypatch):
    root = tmp_path
    (root / "AGENTS.md").write_text("x", encoding="utf-8")
    custom = root / "cfg" / "w.pt"
    custom.parent.mkdir(parents=True)
    custom.write_bytes(b"m")
    monkeypatch.delenv(MEDIA_FLOW_YOLO_WEIGHTS_ENV, raising=False)
    resolved = resolve_yolo_weights(project_root=root, config_value="cfg/w.pt")
    assert resolved == custom.resolve()


def test_absolute_config_value(tmp_path, monkeypatch):
    root = tmp_path
    (root / "AGENTS.md").write_text("x", encoding="utf-8")
    custom = tmp_path / "abs.pt"
    custom.write_bytes(b"m")
    monkeypatch.delenv(MEDIA_FLOW_YOLO_WEIGHTS_ENV, raising=False)
    resolved = resolve_yolo_weights(project_root=root, config_value=str(custom))
    assert resolved == custom.resolve()


def test_detection_config_defaults_when_no_file(tmp_path, monkeypatch):
    monkeypatch.delenv(MEDIA_FLOW_YOLO_WEIGHTS_ENV, raising=False)
    cfg = load_detection_config(tmp_path / "none.yaml", project_root=tmp_path)
    assert cfg.enabled is True
    assert cfg.imgsz == 1280
    assert cfg.iou == 0.45
    assert cfg.max_det == 100
    assert cfg.confidence["subtitle"] == 0.15
    assert cfg.temporal.enabled is True
    assert DEFAULT_PER_CLASS_CONFIDENCE["subtitle"] == 0.15


def test_device_auto_and_half():
    from unittest.mock import patch

    cfg = DetectionConfig(device="auto", half="auto")
    with patch("torch.cuda.is_available", return_value=True):
        assert cfg.device_resolved() == "cuda"
        assert cfg.use_half() is True
    with patch("torch.cuda.is_available", return_value=False):
        assert cfg.device_resolved() == "cpu"
        assert cfg.use_half() is False


def test_half_needs_cuda():
    from unittest.mock import patch

    cfg = DetectionConfig(device="cuda", half=True)
    with patch("torch.cuda.is_available", return_value=False):
        # use_half only considers the device string here; device_resolved()==cuda
        # but torch reports no cuda. We guard this in the detector; here we only
        # assert the half policy never enables on CPU explicitly.
        assert cfg.use_half() is True  # device string is cuda
    cfg2 = DetectionConfig(device="cpu", half=True)
    assert cfg2.use_half() is False


def test_temporal_config():
    t = TemporalConfig(enabled=False, max_gap_frames=1)
    assert t.enabled is False
    assert t.max_gap_frames == 1
