"""Detection configuration (loaded from ``config/detection.yaml``)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.infrastructure.detection.paths import (
    resolve_project_root,
    resolve_yolo_weights,
)

#: Default standalone config file location (relative to project root).
DEFAULT_DETECTION_CONFIG_PATH = Path("config") / "detection.yaml"

# Default per-class confidence thresholds. Low subtitle threshold is intentional:
# subtitles often appear character-by-character or scroll quickly.
DEFAULT_PER_CLASS_CONFIDENCE: Dict[str, float] = {
    "subtitle": 0.15,
    "watermark": 0.20,
    "logo": 0.20,
    "other_burn_in_text": 0.20,
}

# Per-class action policy. Distinguishes *detecting* a class from *acting* on it
# (safe_cover = whether the class may be covered/removed downstream). The default
# keeps every class detected and coverable, which matches the legacy pipeline
# that treated all burn-in regions as coverable text. Providers/rendering may
# tighten `safe_cover` later (e.g. watermark/logo usually should not be covered).
DEFAULT_CLASS_POLICY: Dict[str, Dict[str, bool]] = {
    "subtitle": {"detect": True, "safe_cover": True},
    "watermark": {"detect": True, "safe_cover": True},
    "logo": {"detect": True, "safe_cover": True},
    "other_burn_in_text": {"detect": True, "safe_cover": True},
}


@dataclass
class TemporalConfig:
    enabled: bool = True
    max_gap_frames: int = 2
    min_track_frames: int = 2
    iou_match_threshold: float = 0.40
    box_smoothing_alpha: float = 0.65


@dataclass
class DetectionConfig:
    enabled: bool = True
    weights: Optional[str] = None  # path or relative path to the model file
    device: str = "auto"
    imgsz: int = 1280
    iou: float = 0.45
    max_det: int = 100
    half: str = "auto"  # "auto" | True | False
    agnostic_nms: bool = False
    verbose: bool = False
    confidence: Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_PER_CLASS_CONFIDENCE))
    class_policy: Dict[str, Dict[str, bool]] = field(default_factory=lambda: {
        k: dict(v) for k, v in DEFAULT_CLASS_POLICY.items()
    })
    temporal: TemporalConfig = field(default_factory=TemporalConfig)
    config_path: Optional[str] = None

    def resolved_weights(self, *, project_root: Optional[Path] = None) -> Path:
        env = os.environ.get("MEDIA_FLOW_YOLO_WEIGHTS")
        return resolve_yolo_weights(
            config_value=self.weights,
            env_value=env,
            project_root=project_root,
        )

    def device_resolved(self) -> str:
        """Resolve the 'auto' device string to 'cuda' or 'cpu'."""
        if self.device != "auto":
            return self.device
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    def use_half(self) -> bool:
        """Decide FP16 usage (only on CUDA)."""
        resolved = str(self.half).lower()
        if resolved == "auto":
            return self.device_resolved() == "cuda"
        return self.device_resolved() == "cuda" and bool(self.half)


def _merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def load_detection_config(
    path: Optional[Path | str] = None,
    *,
    project_root: Optional[Path] = None,
    env_prefix: str = "MEDIA_FLOW_DETECTION",
) -> DetectionConfig:
    """Load detection config from YAML (if present), else return defaults.

    The config file is optional: at minimum the settings dataclass defaults are
    used. Values can also be overridden via ``{env_prefix}__{KEY}`` environment
    variables (e.g. ``MEDIA_FLOW_DETECTION__IMGSZ=640``).
    """
    root = project_root or resolve_project_root()
    cfg_path = path or root / DEFAULT_DETECTION_CONFIG_PATH
    raw: Dict[str, Any] = {}
    resolved_p: Path = Path(cfg_path) if not isinstance(cfg_path, Path) else cfg_path
    if not resolved_p.is_absolute():
        resolved_p = root / resolved_p
    if resolved_p.is_file():
        try:
            import yaml

            with open(resolved_p, "r", encoding="utf-8") as fh:
                loaded = yaml.safe_load(fh) or {}
            if isinstance(loaded, dict):
                raw = loaded
        except Exception:
            raw = {}

    yolo_raw = raw.get("yolo", {}) if isinstance(raw.get("yolo"), dict) else {}
    conf_raw = yolo_raw.get("confidence", {})
    per_class = dict(DEFAULT_PER_CLASS_CONFIDENCE)
    if isinstance(conf_raw, dict):
        for k, v in conf_raw.items():
            if isinstance(v, (int, float)):
                per_class[str(k)] = float(v)

    policy_raw = yolo_raw.get("class_policy", {})
    class_policy = {k: dict(v) for k, v in DEFAULT_CLASS_POLICY.items()}
    if isinstance(policy_raw, dict):
        for k, v in policy_raw.items():
            if not isinstance(v, dict):
                continue
            entry = class_policy.get(str(k), {"detect": True, "safe_cover": True})
            if "detect" in v:
                entry["detect"] = _b(v["detect"])
            if "safe_cover" in v:
                entry["safe_cover"] = _b(v["safe_cover"])
            class_policy[str(k)] = entry

    temporal_raw = yolo_raw.get("temporal", {}) if isinstance(yolo_raw.get("temporal"), dict) else {}
    temporal = TemporalConfig(
        enabled=_b(temporal_raw.get("enabled", TemporalConfig().enabled)),
        max_gap_frames=int(temporal_raw.get("max_gap_frames", TemporalConfig().max_gap_frames)),
        min_track_frames=int(temporal_raw.get("min_track_frames", TemporalConfig().min_track_frames)),
        iou_match_threshold=float(temporal_raw.get("iou_match_threshold", TemporalConfig().iou_match_threshold)),
        box_smoothing_alpha=float(temporal_raw.get("box_smoothing_alpha", TemporalConfig().box_smoothing_alpha)),
    )

    cfg = DetectionConfig(
        enabled=_b(yolo_raw.get("enabled", True)),
        weights=yolo_raw.get("weights"),
        device=str(yolo_raw.get("device", "auto")),
        imgsz=int(yolo_raw.get("imgsz", 1280)),
        iou=float(yolo_raw.get("iou", 0.45)),
        max_det=int(yolo_raw.get("max_det", 100)),
        half=str(yolo_raw.get("half", "auto")),
        agnostic_nms=_b(yolo_raw.get("agnostic_nms", False)),
        verbose=_b(yolo_raw.get("verbose", False)),
        confidence=per_class,
        class_policy=class_policy,
        temporal=temporal,
        config_path=str(resolved_p),
    )

    # Environment overrides (shallow for scalars).
    for key in (
        "enabled", "device", "imgsz", "iou", "max_det", "half",
        "agnostic_nms", "verbose", "weights",
    ):
        env_key = f"{env_prefix}__{key.upper()}"
        if env_key in os.environ:
            value: Any = os.environ[env_key]
            if key in ("imgsz", "max_det"):
                value = int(value)
            elif key in ("iou",):
                value = float(value)
            elif key in ("enabled", "agnostic_nms", "verbose"):
                value = _b(value)
            setattr(cfg, key, value)
    return cfg


def _b(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)
