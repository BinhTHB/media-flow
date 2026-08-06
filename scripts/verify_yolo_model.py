"""Verify the YOLO burn-in model.

Resolves the model path, loads it, prints model metadata, verifies the exact
4-class mapping, runs a dry-run on a synthetic blank frame, and returns a
non-zero exit code if the model or config is invalid.

Usage:
    python scripts/verify_yolo_model.py [--weights PATH] [--device cpu|cuda|auto]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Allow running from a fresh interpreter without install.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.infrastructure.detection import (
    MEDIA_FLOW_YOLO_WEIGHTS_ENV,
    DetectionConfig,
    EXPECTED_CLASS_NAMES,
    YoloBurnInDetector,
    load_detection_config,
    resolve_yolo_weights,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify YOLO burn-in model")
    parser.add_argument("--weights", type=str, default=None, help="Path to model weights")
    parser.add_argument(
        "--device", type=str, default=None, choices=["cpu", "cuda", "auto"],
        help="Device override",
    )
    args = parser.parse_args()

    cfg = load_detection_config()
    if args.weights is not None:
        cfg.weights = args.weights
    if args.device is not None:
        cfg.device = args.device

    try:
        weights_path = resolve_yolo_weights(config_value=cfg.weights)
    except Exception as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1

    print(f"Model path        : {weights_path}")
    print(f"File size         : {weights_path.stat().st_size} bytes")

    import ultralytics
    import torch

    print(f"Ultralytics version: {ultralytics.__version__}")
    print(f"PyTorch version    : {torch.__version__}")
    print(f"CUDA available     : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device        : {torch.cuda.get_device_name(0)}")

    try:
        detector = YoloBurnInDetector(config=cfg)
    except Exception as exc:
        print(f"FAIL: model load/config error: {exc}", file=sys.stderr)
        return 1

    print(f"Task               : {detector._model.task}")
    print(f"Device (resolved)  : {detector.device}")
    names = detector.get_class_names()
    print("Class mapping      :")
    for cid in sorted(names):
        print(f"    {cid}: {names[cid]}")

    expected = list(EXPECTED_CLASS_NAMES.values())
    actual = [names[i] for i in sorted(names)]
    if actual != expected or len(actual) != 4:
        print(
            f"FAIL: class mapping mismatch. Expected {expected}, got {actual}",
            file=sys.stderr,
        )
        detector.release()
        return 2

    print(f"Class count        : OK ({len(actual)} classes)")
    print(f"{MEDIA_FLOW_YOLO_WEIGHTS_ENV} : {os.environ.get(MEDIA_FLOW_YOLO_WEIGHTS_ENV, '(not set)')}")

    # Dry-run on a synthetic blank frame.
    import numpy as np

    img = np.full((480, 640, 3), 128, dtype=np.uint8)
    try:
        result = detector.detect_frame(img, frame_index=0, timestamp_seconds=0.0)
        print(
            f"Dry-run            : OK (blank frame, {len(result.detections)} detection(s))"
        )
    except Exception as exc:
        print(f"FAIL: dry-run error: {exc}", file=sys.stderr)
        detector.release()
        return 3
    finally:
        detector.release()

    print("VERIFY: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
