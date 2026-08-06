"""Unit tests for YOLO detector behavior with a mocked Ultralytics model.

No real YOLO model, GPU, or video required here.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from src.infrastructure.detection.config import DetectionConfig
from src.infrastructure.detection.schemas import ClassMappingError
from src.infrastructure.detection.yolo_detector import YoloBurnInDetector


class _CpuTensor:
    """Mimics a torch tensor with .cpu() returning something with .numpy()."""

    def __init__(self, arr):
        self._arr = np.asarray(arr, dtype=np.float32)

    def cpu(self):
        return self

    def numpy(self):
        return self._arr

    def __float__(self):
        if self._arr.size == 1:
            return float(self._arr.reshape(-1)[0])
        raise TypeError("cannot convert non-scalar to float")

    def __iter__(self):
        return iter(self._arr)

    def __len__(self):
        return len(self._arr)

    def __getitem__(self, i):
        return float(self._arr[i])


class _Box:
    def __init__(self, xyxy, confidence, class_id):
        self.xyxy = [_CpuTensor(np.array([xyxy], dtype=np.float32))]
        self.conf = [_CpuTensor(np.array([confidence], dtype=np.float32))]
        self.cls = _CpuTensor(np.array([class_id], dtype=np.float32))


class _Boxes:
    """Mimics ultralytics Boxes: has .xyxy/.conf/.cls and supports len()."""

    def __init__(self, xyxy, conf, cls):
        self.xyxy = _CpuTensor(xyxy)
        self.conf = _CpuTensor(conf)
        self.cls = _CpuTensor(cls)
        self._n = len(np.asarray(xyxy))

    def __len__(self):
        return self._n


def make_result(image, boxes):
    if boxes:
        xyxy = np.array([b.xyxy[0].numpy()[0] for b in boxes], dtype=np.float32).reshape(-1, 4)
        conf = np.array([float(b.conf[0]) for b in boxes], dtype=np.float32)
        cls = np.array([float(b.cls[0]) for b in boxes], dtype=np.float32)
    else:
        xyxy = np.zeros((0, 4), dtype=np.float32)
        conf = np.zeros((0,), dtype=np.float32)
        cls = np.zeros((0,), dtype=np.float32)
    return SimpleNamespace(
        orig_img=image,
        orig_shape=image.shape[:2],
        boxes=_Boxes(xyxy, conf, cls),
    )


FAKE_NAMES = {0: "subtitle", 1: "watermark", 2: "logo", 3: "other_burn_in_text"}


class _FakeModel:
    task = "detect"
    names = FAKE_NAMES

    def __init__(self, results=None):
        self.predict_results = results or []
        self.predict_calls = 0
        self.load_count = 0

    def to(self, *_a, **_k):
        return self

    def predict(self, *_a, **_k):
        self.predict_calls += 1
        return list(self.predict_results)


class _InferenceMode:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _patch_env(monkeypatch, model=None):
    import torch

    model = model if model is not None else _FakeModel()
    monkeypatch.setattr(
        "src.infrastructure.detection.yolo_detector.YOLO",
        lambda *a, **k: model,
    )
    monkeypatch.setattr(torch, "inference_mode", lambda *a, **k: _InferenceMode())
    monkeypatch.setattr(
        "src.infrastructure.detection.yolo_detector.resolve_yolo_weights",
        lambda *a, **k: "fake.pt",
    )
    return model


def _base_config(**kw):
    cfg = DetectionConfig(device="cpu", half=False, imgsz=640, **kw)
    cfg.temporal.enabled = False  # disable temporal for pure detector tests
    for key in cfg.confidence:
        cfg.confidence[key] = 0.0
    return cfg


def test_detector_verifies_class_mapping(monkeypatch):
    _patch_env(monkeypatch)
    d = YoloBurnInDetector(config=_base_config())
    assert d.num_classes() == 4
    assert d.get_class_names() == FAKE_NAMES
    assert d.device == "cpu"


def test_detector_fails_fast_on_wrong_mapping(monkeypatch):
    class _Bad(_FakeModel):
        names = {0: "dog", 1: "cat", 2: "bird", 3: "fish"}

    _patch_env(monkeypatch, _Bad())
    with pytest.raises(ClassMappingError):
        YoloBurnInDetector(config=_base_config())


def test_detector_maps_detection_to_schema(monkeypatch):
    img = np.zeros((48, 64, 3), dtype=np.uint8)
    model = _FakeModel([make_result(img, [_Box([5, 10, 50, 35], 0.9, 0)])])
    _patch_env(monkeypatch, model)
    d = YoloBurnInDetector(config=_base_config())
    r = d.detect_frame(img, frame_index=2, timestamp_seconds=0.1)
    assert len(r.detections) == 1
    det = r.detections[0]
    assert det.class_name == "subtitle"
    assert det.class_id == 0
    assert det.frame_index == 2
    # Post-processing expands the box by a per-class margin, so the raw box is
    # preserved approximately (never shrunk below the original bounds).
    x1, y1, x2, y2 = det.bbox_xyxy
    assert x1 <= 5.0 and y1 <= 10.0
    assert x2 >= 50.0 and y2 >= 35.0
    assert det.frame_width == 64
    assert det.frame_height == 48


def test_detector_batch_handles_timestamps(monkeypatch):
    img0 = np.zeros((48, 64, 3), dtype=np.uint8)
    img1 = np.ones((48, 64, 3), dtype=np.uint8) * 10
    model = _FakeModel(
        [
            make_result(img0, [_Box([1, 2, 30, 20], 0.8, 1)]),
            make_result(img1, [_Box([5, 6, 40, 30], 0.7, 3)]),
        ]
    )
    _patch_env(monkeypatch, model)
    d = YoloBurnInDetector(config=_base_config())
    results = d.detect_batch(
        [img0, img1], frame_index_offset=0, timestamps_seconds=[0.0, 1.0]
    )
    assert len(results) == 2
    assert results[0].detections[0].class_name == "watermark"
    assert results[0].detections[0].timestamp_seconds == 0.0
    assert results[1].detections[0].class_name == "other_burn_in_text"
    assert results[1].detections[0].timestamp_seconds == 1.0


def test_detector_reuses_single_predict_call(monkeypatch):
    img = np.zeros((48, 64, 3), dtype=np.uint8)
    model = _FakeModel([make_result(img, [])])
    _patch_env(monkeypatch, model)
    d = YoloBurnInDetector(config=_base_config())
    d.detect_frame(img, frame_index=0)
    assert model.predict_calls == 1


def test_detector_empty_detections(monkeypatch):
    img = np.zeros((48, 64, 3), dtype=np.uint8)
    model = _FakeModel([make_result(img, [])])
    _patch_env(monkeypatch, model)
    d = YoloBurnInDetector(config=_base_config())
    r = d.detect_frame(img, frame_index=0)
    assert r.detections == []


def test_release_no_raise(monkeypatch):
    model = _FakeModel([])
    _patch_env(monkeypatch, model)
    d = YoloBurnInDetector(config=_base_config())
    d.release()
