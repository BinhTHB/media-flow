import os
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from src.application.dto.proposal_bundle import ProposalBundle
from src.infrastructure.benchmark.hybrid_subtitle.yolo_proposal import (
    YoloTextProposalProvider,
)


class _Tensor:
    def __init__(self, values):
        self._values = values

    def tolist(self):
        return self._values

    def __float__(self):
        # Support conversion for float(box.conf[0])
        if isinstance(self._values, (int, float)):
            return float(self._values)
        if isinstance(self._values, list) and len(self._values) == 1:
            return float(self._values[0])
        return float(self._values)



class _Box:
    def __init__(self, xyxy, confidence):
        self.xyxy = [_Tensor(xyxy)]
        self.conf = [_Tensor(confidence)]


class _Result:
    def __init__(self, image, boxes):
        self.orig_img = image
        self.orig_shape = image.shape[:2]
        self.boxes = boxes


class _Model:
    def __init__(self, results):
        self.results = results
        self.calls = 0

    def predict(self, *_args, **_kwargs):
        self.calls += 1
        return iter(self.results)


def _video(path, width=64, height=48, frame_count=3):
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (width, height)
    )
    for index in range(frame_count):
        image = np.full((height, width, 3), 20 + index * 20, dtype=np.uint8)
        image[:, ::2] = 220
        writer.write(image)
    writer.release()


def _provider(model):
    provider = YoloTextProposalProvider.__new__(YoloTextProposalProvider)
    provider._model = model
    provider._conf = 0.05
    provider._iou = 0.70
    provider._imgsz = 1024
    provider._device = "cpu"
    return provider


def test_detect_video_materializes_yolo_orig_img_and_bbox_same_frame(tmp_path):
    video = tmp_path / "input.mp4"
    frames_dir = tmp_path / "frames"
    _video(video)
    first = np.full((48, 64, 3), 15, dtype=np.uint8)
    first[:, ::2] = 215
    second = np.full((48, 64, 3), 25, dtype=np.uint8)
    second[:, ::2] = 205
    model = _Model([
        _Result(first, [_Box([5, 10, 50, 35], 0.90)]),
        _Result(second, [_Box([6, 11, 51, 36], 0.80)]),
        _Result(second, []),
    ])

    bundle = _provider(model).detect_video(str(video), output_dir=str(frames_dir))

    assert model.calls == 1
    assert bundle.metadata["proposal_source"] == "yolo_video_stream"
    assert bundle.video_width == 64
    assert bundle.video_height == 48
    assert len(bundle.frames) == 3
    assert bundle.frames[0].detections[0]["frame_index"] == 0
    assert bundle.frames[0].detections[0]["bbox_xyxy"] == [5.0, 10.0, 50.0, 35.0]
    assert np.array_equal(bundle.frame_for_index(0).image, first)
    written = cv2.imread(bundle.frame_for_index(0).image_path)
    assert written is not None
    assert written.shape[:2] == first.shape[:2]
    assert written.std() > 1.0


def test_detect_video_rejects_blank_yolo_frame(tmp_path):
    video = tmp_path / "input.mp4"
    _video(video, frame_count=1)
    blank = np.full((48, 64, 3), 128, dtype=np.uint8)

    with pytest.raises(RuntimeError, match="Blank/gray frame"):
        _provider(_Model([_Result(blank, [])])).detect_video(str(video))


def test_detect_video_rejects_resolution_mismatch(tmp_path):
    video = tmp_path / "input.mp4"
    _video(video, width=64, height=48, frame_count=1)
    wrong_size = np.full((24, 32, 3), 50, dtype=np.uint8)
    wrong_size[:, ::2] = 200

    with pytest.raises(RuntimeError, match="resolution mismatch"):
        _provider(_Model([_Result(wrong_size, [])])).detect_video(str(video))


def test_detect_video_rejects_unreadable_video(tmp_path):
    with pytest.raises(RuntimeError, match="Cannot open video"):
        _provider(_Model([])).detect_video(str(tmp_path / "missing.mp4"))


def test_bundle_can_be_shared_without_second_predict(tmp_path):
    video = tmp_path / "input.mp4"
    _video(video)
    image = np.full((48, 64, 3), 30, dtype=np.uint8)
    image[:, ::2] = 210
    model = _Model([_Result(image, []) for _ in range(3)])
    proposal = _provider(model)

    bundle = proposal.detect_video(str(video))
    # OCR and region path receive the same object; only orchestration owns predict.
    ocr_consumer = SimpleNamespace(proposal_bundle=bundle)
    region_consumer = SimpleNamespace(proposal_bundle=bundle)
    assert ocr_consumer.proposal_bundle is region_consumer.proposal_bundle
    assert model.calls == 1
    assert isinstance(bundle, ProposalBundle)
    assert len(bundle.frames) == 3
