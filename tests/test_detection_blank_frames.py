"""Detection must tolerate blank/solid frames.

Fades, title cards and end cards are normal video content with pixel_std near
zero. The detector used to raise and kill the whole job; it must skip those
frames and keep the rest of the video usable.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.infrastructure.detection.yolo_proposal_provider import YoloTextProposalProvider


class _FakeBox:
    def __init__(self, xyxy, conf, cls_id=0):
        self.xyxy = [np.array(xyxy, dtype=float)]
        self.conf = [float(conf)]
        self.cls = [int(cls_id)]


class _FakeResult:
    def __init__(self, image, boxes):
        self.orig_img = image
        self.orig_shape = image.shape[:2]
        self.boxes = boxes


class _FakeModel:
    def __init__(self, frames):
        self._frames = frames

    def predict(self, *args, **kwargs):
        return iter(self._frames)


def _provider_with(frames):
    provider = YoloTextProposalProvider.__new__(YoloTextProposalProvider)
    provider._model = _FakeModel(frames)
    provider._conf = 0.05
    provider._iou = 0.7
    provider._imgsz = 640
    provider._device = "cpu"
    provider._class_names = {0: "subtitle"}
    return provider


@pytest.fixture()
def fake_video(monkeypatch):
    """Report a 3-frame 64x64 video without touching the filesystem."""
    import cv2

    class _Cap:
        def isOpened(self):
            return True

        def get(self, prop):
            return {
                cv2.CAP_PROP_FPS: 30.0,
                cv2.CAP_PROP_FRAME_COUNT: 3,
                cv2.CAP_PROP_FRAME_WIDTH: 64,
                cv2.CAP_PROP_FRAME_HEIGHT: 64,
            }.get(prop, 0)

        def release(self):
            pass

    monkeypatch.setattr(cv2, "VideoCapture", lambda _p: _Cap())


def _textured(width: int = 64, height: int = 64) -> np.ndarray:
    """A frame with real pixel variance (a uniform fill counts as blank)."""
    rng = np.random.default_rng(1234)
    return rng.integers(0, 255, size=(height, width, 3), dtype=np.uint8)


def test_blank_frame_is_skipped_not_fatal(fake_video):
    """A black frame in the middle must not raise."""
    bright = _textured()
    black = np.zeros((64, 64, 3), dtype=np.uint8)

    frames = [
        _FakeResult(bright, [_FakeBox([1, 1, 20, 10], 0.9)]),
        _FakeResult(black, []),
        _FakeResult(bright, [_FakeBox([1, 1, 20, 10], 0.9)]),
    ]
    bundle = _provider_with(frames).detect_video("fake.mp4", stride=1, retain_images=False)

    # Two usable frames survive; the blank one is simply absent.
    assert len(bundle.frames) == 2
    assert [f.frame_index for f in bundle.frames] == [0, 2]


def test_all_blank_frames_still_raises(fake_video):
    """A fully blank video is a real error, not an empty success."""
    black = np.zeros((64, 64, 3), dtype=np.uint8)
    frames = [_FakeResult(black, []) for _ in range(3)]

    with pytest.raises(RuntimeError, match="0 frames"):
        _provider_with(frames).detect_video("fake.mp4", stride=1, retain_images=False)
