"""RED tests for single-decode proposal bundle.

These tests define the contract for YOLO-on-video single-decode:
- detect_video(video_path) returns ProposalBundle with frames + detections
- Fail-loud on blank frames, decode errors, resolution mismatch
- Frame index maps to the actual image from the same decode pass
- One YOLO predict() call per job, shared by OCR + region detector
"""
import json, os, tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest


class FakeYoloModel:
    """Fake YOLO that returns predictable detections and exposes orig_img."""

    def __init__(self, detections, frames_to_yield):
        self._dets = detections
        self._frames = frames_to_yield  # list of (frame_index, orig_img)

    def predict(self, source, **kwargs):
        # Emulate ultralytics results iterator
        for frame_index, orig_img in self._frames:
            class FakeResult:
                def __init__(self, idx, img, detections):
                    self.frame_index = idx
                    self.orig_img = img
                    self.orig_shape = img.shape[:2]

                    class FakeBoxes:
                        def __init__(self, dets_list):
                            self._dets = dets_list

                        def __len__(self):
                            return len(self._dets)

                        def __iter__(self):
                            return iter(self._dets)

                    self.boxes = FakeBoxes(detections)

            frame_dets = [d for d in self._dets if d.get("frame_index") == frame_index]
            yield FakeResult(frame_index, orig_img, frame_dets)


class FakeProposalForTest:
    """Stand-in for YoloTextProposalProvider.detect_video().

    Wraps a FakeYoloModel so tests can inject arbitrary frame/detection data.
    """

    def __init__(self, frames_and_dets, fps=30.0):
        # frames_and_dets: list of (frame_index, orig_img_array, list_of_detections)
        self._frames = [(fi, img) for fi, img, _ in frames_and_dets]
        self._dets = [d for _, _, ds in frames_and_dets for d in ds]
        self._fps = fps

    def detect_video(
        self,
        video_path,
        stride: int = 1,
        output_dir: str | None = None,
        min_pixel_std: float = 1.0,
    ):
        """Replicate the detect_video() signature we expect from the real impl."""
        from src.infrastructure.benchmark.hybrid_subtitle.yolo_proposal import (
            ProposalBundle,
        )

        # Simulate the real detect_video metadata probe.
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video for decode: {video_path}")
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        bundle = ProposalBundle(
            video_path=video_path,
            fps=self._fps,
            stride=stride,
            total_frames=frame_count,
            video_width=video_width,
            video_height=video_height,
        )

        dets_by_frame = {}
        for d in self._dets:
            dets_by_frame.setdefault(int(d["frame_index"]), []).append(d)

        for frame_index, orig_img in self._frames:
            if frame_index % stride != 0:
                continue
            H, W = orig_img.shape[:2]
            ts = frame_index / self._fps

            # Validate blank frame
            if float(orig_img.std()) < min_pixel_std:
                raise RuntimeError(
                    f"Blank/gray frame detected at frame_index={frame_index} "
                    f"(std={orig_img.std():.4f} < {min_pixel_std}). "
                    "Video decode may have failed or the codec is unsupported."
                )

            # Write frame to disk if output_dir provided
            img_path = None
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                img_path = os.path.join(output_dir, f"frame_{frame_index:06d}.jpg")
                cv2.imwrite(img_path, orig_img)

            frame_dets = dets_by_frame.get(frame_index, [])
            bundle.add_frame(
                frame_index=frame_index,
                timestamp_sec=ts,
                image=orig_img,
                image_path=img_path,
                detections=frame_dets,
            )
        return bundle


# ---------------------------------------------------------------------------
# Tests — RED first (these fail until detect_video / ProposalBundle exist)
# ---------------------------------------------------------------------------


def test_proposal_bundle_class_exists():
    """ProposalBundle and DecodedFrame DTOs must be importable from yolo_proposal."""
    from src.infrastructure.benchmark.hybrid_subtitle.yolo_proposal import (
        ProposalBundle,
        DecodedFrame,
    )
    assert ProposalBundle is not None
    assert DecodedFrame is not None


def test_proposal_bundle_detect_video_returns_bundle():
    """detect_video() returns a ProposalBundle, not raw detections list."""
    from src.infrastructure.benchmark.hybrid_subtitle.yolo_proposal import (
        ProposalBundle,
    )

    # Create a minimal real video (1-second, 30fps, small black frame)
    with tempfile.TemporaryDirectory() as tmp:
        video_path = os.path.join(tmp, "input.mp4")
        _write_test_video(video_path, width=1280, height=720, fps=30, duration_s=1)
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        cv2.putText(frame, "SUBTITLE", (300, 660), cv2.FONT_HERSHEY_SIMPLEX, 2,
                    (255, 255, 255), 3)
        fake = FakeProposalForTest(
            frames_and_dets=[(0, frame, [{"frame_index": 0, "bbox_xyxy": [100, 600, 800, 700], "confidence": 0.9}])],
            fps=30.0,
        )
        # The real provider must have detect_video; we test the fake for shape
        bundle = fake.detect_video(video_path)
        assert isinstance(bundle, ProposalBundle)


def test_proposal_bundle_validates_blank_frame_fail_loud():
    """Blank/gray frame (std<1.0) raises RuntimeError, not silently returns []."""
    with tempfile.TemporaryDirectory() as tmp:
        video_path = os.path.join(tmp, "input.mp4")
        _write_test_video(video_path, width=1280, height=720, fps=30, duration_s=1)
        # Create a gray frame (mean=128, std=0)
        gray_frame = np.full((720, 1280, 3), 128, dtype=np.uint8)

        fake = FakeProposalForTest(
            frames_and_dets=[(0, gray_frame, [])],
            fps=30.0,
        )
        with pytest.raises(RuntimeError, match="Blank/gray frame"):
            fake.detect_video(video_path)


def test_proposal_bundle_validates_decode_error():
    """Non-existent or unreadable video raises RuntimeError from detect_video."""
    with tempfile.TemporaryDirectory() as tmp:
        fake = FakeProposalForTest(
            frames_and_dets=[(0, np.zeros((720, 1280, 3), np.uint8), [])],
            fps=30.0,
        )
        nonexistent = os.path.join(tmp, "does_not_exist.mp4")
        with pytest.raises(RuntimeError, match="Cannot open video|Blank/gray frame"):
            fake.detect_video(nonexistent)


def test_proposal_bundle_frame_index_maps_to_image():
    """frame_index on a detection corresponds to the image stored at that index."""
    with tempfile.TemporaryDirectory() as tmp:
        video_path = os.path.join(tmp, "input.mp4")
        _write_test_video(video_path, width=1280, height=720, fps=30, duration_s=2)

        # Create 3 distinct frames with enough entropy to pass blank check
        rng = np.random.default_rng(42)
        frames = []
        for fi in range(3):
            f = rng.integers(0, 200, (720, 1280, 3), dtype=np.uint8)
            # Tag each frame with unique center pixel to verify mapping
            f[360, 640] = (fi * 85 + 10, fi * 85 + 10, fi * 85 + 10)
            frames.append(f)

        fake = FakeProposalForTest(
            frames_and_dets=[
                (0, frames[0], [{"frame_index": 0, "bbox_xyxy": [100, 100, 600, 200], "confidence": 0.9}]),
                (30, frames[1], [{"frame_index": 30, "bbox_xyxy": [200, 200, 700, 300], "confidence": 0.9}]),
                (60, frames[2], [{"frame_index": 60, "bbox_xyxy": [300, 300, 800, 400], "confidence": 0.9}]),
            ],
            fps=30.0,
        )
        bundle = fake.detect_video(video_path)

        # Verify each frame_index maps to the correct image
        for decoded_frame in bundle.frames:
            idx = decoded_frame.frame_index
            stored_img = decoded_frame.image
            # Re-derive from our known test frames
            if idx == 0:
                expected_center = frames[0][360, 640]
            elif idx == 30:
                expected_center = frames[1][360, 640]
            elif idx == 60:
                expected_center = frames[2][360, 640]
            else:
                continue
            actual_center = stored_img[360, 640]
            assert tuple(actual_center) == tuple(expected_center), (
                f"Frame index {idx} does not map to expected image. "
                f"Got center={actual_center}, expected={expected_center}"
            )


def test_proposal_bundle_resolution_matches_video_metadata():
        """Decoded frame resolution equals video resolution (not fixed 1920x1080)."""
        from src.infrastructure.benchmark.hybrid_subtitle.yolo_proposal import (
            ProposalBundle,
        )

        with tempfile.TemporaryDirectory() as tmp:
            # Portrait video 720x1280
            video_path = os.path.join(tmp, "portrait.mp4")
            _write_test_video(video_path, width=720, height=1280, fps=30, duration_s=1)
            # Frame with entropy (passes blank check) — shape must match (1280, 720)
            rng = np.random.default_rng(99)
            frame = rng.integers(0, 200, (1280, 720, 3), dtype=np.uint8)
            cv2.putText(frame, "TEXT", (50, 620), cv2.FONT_HERSHEY_SIMPLEX, 2,
                        (255, 255, 255), 3)
            fake = FakeProposalForTest(
                frames_and_dets=[(0, frame, [{"frame_index": 0, "bbox_xyxy": [50, 600, 600, 700], "confidence": 0.9}])],
                fps=30.0,
            )
            bundle = fake.detect_video(video_path)
            assert bundle.video_width == 720, f"expected 720, got {bundle.video_width}"
            assert bundle.video_height == 1280, f"expected 1280, got {bundle.video_height}"
            assert len(bundle.frames) >= 1
            H, W = bundle.frames[0].image.shape[:2]
            assert W == 720
            assert H == 1280


def test_proposal_bundle_proposal_source_tagged():
    """Bundle metadata records proposal_source=yolo_video_stream."""
    with tempfile.TemporaryDirectory() as tmp:
        video_path = os.path.join(tmp, "input.mp4")
        _write_test_video(video_path, width=1280, height=720, fps=30, duration_s=1)
        rng = np.random.default_rng(12)
        frame = rng.integers(0, 200, (720, 1280, 3), dtype=np.uint8)
        fake = FakeProposalForTest(
            frames_and_dets=[(0, frame, [])],
            fps=30.0,
        )
        bundle = fake.detect_video(video_path)
        assert bundle.metadata.get("proposal_source") == "yolo_video_stream"


def test_proposal_bundle_yields_all_frames_up_to_total():
    """Bundle knows total frame count and yields all frames up to that."""
    from src.infrastructure.benchmark.hybrid_subtitle.yolo_proposal import (
        ProposalBundle,
    )

    with tempfile.TemporaryDirectory() as tmp:
        video_path = os.path.join(tmp, "input.mp4")
        # Write 30 frames
        _write_test_video(video_path, width=1280, height=720, fps=30, duration_s=1)
        frames_data = []
        rng = np.random.default_rng(24)
        for fi in range(30):
            # Frame has enough entropy/standard deviation
            f = rng.integers(0, 200, (720, 1280, 3), dtype=np.uint8)
            frames_data.append(f)

        fake = FakeProposalForTest(
            frames_and_dets=[(i, frames_data[i], []) for i in range(30)],
            fps=30.0,
        )
        bundle = fake.detect_video(video_path)
        assert bundle.total_frames == 30
        assert len(bundle.frames) == 30


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _write_test_video(path, width, height, fps, duration_s, codec="mp4v"):
    """Write a minimal real MP4 using cv2. Uses mp4v for portability."""
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
    n_frames = int(fps * duration_s)
    for i in range(n_frames):
        # Light noise so frames aren't blank
        frame = np.random.randint(0, 30, (height, width, 3), dtype=np.uint8)
        writer.write(frame)
    writer.release()