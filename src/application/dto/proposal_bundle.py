from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class DecodedFrame:
    """A frame and its YOLO detections from one video-stream decode pass.

    Optionally stores the original full-res image plus a 720p context
    version and pre-extracted full-res crops per detection (STEP-0 cache).
    Detection-only callers can disable image retention to keep long videos bounded.
    """

    frame_index: int
    timestamp_sec: float
    image: Any | None
    image_path: str | None
    detections: list[dict[str, Any]] = field(default_factory=list)
    image_720p: Any | None = None  # Downscale ~720px max dimension (context for Gemini)
    crops: list[Any] = field(default_factory=list)  # Full-res crops per detection (JPEG q85)


@dataclass
class ProposalBundle:
    """Single-decode YOLO proposal artifact shared by OCR and region detection."""

    video_path: str
    fps: float
    stride: int
    video_width: int = 0
    video_height: int = 0
    total_frames: int = 0
    frames: list[DecodedFrame] = field(default_factory=list)
    metadata: dict[str, Any] = field(
        default_factory=lambda: {"proposal_source": "yolo_video_stream"}
    )

    @property
    def detections(self) -> list[dict[str, Any]]:
        return [
            detection
            for frame in self.frames
            for detection in frame.detections
        ]

    def add_frame(
        self,
        *,
        frame_index: int,
        timestamp_sec: float,
        image: Any | None,
        image_path: str | None,
        detections: list[dict[str, Any]],
    ) -> None:
        self.frames.append(
            DecodedFrame(
                frame_index=frame_index,
                timestamp_sec=timestamp_sec,
                image=image,
                image_path=image_path,
                detections=detections,
            )
        )

    def frame_for_index(self, frame_index: int) -> DecodedFrame:
        for frame in self.frames:
            if frame.frame_index == frame_index:
                return frame
        raise KeyError(f"Proposal bundle has no frame_index={frame_index}")

    def to_metadata(self) -> dict[str, Any]:
        return {
            **self.metadata,
            "video_path": self.video_path,
            "fps": self.fps,
            "stride": self.stride,
            "video_width": self.video_width,
            "video_height": self.video_height,
            "n_frames_decoded": len(self.frames),
            "n_detections": len(self.detections),
            "total_frames": self.total_frames,
        }
