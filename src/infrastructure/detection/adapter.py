"""Adapter between the new detection schema and the legacy ProposalBundle.

The existing hybrid OCR / region-detector pipeline consumes ``ProposalBundle``
objects whose ``DecodedFrame.detections`` are plain dicts with the required
keys ``frame_index``, ``bbox_xyxy``, ``timestamp_ms``, ``candidate_id``,
``confidence``. The new detector produces structured ``Detection`` objects with
per-class info (``class_id`` / ``class_name``). This adapter converts the new
output into the legacy dict format so the whole existing pipeline keeps working,
while preserving the richer class information as extra keys (which downstream
consumers safely ignore).
"""

from __future__ import annotations

from typing import List

from src.application.dto.proposal_bundle import ProposalBundle
from src.infrastructure.detection.schemas import Detection, DetectionResult


def detection_to_legacy_dict(
    detection: Detection, *, index: int
) -> dict:
    """Convert a :class:`Detection` to the legacy proposal dict format."""
    x1, y1, x2, y2 = detection.bbox_xyxy
    return {
        "frame_index": detection.frame_index,
        "timestamp_ms": round(detection.timestamp_seconds * 1000.0, 1),
        "candidate_id": f"F{detection.frame_index:05d}_C{index:02d}",
        "bbox_xyxy": detection.bbox_xyxy,
        "bbox_normalized": detection.bbox_normalized,
        "confidence": round(detection.confidence, 4),
        # Legacy consumers treat everything as burn-in text.
        "class": detection.class_name,
        # New per-class info preserved for consumers that want it.
        "class_id": int(detection.class_id),
        "class_name": str(detection.class_name),
    }


def results_to_proposal_bundle(
    video_path: str,
    fps: float,
    *,
    video_width: int,
    video_height: int,
    total_frames: int,
    results: List[DetectionResult],
    stride: int = 1,
) -> ProposalBundle:
    """Build a legacy :class:`ProposalBundle` from new detector results.

    The images carried on each :class:`DetectionResult` (via ``bgr_image``) are
    also attached so downstream consumers that pull frames from the bundle keep
    working.
    """
    bundle = ProposalBundle(
        video_path=str(video_path),
        fps=fps,
        stride=stride,
        video_width=video_width,
        video_height=video_height,
        total_frames=total_frames,
    )
    for result in results:
        legacy_detections = [
            detection_to_legacy_dict(det, index=i)
            for i, det in enumerate(result.detections)
        ]
        bundle.add_frame(
            frame_index=result.frame_index,
            timestamp_sec=result.timestamp_seconds,
            image=result.bgr_image,
            image_path=None,
            detections=legacy_detections,
        )
    return bundle
