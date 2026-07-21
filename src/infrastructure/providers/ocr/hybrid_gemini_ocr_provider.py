from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import cv2

from src.application.dto.proposal_bundle import ProposalBundle
from src.application.ports.ocr_provider import OCRProvider
from src.domain.entities.frame import Frame
from src.domain.entities.segment import Segment
from src.infrastructure.benchmark.hybrid_subtitle.core import (
    build_tracks,
    group_frame_candidates,
)


class HybridGeminiOCRProvider(OCRProvider):
    """Emit OCR segments from accepted hybrid-verifier transcriptions.

    Supports two modes:
    1. **Legacy** (``proposal_bundle`` is None): calls ``detect_images``
       on the interval-extracted frame paths, triggering a separate YOLO pass.
    2. **Single-decode** (``proposal_bundle`` is set): reuses the pre-computed
       detections and frame images from one YOLO-on-video pass,
       avoiding a second decode + predict.

    Set ``proposal_bundle`` before calling ``detect_text()`` to use mode 2.
    """

    def __init__(
        self,
        proposal_provider: Any,
        semantic_verifier: Any,
        min_confidence: float = 0.30,
        track_iou_threshold: float = 0.25,
        track_max_gap_frames: int = 5,
    ) -> None:
        self._proposal = proposal_provider
        self._semantic_verifier = semantic_verifier
        self._min_confidence = min_confidence
        self._track_iou_threshold = track_iou_threshold
        self._track_max_gap_frames = track_max_gap_frames
        self.proposal_bundle: ProposalBundle | None = None

    def detect_text(self, frames: List[Frame]) -> List[Segment]:
        if not frames or self._semantic_verifier is None:
            return []

        if self.proposal_bundle is not None:
            # Mode 2: Single-decode Proposal Bundle
            detections = [
                d for d in self.proposal_bundle.detections
                if float(d.get("confidence", 0.0)) >= self._min_confidence
            ]
        else:
            # Mode 1: Legacy frame-extraction path
            detections = [
                item for item in self._proposal.detect_images([frame.image_path for frame in frames])
                if float(item.get("confidence", 0.0)) >= self._min_confidence
            ]

        by_frame: Dict[int, List[Dict[str, Any]]] = {}
        for detection in detections:
            by_frame.setdefault(int(detection["frame_index"]), []).append(detection)
        groups = [
            group
            for frame_index in sorted(by_frame)
            for group in group_frame_candidates(by_frame[frame_index])
        ]
        tracks = build_tracks(
            groups,
            iou_threshold=self._track_iou_threshold,
            max_gap_frames=self._track_max_gap_frames,
        )
        return [
            segment
            for track in tracks
            if (segment := self._segment_for_track(track, frames)) is not None
        ]

    def _segment_for_track(
        self, track: Dict[str, Any], frames: List[Frame]
    ) -> Optional[Segment]:
        groups = track["groups"]
        representative = groups if len(groups) <= 3 else [groups[0], groups[len(groups) // 2], groups[-1]]

        images = []
        for group in representative:
            fi = group["frame_index"]
            if self.proposal_bundle is not None:
                try:
                    decoded_frame = self.proposal_bundle.frame_for_index(fi)
                except KeyError as exc:
                    raise RuntimeError(
                        f"Single-decode bundle is missing frame_index={fi} required by OCR track."
                    ) from exc
                # No seek/disk fallback: this must be the exact YOLO orig_img.
                images.append(decoded_frame.image)
            else:
                if fi < len(frames):
                    img = cv2.imread(frames[fi].image_path)
                    if img is not None:
                        images.append(img)

        if not images or len(images) < len(representative):
            return None
        candidate_ids = [group["candidate_group_id"] for group in representative]
        candidate_bboxes = [group["union_bbox_xyxy"] for group in representative]
        verdict = self._semantic_verifier.classify_track(
            _TrackRef(track["track_id"]), images, candidate_ids, candidate_bboxes
        )
        res = _subtitle_transcription_and_confidence(verdict.get("decisions", []))
        text, confidence = res
        if text is None:
            return None
        first, last = groups[0]["frame_index"], groups[-1]["frame_index"]
        if first == last:
            # Normalization drops a single-frame track when its end equals start.
            # Derive cadence from the available frames to give it a real duration.
            cadence = (
                (frames[-1].timestamp - frames[0].timestamp) / (len(frames) - 1)
                if len(frames) >= 2
                else 1.0
            )
            end = frames[first].timestamp + cadence
        else:
            end = frames[last].timestamp
        box = track["temporal_union_bbox_xyxy"]
        return Segment(
            id=f"ocr_{track['track_id']}",
            start=frames[first].timestamp,
            end=end,
            text=text,
            source="ocr",
            confidence=confidence,
            bbox={"left": box[0], "top": box[1], "right": box[2], "bottom": box[3]},
            metadata={"provider": self.__class__.__name__, "track_id": track["track_id"]},
        )


class _TrackRef:
    def __init__(self, track_id: str) -> None:
        self.track_id = track_id


def _subtitle_transcription_and_confidence(decisions: List[Dict[str, Any]]) -> Tuple[Optional[str], float]:
    for decision in decisions:
        text = decision.get("transcription") if decision.get("is_subtitle") else None
        if isinstance(text, str) and text.strip():
            return text.strip(), float(decision.get("confidence", 0.0))
    return None, 0.0
