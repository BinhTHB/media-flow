"""OCR that reuses hybrid YOLO and Gemini verifier transcriptions."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import cv2

from src.application.ports.ocr_provider import OCRProvider
from src.domain.entities.frame import Frame
from src.domain.entities.segment import Segment
from src.infrastructure.benchmark.hybrid_subtitle.core import (
    build_tracks,
    group_frame_candidates,
)


class HybridGeminiOCRProvider(OCRProvider):
    """Emit OCR segments from accepted hybrid-verifier transcriptions."""

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

    def detect_text(self, frames: List[Frame]) -> List[Segment]:
        if not frames or self._semantic_verifier is None:
            return []
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
        images = [
            image for group in representative
            if (image := cv2.imread(frames[group["frame_index"]].image_path)) is not None
        ]
        if not images:
            return None
        candidate_ids = [group["candidate_group_id"] for group in representative]
        candidate_bboxes = [group["union_bbox_xyxy"] for group in representative]
        verdict = self._semantic_verifier.classify_track(
            _TrackRef(track["track_id"]), images, candidate_ids, candidate_bboxes
        )
        text = _subtitle_transcription(verdict.get("decisions", []))
        if text is None:
            return None
        first, last = groups[0]["frame_index"], groups[-1]["frame_index"]
        box = track["temporal_union_bbox_xyxy"]
        return Segment(
            id=f"ocr_{track['track_id']}",
            start=frames[first].timestamp,
            end=frames[last].timestamp,
            text=text,
            source="ocr",
            bbox={"left": box[0], "top": box[1], "right": box[2], "bottom": box[3]},
            metadata={"provider": self.__class__.__name__, "track_id": track["track_id"]},
        )


class _TrackRef:
    def __init__(self, track_id: str) -> None:
        self.track_id = track_id


def _subtitle_transcription(decisions: List[Dict[str, Any]]) -> Optional[str]:
    for decision in decisions:
        text = decision.get("transcription") if decision.get("is_subtitle") else None
        if isinstance(text, str) and text.strip():
            return text.strip()
    return None
