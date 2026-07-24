from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import cv2
from pathlib import Path

from src.application.dto.proposal_bundle import ProposalBundle
from src.application.ports.ocr_provider import OCRProvider
from src.domain.entities.frame import Frame
from src.domain.entities.segment import Segment
from src.infrastructure.benchmark.hybrid_subtitle.core import (
    build_tracks,
    group_frame_candidates,
)
from src.infrastructure.benchmark.hybrid_subtitle.subtitle_classifier import (
    HybridSubtitleClassifier,
    HeuristicSubtitleClassifier,
)
from src.infrastructure.benchmark.hybrid_subtitle.behavior_discriminator import (
    TrackBehaviorDiscriminator,
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
        min_confidence: float = 0.40,
        track_iou_threshold: float = 0.25,
        track_max_gap_frames: int = 5,
        crop_batch_classifier: Any = None,
    ) -> None:
        self._proposal = proposal_provider
        self._semantic_verifier = semantic_verifier
        self._min_confidence = min_confidence
        self._track_iou_threshold = track_iou_threshold
        self._track_max_gap_frames = track_max_gap_frames
        self._crop_batch_classifier = crop_batch_classifier
        self._behavior_discriminator: TrackBehaviorDiscriminator | None = None
        self.behavior_ledger: list[dict[str, Any]] = []
        self.proposal_bundle: ProposalBundle | None = None

    def set_speech_segments(
        self, speech_segments: list[Any], video_duration_seconds: float
    ) -> None:
        self._behavior_discriminator = TrackBehaviorDiscriminator(
            speech_segments, video_duration_seconds
        )

    def set_run_context(self, job_id: str, artifact_dir: str) -> None:
        if self._crop_batch_classifier is not None:
            self._crop_batch_classifier.artifact_dir = Path(artifact_dir)
            self._crop_batch_classifier.set_run_context(job_id, job_id)

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
        crop_decisions = self._crop_decisions_for_uncertain_tracks(tracks, frames)
        return [
            segment
            for track in tracks
            if (
                segment := self._segment_for_track(
                    track, frames, crop_decisions.get(track["track_id"])
                )
            ) is not None
        ]

    def _crop_decisions_for_uncertain_tracks(
        self, tracks: List[Dict[str, Any]], frames: List[Frame]
    ) -> Dict[str, Dict[str, Any]]:
        if self.proposal_bundle is None or self._crop_batch_classifier is None:
            return {}
        frame_height = self.proposal_bundle.video_height
        heuristic = HeuristicSubtitleClassifier()
        uncertain = []
        canaries = []
        self.behavior_ledger = []
        for track in tracks:
            if heuristic.classify(track, frame_height).is_subtitle:
                continue
            if self._behavior_discriminator is not None:
                evidence = self._behavior_discriminator.evaluate_static_content(
                    track, self.proposal_bundle
                )
                self.behavior_ledger.append(
                    {
                        "track_id": track["track_id"],
                        **evidence.__dict__,
                        "ocr_text_samples": list(evidence.ocr_text_samples),
                    }
                )
                if evidence.rule_fired:
                    continue
                track = {**track, "behavior": evidence.__dict__}
            duration_ms = (
                track["groups"][-1]["timestamp_ms"]
                - track["groups"][0]["timestamp_ms"]
            )
            bbox = (
                track.get("median_bbox_xyxy")
                or track.get("temporal_union_bbox_xyxy")
                or track["groups"][-1].get("union_bbox_xyxy", [])
            )
            bottom_ratio = bbox[3] / frame_height if len(bbox) == 4 else 0.0
            if (
                duration_ms >= 3000
                and bottom_ratio < HeuristicSubtitleClassifier.BOTTOM_RATIO_THRESHOLD
            ):
                canaries.append(track)
                continue
            if len(track["groups"]) < 2:
                if bottom_ratio < HeuristicSubtitleClassifier.BOTTOM_RATIO_THRESHOLD:
                    canaries.append(track)
                # Short bottom tracks are handled by the semantic fallback.
                else:
                    uncertain.append(track)
            else:
                uncertain.append(track)
        if not uncertain:
            return {}
        if not canaries:
            return {}
        return self._crop_batch_classifier.classify_batch(
            uncertain, self.proposal_bundle, batch_size=10, canary_tracks=canaries[:2]
        )

    def _segment_for_track(
        self,
        track: Dict[str, Any],
        frames: List[Frame],
        crop_decision: Dict[str, Any] | None = None,
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
        if crop_decision is not None:
            verdict = {
                "decisions": [crop_decision]
                if crop_decision.get("is_subtitle") is True
                else [],
            }
        else:
            verdict = HybridSubtitleClassifier(
                gemini_verifier=self._semantic_verifier,
                frame_height=images[0].shape[0],
                verify_short_bottom_tracks=True,
            ).classify_track(
                track, images, candidate_ids, candidate_bboxes
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



def _subtitle_transcription_and_confidence(decisions: List[Dict[str, Any]]) -> Tuple[Optional[str], float]:
    for decision in decisions:
        text = decision.get("transcription") if decision.get("is_subtitle") else None
        if isinstance(text, str) and text.strip():
            return text.strip(), float(decision.get("confidence", 0.0))
    return None, 0.0
