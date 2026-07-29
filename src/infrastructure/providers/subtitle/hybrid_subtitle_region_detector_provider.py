"""V4 text-track detector and mask geometry producer."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2

from src.application.dto.proposal_bundle import ProposalBundle
from src.application.ports.subtitle_region_detector_provider import SubtitleRegionDetectorProvider
from src.application.services.mask_authority import DecisionStore
from src.domain.entities.frame import Frame
from src.domain.entities.subtitle_region import SubtitleRegion
from src.infrastructure.benchmark.hybrid_subtitle.bottom_band_gate import passes_bottom_band, track_y1_norm
from src.infrastructure.benchmark.hybrid_subtitle.core import build_tracks, group_frame_candidates
from src.infrastructure.benchmark.hybrid_subtitle.pixel_variance_discriminator import PixelVarianceDiscriminator

logger = logging.getLogger(__name__)
MAX_CLASSIFICATION_CANDIDATES = 26
OVERFLOW_BAND_GATE_THRESHOLD = 0.80


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


class HybridSubtitleRegionDetectorProvider(SubtitleRegionDetectorProvider):
    """Find text tracks, filter locally, classify by track ID, then emit geometry.

    This provider never labels geometry itself. It records one Gemini verdict per
    ``track_id`` and creates regions only for positive track decisions.
    """

    def __init__(
        self,
        weights: str = "best.pt",
        conf: float = 0.05,
        iou: float = 0.70,
        imgsz: int = 1024,
        horizontal_gap_ratio: float = 2.0,
        track_iou_threshold: float = 0.25,
        track_max_gap_frames: int = 5,
        bottom_ratio_threshold: float = 0.65,
        band_gate_threshold: float = 0.68,
        static_variance_threshold: float = 827.12,
        min_confidence: float = 0.40,
        proposal_provider: Optional[Any] = None,
        device: str = "cpu",
        semantic_verifier: Optional[Any] = None,
        classifier: Optional[Any] = None,
        crop_batch_classifier: Optional[Any] = None,
        decision_store: DecisionStore | None = None,
    ) -> None:
        del bottom_ratio_threshold, semantic_verifier, classifier
        self._weights = weights
        self._conf = conf
        self._iou = iou
        self._imgsz = imgsz
        self._h_gap_ratio = horizontal_gap_ratio
        self._track_iou_threshold = track_iou_threshold
        self._track_max_gap_frames = track_max_gap_frames
        self._band_gate_threshold = band_gate_threshold
        self._static_variance_threshold = static_variance_threshold
        self._min_confidence = min_confidence
        self._proposal = proposal_provider
        self._device = device
        self._crop_batch_classifier = crop_batch_classifier
        self._decision_store = decision_store
        self.proposal_bundle: ProposalBundle | None = None
        self.static_variance_ledger: list[dict[str, Any]] = []
        self.band_gate_ledger: list[dict[str, Any]] = []

    def set_decision_store(self, decision_store: DecisionStore) -> None:
        self._decision_store = decision_store

    def set_run_context(self, job_id: str, artifact_dir: str) -> None:
        if self._crop_batch_classifier is not None:
            self._crop_batch_classifier.artifact_dir = Path(artifact_dir)
            self._crop_batch_classifier.set_run_context(job_id, job_id)

    def static_rule_ledger(self) -> list[dict[str, Any]]:
        return list(self.static_variance_ledger)

    def detect_regions(self, frames: List[Frame]) -> List[SubtitleRegion]:
        if not frames:
            return []
        if self._decision_store is None:
            logger.error("LOUD-WARN: decision store unavailable, no masks will be emitted.")
            return []
        bundle, detections, width, height = self._load_detection_data(frames)
        tracks = self._build_tracks(detections)
        if not tracks:
            return []
        candidates = self._cheap_filter(tracks, bundle, height)
        if len(candidates) > MAX_CLASSIFICATION_CANDIDATES:
            logger.warning(
                "V4 candidate overflow: %d candidates exceed capacity %d. "
                "Reapplying cheap filter with deterministic bottom-band threshold %.2f.",
                len(candidates),
                MAX_CLASSIFICATION_CANDIDATES,
                OVERFLOW_BAND_GATE_THRESHOLD,
            )
            candidates = self._cheap_filter(
                tracks,
                bundle,
                height,
                band_gate_threshold=OVERFLOW_BAND_GATE_THRESHOLD,
            )
        if len(candidates) > MAX_CLASSIFICATION_CANDIDATES:
            raise RuntimeError(
                f"V4 candidate overflow: {len(candidates)} candidates remain after "
                f"deterministic bottom-band tightening to "
                f"y1_norm >= {OVERFLOW_BAND_GATE_THRESHOLD:.2f}; capacity is "
                f"{MAX_CLASSIFICATION_CANDIDATES}. Refusing to drop candidates."
            )
        if not candidates:
            return []
        if self._crop_batch_classifier is None:
            logger.error("LOUD-WARN: Gemini track classifier unavailable, all candidates rejected.")
            return []
        try:
            decisions = self._crop_batch_classifier.classify_batch(candidates, bundle)
        except Exception as exc:
            logger.error("LOUD-WARN: Gemini classification failed, all candidates rejected: %s", exc)
            return []
        self._record_decisions(candidates, decisions)
        return self._regions_for_positive_tracks(candidates, frames, width, height)

    def _load_detection_data(
        self, frames: List[Frame]
    ) -> tuple[ProposalBundle, List[Dict[str, Any]], int, int]:
        if self.proposal_bundle is not None:
            return (
                self.proposal_bundle,
                list(self.proposal_bundle.detections),
                self.proposal_bundle.video_width,
                self.proposal_bundle.video_height,
            )
        if self._proposal is None:
            from src.infrastructure.benchmark.hybrid_subtitle.yolo_proposal import YoloTextProposalProvider
            self._proposal = YoloTextProposalProvider(
                weights_path=self._weights, conf=self._conf, iou=self._iou,
                imgsz=self._imgsz, device=self._device,
            )
        image = cv2.imread(frames[0].image_path)
        if image is None:
            raise RuntimeError("Unable to read text-detector frame.")
        height, width = image.shape[:2]
        bundle = ProposalBundle("", fps=1.0, stride=1, video_width=width, video_height=height)
        for index, frame in enumerate(frames):
            decoded = cv2.imread(frame.image_path)
            if decoded is None:
                raise RuntimeError(f"Unable to read detector frame: {frame.image_path}")
            bundle.add_frame(
                frame_index=index,
                timestamp_sec=frame.timestamp,
                image=decoded,
                image_path=frame.image_path,
                detections=[],
            )
        detections = self._proposal.detect_images([frame.image_path for frame in frames])
        return bundle, detections, width, height

    def _build_tracks(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        by_frame: Dict[int, List[Dict[str, Any]]] = {}
        for detection in detections:
            if float(detection.get("confidence", 0.0)) >= self._min_confidence:
                by_frame.setdefault(int(detection["frame_index"]), []).append(detection)
        groups = [
            group
            for frame_index in sorted(by_frame)
            for group in group_frame_candidates(by_frame[frame_index], self._h_gap_ratio)
        ]
        return build_tracks(groups, iou_threshold=self._track_iou_threshold, max_gap_frames=self._track_max_gap_frames) if groups else []

    def _cheap_filter(
        self,
        tracks: List[Dict[str, Any]],
        bundle: ProposalBundle,
        height: int,
        band_gate_threshold: float | None = None,
    ) -> List[Dict[str, Any]]:
        threshold = (
            self._band_gate_threshold
            if band_gate_threshold is None
            else band_gate_threshold
        )
        variance = PixelVarianceDiscriminator(self._static_variance_threshold)
        candidates: List[Dict[str, Any]] = []
        self.band_gate_ledger = []
        self.static_variance_ledger = []
        for track in tracks:
            band_pass = passes_bottom_band(track, height, threshold)
            evidence = variance.evaluate(track, bundle)
            passed = band_pass and not evidence.rejected
            self.band_gate_ledger.append({
                "track_id": track["track_id"], "y1_norm": track_y1_norm(track, height),
                "threshold": threshold, "passed": band_pass,
            })
            self.static_variance_ledger.append({
                "track_id": track["track_id"], "temporal_pixel_variance": evidence.temporal_pixel_variance,
                "sample_count": evidence.sample_count, "rejected": evidence.rejected,
                "reason": evidence.reason, "threshold": self._static_variance_threshold,
                "passed": passed,
            })
            if passed:
                candidates.append(track)
        logger.info("V4 cheap filter kept %d/%d tracks.", len(candidates), len(tracks))
        return candidates

    def _record_decisions(
        self, tracks: List[Dict[str, Any]], decisions: Dict[str, Dict[str, Any]]
    ) -> None:
        assert self._decision_store is not None
        for track in tracks:
            track_id = track["track_id"]
            decision = decisions.get(track_id)
            if decision is None:
                logger.warning("LOUD-WARN: no decision record for track_id=%s", track_id)
                continue
            response_ref = decision.get("response_ref")
            model = decision.get("model")
            if not isinstance(response_ref, str) or not response_ref or not isinstance(model, str) or not model:
                logger.warning("LOUD-WARN: incomplete decision evidence for track_id=%s", track_id)
                continue
            self._decision_store.record(
                track_id=track_id,
                is_subtitle=decision["is_subtitle"],
                model=model,
                response_ref=response_ref,
                cache_hit=bool(decision.get("cache_hit", False)),
            )

    def _regions_for_positive_tracks(
        self, tracks: List[Dict[str, Any]], frames: List[Frame], width: int, height: int
    ) -> List[SubtitleRegion]:
        assert self._decision_store is not None
        timestamps = self._timestamps(frames)
        regions: List[SubtitleRegion] = []
        for track in tracks:
            track_id = track["track_id"]
            if not self._decision_store.permits_mask(track_id):
                logger.warning("LOUD-WARN: track_id=%s rejected for mask rendering.", track_id)
                continue
            confidence = _clamp01(track.get("mean_detector_confidence", 0.0))
            # Each frame group preserves its source geometry. A positive
            # track decision authorizes every geometry window of that track.
            for group in track.get("groups", []):
                bbox = group.get("union_bbox_xyxy", [])
                if len(bbox) != 4:
                    continue
                frame_index = int(group["frame_index"])
                start, end = timestamps.get(frame_index, (0.0, 0.0))
                x1, y1, x2, y2 = bbox
                regions.append(SubtitleRegion(
                    start_time=start, end_time=end,
                    x=_clamp01(x1 / width), y=_clamp01(y1 / height),
                    width=_clamp01((x2 - x1) / width),
                    height=_clamp01((y2 - y1) / height),
                    confidence=confidence, track_id=track_id,
                ))

        return regions

    def _timestamps(self, frames: List[Frame]) -> Dict[int, tuple[float, float]]:
        if self.proposal_bundle is not None:
            ordered = [(item.frame_index, item.timestamp_sec) for item in self.proposal_bundle.frames]
        else:
            ordered = list(enumerate(frame.timestamp for frame in frames))
        ordered.sort(key=lambda item: item[1])
        result: Dict[int, tuple[float, float]] = {}
        for position, (frame_index, timestamp) in enumerate(ordered):
            if position + 1 < len(ordered):
                next_timestamp = ordered[position + 1][1]
                duration = max(0.0, next_timestamp - timestamp)
            elif position > 0:
                duration = max(0.0, timestamp - ordered[position - 1][1])
            else:
                duration = 0.0
            result[frame_index] = (max(0.0, timestamp - duration / 2), timestamp + duration / 2)
        return result
