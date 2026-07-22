from __future__ import annotations
import logging
import os
import re
from typing import Any, Dict, List, Optional

import cv2

from src.application.dto.proposal_bundle import ProposalBundle
from src.application.ports.subtitle_region_detector_provider import (
    SubtitleRegionDetectorProvider,
)
from src.domain.entities.frame import Frame
from src.domain.entities.subtitle_region import SubtitleRegion
from src.infrastructure.benchmark.hybrid_subtitle.core import (
    box_iou,
    group_frame_candidates,
    build_tracks,
    representative_group_ids,
)
from src.infrastructure.benchmark.hybrid_subtitle.runner import (
    classify_track_by_position,
)
from src.infrastructure.benchmark.hybrid_subtitle.subtitle_classifier import (
    HybridSubtitleClassifier,
    HeuristicSubtitleClassifier,
)

logger = logging.getLogger(__name__)


def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, float(v)))


class HybridSubtitleRegionDetectorProvider(SubtitleRegionDetectorProvider):
    """YOLO proposal -> grouping -> tracking -> heuristic vi tri.

    Supports two modes:
    1. **Legacy** (``proposal_bundle`` is None): calls ``detect_images``
       on the interval-extracted frame paths, triggering a separate YOLO pass.
    2. **Single-decode** (``proposal_bundle`` is set): reuses the pre-computed
       detections and frame images from one YOLO-on-video pass.

    proposal_provider cho phep inject fake trong test (khong can best.pt).
    YOLO that chi duoc khoi tao lazy trong detect_regions.
    """

    def __init__(self, weights: str = "best.pt", conf: float = 0.05,
                 iou: float = 0.70, imgsz: int = 1024,
                 horizontal_gap_ratio: float = 2.0,
                 track_iou_threshold: float = 0.25,
                 track_max_gap_frames: int = 5,
                 bottom_ratio_threshold: float = 0.65,
                 min_confidence: float = 0.45,
                 proposal_provider: Optional[Any] = None,
                 device: str = "cpu",
                 semantic_verifier: Optional[Any] = None,
                 classifier: Optional[Any] = None,
                 crop_batch_classifier: Optional[Any] = None) -> None:
        self._weights = weights
        self._conf = conf
        self._iou = iou
        self._imgsz = imgsz
        self._h_gap_ratio = horizontal_gap_ratio
        self._track_iou_threshold = track_iou_threshold
        self._track_max_gap_frames = track_max_gap_frames
        self._bottom_ratio_threshold = bottom_ratio_threshold
        self._min_confidence = min_confidence
        self._proposal = proposal_provider  # lazy-init trong detect_regions
        self._device = device
        self._semantic_verifier = semantic_verifier
        self._classifier = classifier  # set by provider_registry or test
        self._crop_batch_classifier = crop_batch_classifier
        self.proposal_bundle: ProposalBundle | None = None

    def detect_regions(self, frames: List[Frame]) -> List[SubtitleRegion]:
        if not frames:
            return []

        # 1) Get proposals
        if self.proposal_bundle is not None:
            # Mode 2: Single-decode Proposal Bundle
            dets = self.proposal_bundle.detections
            H = self.proposal_bundle.video_height
            W = self.proposal_bundle.video_width
        else:
            # Mode 1: Legacy Frame-extraction path
            # lazy-init YOLO
            if self._proposal is None:
                from src.infrastructure.benchmark.hybrid_subtitle.yolo_proposal import YoloTextProposalProvider
                self._proposal = YoloTextProposalProvider(
                    weights_path=self._weights,
                    conf=self._conf,
                    iou=self._iou,
                    imgsz=self._imgsz,
                    device=self._device,
                )
            image_paths = [f.image_path for f in frames]
            dets = self._proposal.detect_images(image_paths)
            img = cv2.imread(frames[0].image_path)
            H, W = img.shape[:2]

        # Discard proposals below the minimum confidence threshold before grouping.
        # Otherwise giant low-confidence noise boxes (e.g. 0.053) sort first and swallow
        # valid subtitle word boxes during baseline grouping.
        dets = [d for d in dets if float(d.get("confidence", 0.0)) >= self._min_confidence]

        # 2) group per frame
        by_frame: Dict[int, List[Dict]] = {}
        for d in dets:
            by_frame.setdefault(d["frame_index"], []).append(d)

        flat_groups: List[Dict] = []
        for fi in sorted(by_frame):
            groups = group_frame_candidates(by_frame[fi], self._h_gap_ratio)
            flat_groups.extend(groups)

        if not flat_groups:
            return []

        # 3) build tracks
        tracks = build_tracks(
            flat_groups,
            iou_threshold=self._track_iou_threshold,
            max_gap_frames=self._track_max_gap_frames,
        )
        if not tracks:
            return []

        # Map decoded frame indexes to timestamp-bearing frames. A ProposalBundle is
        # authoritative for sparse strides; legacy extracted frames stay sequential.
        if self.proposal_bundle is not None:
            frames_by_index = {
                decoded.frame_index: Frame(
                    timestamp=decoded.timestamp_sec,
                    image_path=decoded.image_path or "",
                )
                for decoded in self.proposal_bundle.frames
            }
        else:
            frames_by_index: Dict[int, Frame] = {}
            for frame in frames:
                match = re.search(
                    r"frame_(\d+)\.(jpg|png|jpeg)",
                    os.path.basename(frame.image_path or ""),
                )
                if match:
                    frames_by_index[int(match.group(1))] = frame
            if not frames_by_index:
                frames_by_index = {index: frame for index, frame in enumerate(frames)}

        crop_decisions = self._crop_decisions_for_uncertain_tracks(tracks, H)
        regions: List[SubtitleRegion] = []
        for track in tracks:
            decision = self._verifier_decision_for_track(
                track, frames_by_index, H, crop_decisions.get(track["track_id"])
            )
            if decision is None:
                continue

            conf = _clamp01(track.get("mean_detector_confidence", 0.0))
            if conf < self._min_confidence:
                continue

            # Frame cadence (frames are sampled at the middle of each interval).
            if len(frames) >= 2:
                frame_gap = (frames[-1].timestamp - frames[0].timestamp) / (len(frames) - 1)
            else:
                frame_gap = 0.0

            # Detect moving track: if first and last frame bboxes diverge, emit
            # one region per frame instead of a single union that spans all frames.
            first_bbox = track["groups"][0]["union_bbox_xyxy"]
            last_bbox = track["groups"][-1]["union_bbox_xyxy"]
            moving = len(track["groups"]) >= 2 and box_iou(first_bbox, last_bbox) < 0.30

            # Gather proposal ids
            proposal_ids = sorted(
                {
                    candidate_id
                    for group in track["groups"]
                    for candidate_id in group.get("member_candidate_ids", [])
                }
            )

            if moving:
                for g in track["groups"]:
                    bbox = g["union_bbox_xyxy"]
                    if not bbox:
                        continue
                    x1, y1, x2, y2 = bbox
                    fi = g["frame_index"]
                    src_frame = frames_by_index.get(fi)
                    if src_frame is None:
                        continue
                    ts = src_frame.timestamp
                    start_time = max(0.0, ts - frame_gap / 2.0)
                    end_time = ts + frame_gap / 2.0
                    try:
                        region = SubtitleRegion(
                            start_time=start_time,
                            end_time=end_time,
                            x=_clamp01(x1 / W),
                            y=_clamp01(y1 / H),
                            width=_clamp01((x2 - x1) / W),
                            height=_clamp01((y2 - y1) / H),
                            confidence=conf,
                            classification_id=f"classification:{track['track_id']}",
                            is_subtitle=True,
                            classifier_model=decision.get("model", "unknown"),
                            track_id=track["track_id"],
                            proposal_ids=proposal_ids,
                            category=decision.get("category", "dialogue_subtitle"),
                            classifier_confidence=decision.get("confidence", 1.0),
                        )
                    except Exception:
                        continue
                    regions.append(region)
                continue

            # Stable track: emit a single region spanning the full track.
            union = (
                track.get("temporal_union_bbox_xyxy")
                or track.get("median_bbox_xyxy")
                or []
            )
            if not union:
                continue
            x1, y1, x2, y2 = union

            first_idx = track["groups"][0]["frame_index"]
            last_idx = track["groups"][-1]["frame_index"]
            first_frame = frames_by_index.get(first_idx)
            last_frame = frames_by_index.get(last_idx)
            if first_frame is None or last_frame is None:
                continue

            start_time = first_frame.timestamp
            end_time = last_frame.timestamp
            if end_time < start_time:
                end_time = start_time
            start_time = max(0.0, start_time - frame_gap / 2.0)
            end_time = end_time + frame_gap / 2.0

            try:
                region = SubtitleRegion(
                    start_time=start_time,
                    end_time=end_time,
                    x=_clamp01(x1 / W),
                    y=_clamp01(y1 / H),
                    width=_clamp01((x2 - x1) / W),
                    height=_clamp01((y2 - y1) / H),
                    confidence=conf,
                    classification_id=f"classification:{track['track_id']}",
                    is_subtitle=True,
                    classifier_model=decision.get("model", "unknown"),
                    track_id=track["track_id"],
                    proposal_ids=proposal_ids,
                    category=decision.get("category", "dialogue_subtitle"),
                    classifier_confidence=decision.get("confidence", 1.0),
                )
            except Exception:
                continue
            regions.append(region)

        return regions

    def _crop_decisions_for_uncertain_tracks(
        self, tracks: List[Dict[str, Any]], frame_height: int
    ) -> Dict[str, Dict[str, Any]]:
        if self.proposal_bundle is None:
            return {}
        heuristic = HeuristicSubtitleClassifier()
        uncertain = []
        canaries = []
        for track in tracks:
            result = heuristic.classify(track, frame_height)
            if result.is_subtitle:
                continue
            duration_ms = (
                track["groups"][-1]["timestamp_ms"]
                - track["groups"][0]["timestamp_ms"]
            )
            if len(track["groups"]) < 2 or duration_ms < 300:
                canaries.append(track)
            else:
                uncertain.append(track)
        if not uncertain:
            return {}
        if not canaries:
            logger.error("LOUD-WARN: no local negative canary, uncertain tracks fail-closed")
            return {}
        if self._crop_batch_classifier is None:
            logger.error("LOUD-WARN: crop classifier unavailable, uncertain tracks fail-closed")
            return {}
        return self._crop_batch_classifier.classify_batch(
            uncertain, self.proposal_bundle, batch_size=10, canary_tracks=canaries[:2]
        )

    def _verifier_decision_for_track(
        self,
        track: Dict,
        frames_by_index: Dict[int, Frame],
        H: int,
        crop_decision: Dict[str, Any] | None = None,
    ) -> Dict[str, Any] | None:
        """Evaluate the track using the hybrid classifier (heuristic + optional Gemini).

        Returns a decision dict if confirmed as subtitle, or None to fail-closed (discard).
        Decision sources are tracked via provenance:
          - heuristic_bottom_stable: confirmed by position+duration heuristic
          - gemini_verified: confirmed by Gemini after heuristic borderline
          - heuristic_top_or_short: heuristic non-subtitle (no Gemini fallback)
        """
        if crop_decision is not None:
            if crop_decision.get("is_subtitle") is True:
                return {
                    "model": "crop_classifier",
                    "category": crop_decision.get("category", "dialogue_subtitle"),
                    "confidence": crop_decision.get("confidence", 0.0),
                    "provenance": "crop_classifier",
                }
            return None

        # If no classifier configured, fail-closed
        if self._classifier is None and self._semantic_verifier is None:
            return None

        # Setup classifier if not already done
        if self._classifier is None and self._semantic_verifier is not None:
            from src.infrastructure.benchmark.hybrid_subtitle.subtitle_classifier import HybridSubtitleClassifier
            self._classifier = HybridSubtitleClassifier(
                gemini_verifier=self._semantic_verifier,
                frame_height=H
            )

        groups = track.get("groups", [])
        if not groups:
            return None

        # Gather representative groups and frames for classification
        if len(groups) <= 3:
            rep_groups = groups
        else:
            rep_groups = [groups[0], groups[len(groups) // 2], groups[-1]]

        candidate_ids = [g["candidate_group_id"] for g in rep_groups]
        candidate_bboxes = [g["union_bbox_xyxy"] for g in rep_groups]
        frame_bgr = []
        for g in rep_groups:
            fi = g["frame_index"]
            if self.proposal_bundle is not None:
                try:
                    decoded_frame = self.proposal_bundle.frame_for_index(fi)
                    frame_bgr.append(decoded_frame.image)
                except KeyError as exc:
                    raise RuntimeError(
                        f"Single-decode bundle is missing frame_index={fi}."
                    ) from exc
            else:
                src_frame = frames_by_index.get(fi)
                if src_frame is not None:
                    frame = cv2.imread(src_frame.image_path)
                    if frame is not None:
                        frame_bgr.append(frame)

        if not frame_bgr:
            logger.warning("LOUD-WARN: no frames for track %s, fail-closed", track["track_id"])
            return None

        # Use hybrid classifier
        result = self._classifier.classify_track(
            track, frame_bgr, candidate_ids, candidate_bboxes
        )

        decisions = result.get("decisions", [])
        if not decisions:
            logger.warning(
                "LOUD-WARN: region verifier did not confirm any valid candidate for track %s. "
                "Fail-closed used: track discarded.",
                track["track_id"]
            )
            return None

        # Return the first confirmed subtitle decision
        for d in decisions:
            if d.get("is_subtitle") is True:
                return {
                    "model": result.get("model", "unknown"),
                    "category": d.get("category", "dialogue_subtitle"),
                    "confidence": d.get("confidence", 1.0),
                    "provenance": d.get("provenance", "unknown"),
                }
        return None