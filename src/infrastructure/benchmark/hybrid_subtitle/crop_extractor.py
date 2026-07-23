"""Track crop extractor for STEP B evidence pack.

Extracts representative crops and trajectory metadata from ProposalBundle.
Single-decode: NO re-decode or video seek (frame_for_index only).
"""
from __future__ import annotations
from typing import Any, Dict, List

import numpy as np

class TrackCropExtractor:
    """Extract up to max_crops padded crops + clean frames + trajectory for a track.

    The proposal_bundle is the authoritative source of decoded frames (single-decode
    pass).  This extractor never re-decodes or seeks the video.
    """

    def extract_evidence(
        self,
        track: Dict[str, Any],
        proposal_bundle: Any,
        *,
        max_crops: int = 5,
        padding: int = 8,
    ) -> Dict[str, Any]:
        """Return evidence pack dict or empty dict if no valid groups.

        Uses pre-cached 720p frames and full-res crops from ProposalBundle
        when available (STEP-0 single-pass). Falls back to on-demand extraction
        for legacy code paths.
        """
        from src.application.dto.proposal_bundle import ProposalBundle
        groups = track.get("groups", [])
        if not groups:
            return {}

        # Select up to max_crops representative frame indices
        n = len(groups)
        if n <= max_crops:
            rep_indices = list(range(n))
        else:
            indices = {0, n // 2, n - 1}
            step = n / (max_crops - 1)
            for i in range(1, max_crops - 1):
                indices.add(int(round(i * step)))
            rep_indices = sorted(indices)[:max_crops]

        crops: List[Any] = []
        clean_frames: List[Any] = []
        # Track whether we use cached crops so we can skip per-candidate crop IDs
        use_cached_crops = True

        for idx in rep_indices:
            group = groups[idx]
            fi = group["frame_index"]
            bbox = group["union_bbox_xyxy"]  # [x1, y1, x2, y2]
            try:
                decoded = proposal_bundle.frame_for_index(fi)
            except (KeyError, AttributeError):
                continue

            # Prefer cached 720p context frame (STEP 0), fallback to full-res
            if decoded.image_720p is not None:
                clean_frames.append(decoded.image_720p.copy())
            else:
                clean_frames.append(decoded.image.copy())

            # Prefer cached full-res crops (STEP 0), fallback to on-demand slice
            if decoded.crops:
                # Find which detection index corresponds to this group's candidates
                candidate_ids = set(group.get("member_candidate_ids", []))
                found = False
                for det_idx, det in enumerate(decoded.detections):
                    det_cid = det.get("candidate_id", "")
                    if det_cid in candidate_ids and det_idx < len(decoded.crops):
                        crops.append(decoded.crops[det_idx].copy())
                        found = True
                        break
                if not found:
                    use_cached_crops = False
                    crops.append(self._extract_crop(decoded.image, bbox, padding))
            else:
                use_cached_crops = False
                crops.append(self._extract_crop(decoded.image, bbox, padding))

        # If not all crops resolved from cache, fall back entirely
        if not use_cached_crops and len(crops) == len(rep_indices):
            pass  # All resolved via mixed strategy

        if not crops or not clean_frames:
            return {}

        # Trajectory metrics from temporal union bbox
        union_bbox = (
            track.get("temporal_union_bbox_xyxy")
            or track.get("median_bbox_xyxy")
            or [0, 0, 0, 0]
        )
        _, _, _, y2_union = union_bbox
        H_meta = getattr(proposal_bundle, "video_height", 1280) or 1280
        bottom_y_ratio = y2_union / H_meta if H_meta > 0 else 0.0

        first_group = groups[0]
        last_group = groups[-1]
        first_ts_ms = first_group.get("timestamp_ms", 0.0)
        last_ts_ms = last_group.get("timestamp_ms", 0.0)
        duration = (last_ts_ms - first_ts_ms) / 1000.0

        first_bbox = first_group["union_bbox_xyxy"]
        last_bbox = last_group["union_bbox_xyxy"]
        fc_x = (first_bbox[0] + first_bbox[2]) / 2.0
        fc_y = (first_bbox[1] + first_bbox[3]) / 2.0
        lc_x = (last_bbox[0] + last_bbox[2]) / 2.0
        lc_y = (last_bbox[1] + last_bbox[3]) / 2.0
        displacement = ((fc_x - lc_x) ** 2 + (fc_y - lc_y) ** 2) ** 0.5

        return {
            "crops": crops,
            "clean_frames": clean_frames,
            "trajectory": {
                "first_bbox": first_bbox,
                "last_bbox": last_bbox,
                "displacement": displacement,
                "bottom_y_ratio": bottom_y_ratio,
                "duration": duration,
                "first_center": (fc_x, fc_y),
                "last_center": (lc_x, lc_y),
            },
        }

    @staticmethod
    def _extract_crop(
        img: Any, bbox: list[float], padding: int
    ) -> Any:
        """On-demand crop extraction from a decoded frame."""
        import cv2
        H, W = img.shape[:2]
        x1, y1, x2, y2 = bbox
        px1 = max(0, int(round(x1 - padding)))
        py1 = max(0, int(round(y1 - padding)))
        px2 = min(W, int(round(x2 + padding)))
        py2 = min(H, int(round(y2 + padding)))
        crop = img[py1:py2, px1:px2]
        return crop.copy()

    @staticmethod
    def add_behavior_metadata(
        evidence: Dict[str, Any], behavior: Dict[str, Any]
    ) -> Dict[str, Any]:
        evidence["trajectory"].update(
            {
                "speech_overlap_ratio": float(
                    behavior.get("speech_overlap_ratio", 0.0)
                ),
                "coverage_ratio": float(behavior.get("coverage_ratio", 0.0)),
            }
        )
        return evidence
