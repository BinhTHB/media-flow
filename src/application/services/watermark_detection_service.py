from __future__ import annotations

from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from statistics import median
from typing import Iterable

from src.application.services.adaptive_subtitle_region_service import BoundingBox


@dataclass(frozen=True)
class WatermarkObservation:
    timestamp: float
    bbox: BoundingBox
    text_content: str = ""
    ocr_confidence: float | None = None
    asr_match_score: float = 0.0
    subtitle_overlap_score: float = 0.0
    scene_text_risk_score: float = 0.0
    overlay_style_score: float = 0.0


class WatermarkDetectionService:
    """Groups temporal visual observations and applies conservative watermark policy."""

    def __init__(
        self,
        minimum_coverage: float = 0.40,
        edge_band_ratio: float = 0.18,
        allow_center_watermark: bool = False,
    ) -> None:
        self.minimum_coverage = minimum_coverage
        self.edge_band_ratio = edge_band_ratio
        self.allow_center_watermark = allow_center_watermark

    @staticmethod
    def _similar(left: WatermarkObservation, right: WatermarkObservation) -> bool:
        center_distance = (
            (left.bbox.x_center - right.bbox.x_center) ** 2
            + (left.bbox.y_center - right.bbox.y_center) ** 2
        ) ** 0.5
        size_ratio = min(left.bbox.area, right.bbox.area) / max(left.bbox.area, right.bbox.area, 1.0)
        text_match = (
            not left.text_content or not right.text_content
            or SequenceMatcher(None, left.text_content.casefold(), right.text_content.casefold()).ratio() >= 0.55
        )
        tolerance = max(left.bbox.width, right.bbox.width, left.bbox.height, right.bbox.height) * 0.45
        return text_match and size_ratio >= 0.55 and (
            left.bbox.iou(right.bbox) >= 0.25 or center_distance <= tolerance
        )

    @staticmethod
    def clamp_bbox(bbox: BoundingBox, frame_width: int, frame_height: int) -> BoundingBox:
        return BoundingBox(
            max(0.0, min(float(frame_width), bbox.left)),
            max(0.0, min(float(frame_height), bbox.top)),
            max(0.0, min(float(frame_width), bbox.right)),
            max(0.0, min(float(frame_height), bbox.bottom)),
        )

    def detect(
        self,
        observations: Iterable[WatermarkObservation],
        sampled_frame_count: int,
        frame_width: int,
        frame_height: int,
    ) -> list[dict]:
        groups: list[list[WatermarkObservation]] = []
        for observation in sorted(observations, key=lambda item: item.timestamp):
            match = next(
                (group for group in groups if self._similar(observation, group[0])),
                None,
            )
            if match is None:
                groups.append([observation])
            else:
                match.append(observation)
        return [
            self._classify(index, group, sampled_frame_count, frame_width, frame_height)
            for index, group in enumerate(groups, start=1)
        ]

    @staticmethod
    def apply_auto_blur_safety(
        candidate: dict,
        frame_width: int,
        frame_height: int,
        final_candidate_count: int,
        resource_profile: str,
    ) -> dict:
        box = candidate["bbox"]
        area_ratio = (
            (box["right"] - box["left"]) * (box["bottom"] - box["top"])
            / max(1, frame_width * frame_height)
        )
        position_stability = candidate.get("position_stability_score", 0.0)
        size_stability = candidate.get("size_stability_score", 0.0)
        edge_score = candidate.get("corner_or_edge_score", 0.0)
        motion_risk = max(0.0, 1.0 - position_stability)
        centrality_risk = max(0.0, 1.0 - edge_score)
        person_risk = max(
            candidate.get("face_person_overlap_risk", 0.0),
            centrality_risk * 0.5 + motion_risk * 2.0,
        )
        object_risk = max(
            candidate.get("scene_attachment_risk", 0.0),
            max(0.0, 1.0 - size_stability) * 0.8 + motion_risk * 1.5,
        )
        independence = position_stability * 0.6 + size_stability * 0.4
        reason = None
        if person_risk > 0.25:
            reason = "person_overlap_risk"
        elif object_risk > 0.30:
            reason = "object_attachment_risk"
        elif area_ratio > 0.025:
            reason = "region_too_large"
        elif edge_score < 0.75:
            reason = "region_too_central"
        elif len(candidate.get("merged_candidate_ids", (candidate["candidate_id"],))) > 1:
            reason = "merged_scene_fragments"
        elif resource_profile == "balanced" and final_candidate_count > 1:
            reason = "multiple_distinct_candidates"
        enriched = {
            **candidate,
            "person_overlap_risk": round(person_risk, 4),
            "face_person_overlap_risk": round(person_risk, 4),
            "object_attachment_risk": round(object_risk, 4),
            "overlay_independence_score": round(independence, 4),
            "screen_position_stability_score": round(position_stability, 4),
            "bbox_area_ratio": round(area_ratio, 6),
            "balanced_auto_blur_allowed": reason is None,
            "rejection_reason": reason,
            "review_reason": reason,
        }
        if reason and enriched.get("classification") != "rejected":
            enriched["classification"] = "uncertain"
            enriched["rejected_reasons"] = [
                *enriched.get("rejected_reasons", []), reason
            ]
        return enriched

    @staticmethod
    def merge_nearby(candidates: Iterable[dict], frame_width: int, frame_height: int) -> list[dict]:
        """Merges final fragments of one overlay before visual review or rendering."""
        items = sorted(candidates, key=lambda item: item["watermark_score"], reverse=True)
        merged: list[dict] = []
        gap = max(12.0, min(frame_width, frame_height) * 0.035)
        for candidate in items:
            box = candidate["bbox"]
            match = next((
                existing for existing in merged
                if (
                    max(existing["bbox"]["left"], box["left"]) <= min(existing["bbox"]["right"], box["right"]) + gap
                    and max(existing["bbox"]["top"], box["top"]) <= min(existing["bbox"]["bottom"], box["bottom"]) + gap
                )
            ), None)
            if match is None:
                merged.append({**candidate, "merged_candidate_ids": [candidate["candidate_id"]]})
                continue
            match["bbox"] = {
                "left": min(match["bbox"]["left"], box["left"]),
                "top": min(match["bbox"]["top"], box["top"]),
                "right": max(match["bbox"]["right"], box["right"]),
                "bottom": max(match["bbox"]["bottom"], box["bottom"]),
            }
            match["frame_coverage_ratio"] = max(match["frame_coverage_ratio"], candidate["frame_coverage_ratio"])
            match["position_stability_score"] = min(match["position_stability_score"], candidate["position_stability_score"])
            match["size_stability_score"] = min(match["size_stability_score"], candidate["size_stability_score"])
            match["watermark_score"] = max(match["watermark_score"], candidate["watermark_score"])
            match["first_seen_timestamp"] = min(match["first_seen_timestamp"], candidate["first_seen_timestamp"])
            match["last_seen_timestamp"] = max(match["last_seen_timestamp"], candidate["last_seen_timestamp"])
            match["merged_candidate_ids"].append(candidate["candidate_id"])
        for index, candidate in enumerate(merged, start=1):
            candidate["candidate_id"] = f"watermark_final_{index:03d}"
        return merged

    def _classify(
        self,
        index: int,
        group: list[WatermarkObservation],
        sampled_frame_count: int,
        frame_width: int,
        frame_height: int,
    ) -> dict:
        unique_frames = len({round(item.timestamp, 3) for item in group})
        coverage = unique_frames / max(1, sampled_frame_count)
        centers_x = [item.bbox.x_center / frame_width for item in group]
        centers_y = [item.bbox.y_center / frame_height for item in group]
        areas = [item.bbox.area / (frame_width * frame_height) for item in group]
        position_stability = max(
            0.0,
            1.0 - (max(centers_x) - min(centers_x) + max(centers_y) - min(centers_y)) * 4,
        )
        size_stability = max(0.0, 1.0 - (max(areas) - min(areas)) / max(median(areas), 0.001))
        box = BoundingBox(
            median(item.bbox.left for item in group),
            median(item.bbox.top for item in group),
            median(item.bbox.right for item in group),
            median(item.bbox.bottom for item in group),
        )
        edge_distance = min(box.left, box.top, frame_width - box.right, frame_height - box.bottom)
        edge_score = max(0.0, 1.0 - edge_distance / max(1.0, min(frame_width, frame_height) * self.edge_band_ratio))
        subtitle_overlap = sum(item.subtitle_overlap_score for item in group) / len(group)
        asr_match = sum(item.asr_match_score for item in group) / len(group)
        scene_risk = max(item.scene_text_risk_score for item in group)
        overlay_style = sum(item.overlay_style_score for item in group) / len(group)
        persistence = min(1.0, coverage / 0.60)
        score = max(0.0, min(1.0,
            0.30 * persistence + 0.25 * position_stability + 0.15 * size_stability
            + 0.15 * edge_score + 0.15 * overlay_style
            - 0.30 * subtitle_overlap - 0.25 * asr_match - 0.35 * scene_risk
        ))
        reasons: list[str] = []
        if coverage < self.minimum_coverage:
            classification = "rejected"
            reasons.append("low_frame_coverage")
        elif subtitle_overlap >= 0.50 or asr_match >= 0.60:
            classification = "rejected"
            reasons.append("subtitle_or_asr_aligned")
        elif scene_risk >= 0.60 or box.area / (frame_width * frame_height) > 0.12:
            classification = "uncertain"
            reasons.append("scene_text_or_large_region_risk")
        elif (
            coverage >= 0.60 and position_stability >= 0.85 and edge_score >= 0.70
            and score >= 0.70
        ):
            classification = "logo" if not any(item.text_content for item in group) else "watermark"
        elif coverage >= 0.60 and position_stability >= 0.85 and score >= 0.62:
            classification = "persistent_overlay" if edge_score >= 0.55 else "uncertain"
        else:
            classification = "uncertain"
            reasons.append("insufficient_overlay_evidence")
        text = max((item.text_content for item in group), key=len, default="")
        confidences = [item.ocr_confidence for item in group if item.ocr_confidence is not None]
        return {
            "candidate_id": f"watermark_{index:03d}",
            "bbox": self.clamp_bbox(box, frame_width, frame_height).to_dict(),
            "frame_occurrences": unique_frames,
            "first_seen_timestamp": min(item.timestamp for item in group),
            "last_seen_timestamp": max(item.timestamp for item in group),
            "frame_coverage_ratio": round(coverage, 4),
            "persistence_score": round(persistence, 4),
            "temporal_persistence_score": round(persistence, 4),
            "position_stability_score": round(position_stability, 4),
            "size_stability_score": round(size_stability, 4),
            "corner_or_edge_score": round(edge_score, 4),
            "text_content": text or None,
            "ocr_confidence": round(sum(confidences) / len(confidences), 4) if confidences else None,
            "asr_match_score": round(asr_match, 4),
            "subtitle_overlap_score": round(subtitle_overlap, 4),
            "subtitle_overlap_risk": round(subtitle_overlap, 4),
            "scene_text_risk_score": round(scene_risk, 4),
            "scene_attachment_risk": round(scene_risk, 4),
            "face_person_overlap_risk": 0.0,
            "watermark_score": round(score, 4),
            "classification": classification,
            "rejected_reasons": reasons,
            "observations": [
                {"timestamp": item.timestamp, "bbox": item.bbox.to_dict(), **{
                    key: value for key, value in asdict(item).items() if key not in {"timestamp", "bbox"}
                }}
                for item in group
            ],
        }
