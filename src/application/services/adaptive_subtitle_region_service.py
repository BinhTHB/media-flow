from __future__ import annotations

from dataclasses import dataclass, field
from statistics import median
from typing import Iterable, List, Optional
import unicodedata


@dataclass(frozen=True)
class BoundingBox:
    left: float
    top: float
    right: float
    bottom: float

    @property
    def width(self) -> float:
        return self.right - self.left

    @property
    def height(self) -> float:
        return self.bottom - self.top

    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)

    @property
    def x_center(self) -> float:
        return (self.left + self.right) / 2

    @property
    def y_center(self) -> float:
        return (self.top + self.bottom) / 2

    def to_dict(self) -> dict:
        return {
            "left": self.left, "top": self.top,
            "right": self.right, "bottom": self.bottom,
        }

    @classmethod
    def union(cls, boxes: Iterable["BoundingBox"]) -> "BoundingBox":
        boxes = list(boxes)
        return cls(
            min(box.left for box in boxes), min(box.top for box in boxes),
            max(box.right for box in boxes), max(box.bottom for box in boxes),
        )

    def iou(self, other: "BoundingBox") -> float:
        overlap = max(0.0, min(self.right, other.right) - max(self.left, other.left))
        overlap *= max(0.0, min(self.bottom, other.bottom) - max(self.top, other.top))
        total = self.area + other.area - overlap
        return overlap / total if total else 0.0


@dataclass
class TextBox:
    bbox: BoundingBox
    confidence: float
    text: str = ""
    polygon: List[List[float]] = field(default_factory=list)
    source: str = "ocr"

    def to_dict(self) -> dict:
        return {
            "text": self.text, "confidence": self.confidence,
            "polygon": self.polygon, "bbox": self.bbox.to_dict(),
            "width": self.bbox.width, "height": self.bbox.height,
            "x_center": self.bbox.x_center, "y_center": self.bbox.y_center,
            "area": self.bbox.area, "source": self.source,
        }


@dataclass
class SubtitleLineGroup:
    boxes: List[TextBox]

    @property
    def bbox(self) -> BoundingBox:
        return BoundingBox.union(box.bbox for box in self.boxes)

    def to_dict(self) -> dict:
        bbox = self.bbox
        return {
            "line_bbox": bbox.to_dict(),
            "line_text": " ".join(box.text for box in self.boxes if box.text),
            "line_confidence": sum(box.confidence for box in self.boxes) / len(self.boxes),
            "token_count": len(self.boxes),
            "width_ratio": bbox.width,
            "height_ratio": bbox.height,
            "density_score": min(1.0, sum(box.bbox.area for box in self.boxes) / max(bbox.area, 1)),
            "boxes": [box.to_dict() for box in self.boxes],
        }


@dataclass
class SubtitleBlock:
    lines: List[SubtitleLineGroup]

    @property
    def bbox(self) -> BoundingBox:
        return BoundingBox.union(line.bbox for line in self.lines)

    def to_dict(self) -> dict:
        return {"subtitle_block_bbox": self.bbox.to_dict(), "lines": [line.to_dict() for line in self.lines]}


class AdaptiveSubtitleRegionService:
    """Pure geometry and policy for adaptive subtitle localization."""

    @staticmethod
    def classify_candidate(text: str, token_count: int = 1) -> dict:
        normalized = unicodedata.normalize("NFKC", text).strip()
        alphanumeric = sum(character.isalnum() for character in normalized)
        punctuation = sum(
            not character.isalnum() and not character.isspace()
            for character in normalized
        )
        visible = alphanumeric + punctuation
        letter_ratio = alphanumeric / visible if visible else 0.0
        punctuation_ratio = punctuation / visible if visible else 1.0
        if not alphanumeric:
            classification, reason = "punctuation_only", "punctuation_only"
        elif letter_ratio < 0.5:
            classification, reason = "low_letter_ratio", "low_letter_ratio"
        elif len(normalized) < 2 and token_count <= 1:
            classification, reason = "decorative_symbol", "too_short_isolated_text"
        else:
            classification, reason = "meaningful_text", None
        return {
            "normalized_text": normalized,
            "text_classification": classification,
            "letter_or_digit_ratio": round(letter_ratio, 4),
            "punctuation_ratio": round(punctuation_ratio, 4),
            "ignored_reason": reason,
        }

    @staticmethod
    def candidate_features(
        text: str, bbox: BoundingBox, frame_width: float, frame_height: float,
        persistence_score: float = 0.0,
    ) -> dict:
        text_score = AdaptiveSubtitleRegionService.classify_candidate(text)["letter_or_digit_ratio"]
        central = (
            bbox.left < frame_width * 0.8 and bbox.right > frame_width * 0.2
            and bbox.top < frame_height * 0.75 and bbox.bottom > frame_height * 0.2
        )
        area_ratio = bbox.area / (frame_width * frame_height)
        scene_risk = min(1.0, persistence_score * 0.7 + (0.3 if central else 0.0))
        likeness = max(0.0, text_score * (1.0 - scene_risk))
        return {
            "text_content_score": round(text_score, 4),
            "subtitle_likeness_score": round(likeness, 4),
            "scene_text_risk_score": round(scene_risk, 4),
            "track_consistency_score": 0.0,
            "mask_area_ratio": round(area_ratio, 6),
            "central_content_overlap_risk": central and area_ratio > 0.20,
            "persistence_score": round(persistence_score, 4),
        }

    @staticmethod
    def video_eligibility(metrics: dict) -> dict:
        total = max(1, metrics.get("total_intervals", 0))
        reliable = metrics.get("reliable_subtitle_intervals", 0)
        skipped = metrics.get("no_subtitle_skip_intervals", 0)
        skip_rate = skipped / total
        presence = reliable / total
        jumps = metrics.get("track_jump_count", 0)
        max_area = metrics.get("max_mask_area_ratio", 0.0)
        scene_rejections = metrics.get("scene_text_rejection_count", 0)
        scene_risk = min(1.0, scene_rejections / total)
        stability = max(0.0, 1.0 - jumps / total)
        score = max(
            0.0,
            0.45 * presence + 0.30 * stability
            + 0.15 * (1.0 - scene_risk) + 0.10 * (1.0 - min(1.0, max_area / 0.20)),
        )
        triggers = []
        if max_area > 0.20:
            triggers.append("max_mask_area_above_20_percent")
        if max_area > 0.35:
            triggers.append("max_mask_area_above_35_percent")
        if jumps > 10:
            triggers.append("track_jumps_above_10")
        if skip_rate > 0.40:
            triggers.append("skip_rate_above_40_percent")
        if presence < 0.30:
            triggers.append("insufficient_reliable_subtitle_intervals")
        if skip_rate > 0.40 or presence < 0.30 or max_area > 0.35:
            decision, action = "SKIP_REPLACEMENT", "skip_hardsub_replacement"
        elif triggers or scene_risk > 0.25:
            decision, action = "REVIEW_REQUIRED", "manual_review"
        else:
            decision, action = "AUTO_REPLACE", "auto_replace"
        return {
            "eligibility_decision": decision,
            "eligibility_score": round(score, 4),
            "reasons": triggers,
            "thresholds_triggered": triggers,
            "interval_counts": {
                "total_intervals": total,
                "reliable_subtitle_intervals": reliable,
                "no_subtitle_skip_intervals": skipped,
            },
            "risk_metrics": {
                "skip_rate": round(skip_rate, 4),
                "subtitle_track_stability_score": round(stability, 4),
                "source_subtitle_presence_score": round(presence, 4),
                "scene_text_risk_score": round(scene_risk, 4),
                "auto_replace_eligibility_score": round(score, 4),
                "max_mask_area_ratio": max_area,
                "track_jump_count": jumps,
            },
            "recommended_action": action,
        }

    @staticmethod
    def interval_eligibility(interval: dict) -> dict:
        strategy = interval.get("strategy")
        area = interval.get("mask_area_ratio", 0.0)
        candidates = interval.get("candidates", [])
        selected = [item for item in candidates if item.get("selected")]
        rejected_scene = sum(
            item.get("ignored_reason") in {"scene_text", "signage_text", "watermark"}
            for item in candidates
        )
        track = interval.get("moving_track_score", interval.get("reliability_score", 0.0))
        selected_text = selected[0].get("raw_ocr_text", "") if selected else ""
        asr = AdaptiveSubtitleRegionService.asr_ocr_consistency(
            selected_text, interval.get("asr_text"), interval.get("asr_confidence", 0.0)
        )
        floating = AdaptiveSubtitleRegionService.floating_caption_features(
            selected[0] if selected else {},
            asr.get("asr_ocr_consistency_score"),
            interval.get("candidate_duration_score", 0.0),
        )
        layout_mode = interval.get("layout_mode", "stable_track_subtitle")
        present = interval.get("source_subtitle_present", bool(selected))
        likeness = max(
            (item.get("subtitle_likeness_score", item.get("letter_or_digit_ratio", 0.0)) for item in selected),
            default=0.0,
        )
        scene_risk = rejected_scene / max(1, len(candidates))
        confidence = 0.40 * float(bool(present)) + 0.30 * likeness + 0.30 * track
        if asr["asr_available"]:
            confidence += asr["asr_weight_used"] * (
                asr["asr_ocr_consistency_score"] - 0.5
            )
        confidence = max(0.0, min(1.0, confidence - scene_risk * 0.4))
        if strategy == "no_subtitle_skip" or not present:
            decision, reason = "SKIP_REPLACEMENT", "no_reliable_source_subtitle"
        elif rejected_scene and not selected:
            decision, reason = "SKIP_REPLACEMENT", "scene_text_only"
        elif floating["rejected_as_scene_text"]:
            decision, reason = "SKIP_REPLACEMENT", floating["rejection_reason"]
        elif area > 0.20:
            decision, reason = "REVIEW_REQUIRED", "area_or_track_uncertainty"
        elif (
            layout_mode == "floating_caption_subtitle"
            and selected
            and floating["final_floating_caption_score"] >= 0.72
            and interval.get("coordinate_mapping", "PASS") == "PASS"
            and not interval.get("target_text_clipped", False)
        ):
            decision, reason = "AUTO_REPLACE", "floating_caption_candidate"
        elif interval.get("rejected_as_jump", False) and layout_mode != "floating_caption_subtitle":
            decision, reason = "REVIEW_REQUIRED", "area_or_track_uncertainty"
        elif selected and track >= 0.70 and confidence >= 0.78:
            decision, reason = "AUTO_REPLACE", "reliable_subtitle_candidate"
        else:
            decision, reason = "REVIEW_REQUIRED", "insufficient_track_evidence"
        return {
            "decision": decision, "reason": reason,
            "source_subtitle_present_score": float(bool(present)),
            "subtitle_likeness_score": likeness,
            "scene_text_risk_score": scene_risk,
            "track_stability_score": track,
            "mask_area_ratio": area,
            "candidate_count": len(candidates),
            "rejected_scene_text_count": rejected_scene,
            "coordinate_mapping_status": interval.get("coordinate_mapping", "PASS"),
            "target_text_clipping_risk": bool(interval.get("target_text_clipped", False)),
            "target_text_fit_score": 0.0 if interval.get("target_text_clipped") else 1.0,
            "final_interval_confidence": round(confidence, 4),
            "skip_reason": reason if decision == "SKIP_REPLACEMENT" else None,
            "review_reason": reason if decision == "REVIEW_REQUIRED" else None,
            "layout_mode": layout_mode,
            **floating,
            **asr,
        }

    @staticmethod
    def floating_caption_features(
        candidate: dict,
        asr_score: float | None,
        duration_score: float = 0.0,
    ) -> dict:
        persistence = float(candidate.get("persistence_score", 0.0))
        perspective = float(candidate.get("perspective_attached_text_risk", 0.0))
        watermark = float(candidate.get("watermark_risk_score", 0.0))
        menu = float(candidate.get("menu_signage_risk_score", 0.0))
        overlay = float(candidate.get("overlay_style_score", candidate.get("style_score") or 0.0))
        text_box = float(candidate.get("text_box_style_score", overlay))
        background = float(candidate.get("caption_box_background_score", 0.0))
        alignment = float(asr_score or 0.0)
        scene = min(1.0, max(
            float(candidate.get("scene_text_risk_score", 0.0)),
            persistence * 0.70 + perspective * 0.20 + menu * 0.10,
        ))
        risk = max(scene, watermark, menu, perspective)
        score = max(0.0, min(1.0,
            0.25 * overlay + 0.15 * text_box + 0.10 * background
            + 0.25 * alignment + 0.15 * duration_score + 0.10 * (1.0 - risk)
        ))
        reason = (
            "watermark_or_logo" if watermark >= 0.7
            else "wall_menu_signage" if menu >= 0.7
            else "perspective_surface_text" if perspective >= 0.7
            else "persistent_scene_text" if persistence >= 0.8 and alignment < 0.2
            else None
        )
        return {
            "floating_caption_score": round(score, 4),
            "final_floating_caption_score": round(score, 4),
            "overlay_style_score": round(overlay, 4),
            "text_box_style_score": round(text_box, 4),
            "caption_box_background_score": round(background, 4),
            "candidate_duration_score": round(duration_score, 4),
            "temporal_speech_alignment_score": round(alignment, 4),
            "asr_alignment_score": round(alignment, 4),
            "persistence_score": round(persistence, 4),
            "scene_text_risk_score": round(scene, 4),
            "watermark_risk_score": round(watermark, 4),
            "menu_signage_risk_score": round(menu, 4),
            "perspective_attached_text_risk": round(perspective, 4),
            "selected_as_floating_caption": score >= 0.72 and reason is None,
            "rejected_as_scene_text": reason is not None,
            "rejection_reason": reason,
        }

    @staticmethod
    def classify_layout_mode(intervals: Iterable[dict]) -> dict:
        items = list(intervals)
        selected = [item for item in items if item.get("selected_source_bbox")]
        positions = [
            (box["top"] + box["bottom"]) / 2 / max(1, item.get("frame_height", 1))
            for item in selected for box in [item["selected_source_bbox"]]
        ]
        bands = ["top" if value < 0.35 else "middle" if value <= 0.65 else "bottom" for value in positions]
        transitions = sum(left != right for left, right in zip(bands, bands[1:]))
        distinct = len(set(bands))
        floating_candidates = sum(item.get("floating_caption_score", 0.0) >= 0.5 for item in items)
        scene_heavy = sum(item.get("scene_text_risk_score", 0.0) >= 0.5 for item in items) > len(items) * 0.4
        if selected and distinct >= 2 and transitions:
            mode = "floating_caption_subtitle"
        elif selected and any(item.get("track_mode") in {"moving_track", "slowly_moving_track"} for item in items):
            mode = "moving_track_subtitle"
        elif selected:
            mode = "stable_track_subtitle"
        elif scene_heavy:
            mode = "scene_text_heavy"
        else:
            mode = "no_reliable_subtitle"
        return {
            "layout_mode": mode,
            "position_bands_seen": sorted(set(bands)),
            "position_transition_count": transitions,
            "floating_caption_candidate_count": floating_candidates,
            "scene_text_heavy": scene_heavy,
        }

    @staticmethod
    def asr_ocr_consistency(ocr_text: str, asr_text: str | None, asr_confidence: float = 0.0) -> dict:
        if not asr_text or asr_confidence < 0.50:
            return {"asr_available": False, "asr_ocr_consistency_score": None, "asr_weight_used": 0.0}
        normalize = lambda value: "".join(
            character for character in unicodedata.normalize("NFKD", value).lower()
            if character.isalnum()
        )
        left, right = normalize(ocr_text), normalize(asr_text)
        overlap = len(set(left) & set(right)) / max(1, len(set(left) | set(right)))
        return {
            "asr_available": True,
            "asr_ocr_consistency_score": round(overlap, 4),
            "asr_weight_used": round(min(0.20, asr_confidence * 0.20), 4),
        }

    def filter_boxes(
        self, boxes: Iterable[TextBox], frame_width: float, frame_height: float
    ) -> tuple[List[TextBox], List[dict]]:
        selected, ignored = [], []
        for box in boxes:
            reason = None
            width_ratio, height_ratio = box.bbox.width / frame_width, box.bbox.height / frame_height
            if box.confidence < 0.70:
                reason = "low_confidence"
            elif width_ratio < 0.015 or height_ratio < 0.008:
                reason = "tiny_isolated_box"
            elif box.bbox.height > box.bbox.width * 1.5:
                reason = "vertical_box"
            elif box.bbox.left <= 1 or box.bbox.top <= 1 or box.bbox.right >= frame_width - 1:
                reason = "border_artifact"
            if reason:
                item = box.to_dict()
                item["ignored_reason"] = reason
                ignored.append(item)
            else:
                selected.append(box)
        return selected, ignored

    def group_lines(self, boxes: Iterable[TextBox]) -> List[SubtitleLineGroup]:
        boxes = sorted(boxes, key=lambda box: box.bbox.y_center)
        if not boxes:
            return []
        threshold = max(5.0, 0.35 * median(box.bbox.height for box in boxes))
        groups: List[List[TextBox]] = []
        for box in boxes:
            if not groups or abs(box.bbox.y_center - median(item.bbox.y_center for item in groups[-1])) > threshold:
                groups.append([box])
            else:
                groups[-1].append(box)
        return [SubtitleLineGroup(sorted(group, key=lambda box: box.bbox.x_center)) for group in groups]

    def group_blocks(self, lines: Iterable[SubtitleLineGroup]) -> List[SubtitleBlock]:
        lines = sorted(lines, key=lambda line: line.bbox.top)
        blocks: List[List[SubtitleLineGroup]] = []
        for line in lines:
            if not blocks:
                blocks.append([line])
                continue
            previous = blocks[-1][-1].bbox
            plausible_gap = max(previous.height, line.bbox.height) * 1.5
            if line.bbox.top - previous.bottom <= plausible_gap:
                blocks[-1].append(line)
            else:
                blocks.append([line])
        return [SubtitleBlock(block) for block in blocks]

    def aggregate(self, boxes: Iterable[BoundingBox]) -> tuple[Optional[BoundingBox], float]:
        boxes = list(boxes)
        if not boxes:
            return None, 0.0
        center_x, center_y = median(box.x_center for box in boxes), median(box.y_center for box in boxes)
        stable = [box for box in boxes if abs(box.x_center - center_x) <= max(box.width, 20) and abs(box.y_center - center_y) <= max(box.height, 20)]
        stable = stable or boxes
        result = BoundingBox(
            median(box.left for box in stable), median(box.top for box in stable),
            median(box.right for box in stable), median(box.bottom for box in stable),
        )
        return result, len(stable) / len(boxes)

    @staticmethod
    def classify_track(samples: Iterable[tuple[float, BoundingBox]], frame_height: float) -> dict:
        points = sorted(samples, key=lambda item: item[0])
        if len(points) < 2 or points[-1][0] == points[0][0]:
            return {
                "track_mode": "unreliable_track", "track_velocity": 0.0,
                "moving_track_score": 0.0, "y_center_trajectory": [],
            }
        times = [point[0] for point in points]
        centers = [point[1].y_center for point in points]
        mean_time, mean_center = sum(times) / len(times), sum(centers) / len(centers)
        denominator = sum((value - mean_time) ** 2 for value in times)
        velocity = (
            sum((time - mean_time) * (center - mean_center) for time, center in zip(times, centers))
            / denominator
        ) if denominator else 0.0
        intercept = mean_center - velocity * mean_time
        residuals = [abs(center - (velocity * time + intercept)) for time, center in zip(times, centers)]
        tolerance = max(median(box.height for _, box in points), frame_height * 0.025)
        score = max(0.0, 1.0 - sum(residuals) / len(residuals) / tolerance)
        speed = abs(velocity)
        if score < 0.55:
            mode = "unreliable_track"
        elif speed <= frame_height * 0.01:
            mode = "stationary_track"
        elif speed <= frame_height * 0.08:
            mode = "slowly_moving_track"
        else:
            mode = "moving_track"
        return {
            "track_mode": mode,
            "track_velocity": round(velocity, 4),
            "moving_track_score": round(score, 4),
            "trajectory_intercept": round(intercept, 4),
            "trajectory_tolerance": round(tolerance, 4),
            "y_center_trajectory": [
                {"timestamp": time, "y_center": center, "expected_y_center": round(velocity * time + intercept, 4)}
                for time, center in zip(times, centers)
            ],
        }

    @staticmethod
    def interpolate_track_bbox(
        timestamp: float,
        samples: Iterable[tuple[float, BoundingBox]],
        track: dict,
    ) -> Optional[BoundingBox]:
        points = list(samples)
        if not points or track["track_mode"] == "unreliable_track":
            return None
        width = median(box.width for _, box in points)
        height = median(box.height for _, box in points)
        x_center = median(box.x_center for _, box in points)
        y_center = track["track_velocity"] * timestamp + track["trajectory_intercept"]
        return BoundingBox(
            x_center - width / 2, y_center - height / 2,
            x_center + width / 2, y_center + height / 2,
        )

    def reliability_score(
        self,
        block: Optional[SubtitleBlock],
        temporal_consistency: float,
        frame_width: float,
        frame_height: float,
        cv_agrees: Optional[bool] = None,
    ) -> float:
        if block is None:
            return 0.0
        boxes = [box for line in block.lines for box in line.boxes]
        confidence = sum(box.confidence for box in boxes) / len(boxes)
        bbox = block.bbox
        width_ratio, height_ratio = bbox.width / frame_width, bbox.height / frame_height
        token_score = min(1.0, len(boxes) / 2)
        aspect_score = min(1.0, (width_ratio / max(height_ratio, 0.001)) / 8)
        width_score = min(1.0, width_ratio / 0.15)
        cv_score = 1.0 if cv_agrees is True else 0.5 if cv_agrees is None else 0.0
        return round(
            0.30 * confidence + 0.20 * token_score + 0.15 * aspect_score
            + 0.15 * width_score + 0.15 * temporal_consistency + 0.05 * cv_score,
            4,
        )

    def select_strategy(
        self, bbox: Optional[BoundingBox], reliability: float, frame_width: float, frame_height: float
    ) -> str:
        if bbox is None:
            return "full_width_fallback"
        area_ratio = bbox.area / (frame_width * frame_height)
        if reliability >= 0.80 and area_ratio < 0.12:
            return "expanded_line_bbox"
        if area_ratio < 0.20:
            return "local_band"
        return "full_width_fallback"

    @staticmethod
    def production_style_replacement(
        strategy: str,
        mask_area_ratio: float,
        coordinate_mapping: str = "PASS",
    ) -> str:
        if coordinate_mapping != "PASS":
            return "FAIL"
        if strategy in {"full_width_fallback", "no_subtitle_skip"}:
            return "CONDITIONAL"
        if mask_area_ratio > 0.35:
            return "FAIL"
        if strategy == "expanded_line_bbox" and mask_area_ratio < 0.12:
            return "PASS"
        if strategy == "local_band" and mask_area_ratio < 0.20:
            return "PASS"
        return "CONDITIONAL"

    @staticmethod
    def validate_coordinate_mapping(
        bbox: Optional[BoundingBox],
        cv_text_bbox: Optional[BoundingBox],
        minimum_iou: float = 0.05,
    ) -> str:
        if bbox is None or cv_text_bbox is None:
            return "FAIL"
        intersects = (
            bbox.left < cv_text_bbox.right
            and bbox.right > cv_text_bbox.left
            and bbox.top < cv_text_bbox.bottom
            and bbox.bottom > cv_text_bbox.top
        )
        return "PASS" if intersects and bbox.iou(cv_text_bbox) >= minimum_iou else "FAIL"

    def caption_report(
        self,
        caption_id: str,
        sampled_frames: List[dict],
        selected_block: Optional[SubtitleBlock],
        frame_width: float,
        frame_height: float,
        cv_agrees: Optional[bool] = None,
    ) -> dict:
        bbox, consistency = self.aggregate(
            frame["selected_block"].bbox
            for frame in sampled_frames
            if frame.get("selected_block") is not None
        )
        reliability = self.reliability_score(
            selected_block, consistency, frame_width, frame_height, cv_agrees
        )
        strategy = self.select_strategy(bbox, reliability, frame_width, frame_height)
        area = bbox.area / (frame_width * frame_height) if bbox else 1.0
        return {
            "caption_id": caption_id,
            "sampled_frames": sampled_frames,
            "selected_block": selected_block.to_dict() if selected_block else None,
            "aggregated_source_bbox": bbox.to_dict() if bbox else None,
            "temporal_consistency_score": consistency,
            "reliability_score": reliability,
            "strategy": strategy,
            "mask_area_ratio": area,
            "fallback_reason": "missing_or_unreliable_detection" if strategy == "full_width_fallback" else None,
            "production_style_replacement": self.production_style_replacement(strategy, area),
        }
