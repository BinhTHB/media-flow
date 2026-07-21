from typing import List, Optional, Tuple

from src.domain.entities.placement_config import PlacementConfig
from src.domain.entities.segment import Segment
from src.domain.entities.subtitle_placement import SubtitlePlacement
from src.domain.entities.subtitle_region import SubtitleRegion
from src.domain.entities.subtitle_style import SubtitleStyle
from src.shared.exceptions import ValidationError


class SubtitlePlacementService:
    def __init__(self, config: PlacementConfig) -> None:
        self.config = config

    def match_regions_to_subtitles(
        self,
        regions: List[SubtitleRegion],
        subtitles: List[Segment],
    ) -> List[Tuple[Segment, Optional[SubtitleRegion]]]:
        matched: List[Tuple[Segment, Optional[SubtitleRegion]]] = []
        for subtitle in subtitles:
            candidates = [
                region
                for region in regions
                if subtitle.start < region.end_time
                and subtitle.end > region.start_time
            ]
            if subtitle.bbox:
                # Prefer regions that spatially overlap the subtitle's own
                # track bbox; a long-lived track must be masked where it is
                # actually rendered, not wherever confidence is highest.
                spatial = [
                    region
                    for region in candidates
                    if self._region_intersects_bbox(region, subtitle.bbox)
                ]
                if spatial:
                    candidates = spatial
            best_region: Optional[SubtitleRegion] = None
            best_confidence = -1.0
            for region in candidates:
                if region.confidence > best_confidence:
                    best_confidence = region.confidence
                    best_region = region
            matched.append((subtitle, best_region))
        return matched

    def _region_intersects_bbox(
        self, region: SubtitleRegion, bbox: dict
    ) -> bool:
        left = region.x * self.config.video_width
        right = (region.x + region.width) * self.config.video_width
        top = region.y * self.config.video_height
        bottom = (region.y + region.height) * self.config.video_height
        return not (
            right <= bbox["left"]
            or bbox["right"] <= left
            or bottom <= bbox["top"]
            or bbox["bottom"] <= top
        )

    def compute_placement(
        self,
        region: Optional[SubtitleRegion],
        subtitle_idx: int,
    ) -> SubtitlePlacement:
        if self.config.placement_mode == "fixed_safe_zone":
            return SubtitlePlacement(
                alignment=2,
                x=int(
                    (self.config.safe_zone_left + self.config.safe_zone_right)
                    * self.config.video_width
                    / 2
                ),
                y=int(self.config.safe_zone_bottom * self.config.video_height),
            )
        if (
            self.config.placement_mode == "replace_source_subtitle"
            and (region is None or region.confidence < self.config.min_region_confidence)
        ):
            band_top = self.config.video_height * 0.72
            band_bottom = self.config.video_height
            return SubtitlePlacement(
                alignment=5,
                x=int(self.config.video_width / 2),
                y=int((band_top + band_bottom) / 2),
                mask_bbox={
                    "left": 0.0,
                    "top": band_top,
                    "right": float(self.config.video_width),
                    "bottom": band_bottom,
                },
                is_fallback=True,
            )
        if region is None or region.confidence < self.config.min_region_confidence:
            return SubtitlePlacement(
                alignment=self.config.fallback_alignment,
                margin_v=self.config.fallback_margin_v,
            )

        source_top = region.y * self.config.video_height
        source_bottom = (region.y + region.height) * self.config.video_height
        target_x = int((region.x + region.width / 2) * self.config.video_width)

        if (
            self.config.placement_mode == "replace_source_subtitle"
            or self.config.preferred_relation == "override"
        ):
            mask_bbox = None
            if self.config.placement_mode == "replace_source_subtitle":
                padding = max(32, self.config.padding_px)
                band_top = source_top - padding
                band_bottom = source_bottom + padding
                mask_bbox = {
                    "left": max(
                        0.0,
                        region.x * self.config.video_width - padding,
                    ),
                    "top": max(0.0, band_top),
                    "right": min(
                        float(self.config.video_width),
                        (region.x + region.width) * self.config.video_width
                        + padding,
                    ),
                    "bottom": min(
                        float(self.config.video_height),
                        band_bottom,
                    ),
                }
                target_x = self.config.video_width / 2
                source_top = mask_bbox["top"]
                source_bottom = mask_bbox["bottom"]
            return SubtitlePlacement(
                alignment=5,
                x=self._clamp_x(target_x),
                y=self._clamp_center_y(int((source_top + source_bottom) / 2)),
                mask_bbox=mask_bbox,
                is_fallback=False,
            )
        if self.config.preferred_relation == "below_source" and self._fits_below(
            source_bottom
        ):
            return SubtitlePlacement(
                alignment=8,
                x=self._clamp_x(target_x),
                y=int(source_bottom + self.config.padding_px),
            )
        if self._fits_above(source_top):
            return SubtitlePlacement(
                alignment=2,
                x=self._clamp_x(target_x),
                y=int(source_top - self.config.padding_px),
            )
        if self._fits_below(source_bottom):
            return SubtitlePlacement(
                alignment=8,
                x=self._clamp_x(target_x),
                y=int(source_bottom + self.config.padding_px),
            )
        return SubtitlePlacement(
            alignment=self.config.fallback_alignment,
            margin_v=self.config.fallback_margin_v,
        )

    def calculate_placements(
        self,
        regions: List[SubtitleRegion],
        subtitles: List[Segment],
        style: Optional[SubtitleStyle] = None,
    ) -> List[SubtitlePlacement]:
        style = style or SubtitleStyle()
        placements: List[SubtitlePlacement] = []
        for subtitle_idx, (subtitle, region) in enumerate(
            self.match_regions_to_subtitles(regions, subtitles)
        ):
            placement = self.compute_placement(region, subtitle_idx)
            if placement.x is not None and placement.y is not None:
                placement = self._resolve_collisions(
                    subtitle, placement, subtitles[:subtitle_idx], placements
                )
            text = subtitle.translated_text or subtitle.text
            placement = self._fit_rendered_bbox(text, placement, style)
            placement.rendered_bbox = self.estimate_rendered_bbox(text, placement, style)
            if (
                self.config.placement_mode == "replace_source_subtitle"
                and placement.mask_bbox is not None
            ):
                placement.mask_bbox = self._expand_mask_for_rendered_text(
                    placement.mask_bbox, placement.rendered_bbox
                )
            if (
                self.config.placement_mode == "replace_source_subtitle"
                and placement.mask_bbox is None
            ):
                raise ValidationError(
                    f"Replacement placement {subtitle.id} did not create mask_bbox."
                )
            placements.append(placement)
        return placements

    def _expand_mask_for_rendered_text(
        self, source_mask: dict, rendered_bbox: dict
    ) -> dict:
        return {
            "left": max(
                0.0,
                min(source_mask["left"], rendered_bbox["left"] - self.config.padding_px),
            ),
            "top": max(
                0.0,
                min(source_mask["top"], rendered_bbox["top"] - self.config.padding_px),
            ),
            "right": min(
                float(self.config.video_width),
                max(source_mask["right"], rendered_bbox["right"] + self.config.padding_px),
            ),
            "bottom": min(
                float(self.config.video_height),
                max(source_mask["bottom"], rendered_bbox["bottom"] + self.config.padding_px),
            ),
        }

    def estimate_rendered_bbox(
        self,
        text: str,
        placement: SubtitlePlacement,
        style: SubtitleStyle,
    ) -> dict:
        lines = text.replace(r"\N", "\n").splitlines() or [""]
        char_width = style.font_size * 0.55
        width = min(
            max(len(line) * char_width for line in lines),
            self._safe_right - self._safe_left,
        )
        height = max(1, len(lines)) * style.font_size * style.line_height
        x = placement.x if placement.x is not None else self.config.video_width / 2
        if placement.y is not None:
            y = placement.y
        elif placement.alignment in (1, 2, 3):
            y = self.config.video_height - (placement.margin_v or self.config.fallback_margin_v)
        elif placement.alignment in (7, 8, 9):
            y = placement.margin_v or self.config.fallback_margin_v
        else:
            y = self.config.video_height / 2
        if placement.alignment in (1, 4, 7):
            left, right = x, x + width
        elif placement.alignment in (3, 6, 9):
            left, right = x - width, x
        else:
            left, right = x - width / 2, x + width / 2
        if placement.alignment in (1, 2, 3):
            top, bottom = y - height, y
        elif placement.alignment in (7, 8, 9):
            top, bottom = y, y + height
        else:
            top, bottom = y - height / 2, y + height / 2
        return {"left": left, "top": top, "right": right, "bottom": bottom}

    @property
    def _safe_left(self) -> float:
        return self.config.safe_zone_left * self.config.video_width

    @property
    def _safe_right(self) -> float:
        return self.config.safe_zone_right * self.config.video_width

    def _fit_rendered_bbox(
        self, text: str, placement: SubtitlePlacement, style: SubtitleStyle
    ) -> SubtitlePlacement:
        bbox = self.estimate_rendered_bbox(text, placement, style)
        x = placement.x if placement.x is not None else self.config.video_width / 2
        y = placement.y
        if y is None:
            if placement.alignment in (1, 2, 3):
                y = self.config.video_height - (placement.margin_v or self.config.fallback_margin_v)
            elif placement.alignment in (7, 8, 9):
                y = placement.margin_v or self.config.fallback_margin_v
            else:
                y = self.config.video_height / 2
        x += max(0, self._safe_left - bbox["left"])
        x -= max(0, bbox["right"] - self._safe_right)
        y += max(0, self._safe_top - bbox["top"])
        y -= max(0, bbox["bottom"] - self._safe_bottom)
        return SubtitlePlacement(
            alignment=placement.alignment,
            x=round(x),
            y=round(y),
            margin_v=placement.margin_v,
            margin_l=placement.margin_l,
            margin_r=placement.margin_r,
            mask_bbox=placement.mask_bbox,
        )

    def build_quality_report(
        self,
        regions: List[SubtitleRegion],
        subtitles: List[Segment],
        placements: List[SubtitlePlacement],
        style: SubtitleStyle,
    ) -> dict:
        safe_zone_violations = 0
        source_overlap_count = 0
        fallback_count = 0
        clipping_violations = []
        caption_reports = []
        source_visibility_not_evaluated = 0
        matches = self.match_regions_to_subtitles(regions, subtitles)
        for index, ((subtitle, region), placement) in enumerate(zip(matches, placements), 1):
            text = subtitle.translated_text or subtitle.text
            bbox = placement.rendered_bbox or self.estimate_rendered_bbox(
                text, placement, style
            )
            if placement.is_fallback:
                fallback_count += 1
            outside_safe_zone = (
                bbox["left"] < self._safe_left - 1
                or bbox["right"] > self._safe_right + 1
                or bbox["top"] < self._safe_top - 1
                or bbox["bottom"] > self._safe_bottom + 1
            )
            if outside_safe_zone:
                safe_zone_violations += 1
            overlaps_source = (
                region is not None
                and bbox["left"] < (region.x + region.width) * self.config.video_width
                and bbox["right"] > region.x * self.config.video_width
                and bbox["top"] < (region.y + region.height) * self.config.video_height
                and bbox["bottom"] > region.y * self.config.video_height
            )
            if region is not None:
                source_bbox = {
                    "left": region.x * self.config.video_width,
                    "top": region.y * self.config.video_height,
                    "right": (region.x + region.width) * self.config.video_width,
                    "bottom": (region.y + region.height) * self.config.video_height,
                }
            else:
                source_bbox = None
            mask = placement.mask_bbox
            mask_covers_source = bool(
                source_bbox
                and mask
                and mask["left"] <= max(0.0, source_bbox["left"] - self.config.padding_px)
                and mask["top"] <= max(0.0, source_bbox["top"] - self.config.padding_px)
                and mask["right"] >= min(float(self.config.video_width), source_bbox["right"] + self.config.padding_px)
                and mask["bottom"] >= min(float(self.config.video_height), source_bbox["bottom"] + self.config.padding_px)
            )
            used_full_width_band = bool(
                mask
                and mask["left"] <= 0
                and mask["right"] >= self.config.video_width
            )
            mask_area_ratio = (
                (mask["right"] - mask["left"])
                * (mask["bottom"] - mask["top"])
                / (self.config.video_width * self.config.video_height)
                if mask
                else 0.0
            )
            if placement.is_fallback:
                strategy = "full_width_fallback"
                fallback_reason = "source_bbox_missing_or_unreliable"
            elif used_full_width_band:
                strategy = "local_band"
                fallback_reason = "translated_text_exceeds_precise_bbox"
            else:
                strategy = "precise_bbox"
                fallback_reason = None
            replacement_violation = (
                self.config.placement_mode == "replace_source_subtitle"
                and (
                    not mask
                    or not mask_covers_source
                    or (placement.is_fallback and not used_full_width_band)
                )
            )
            if replacement_violation:
                source_overlap_count += 1
            elif self.config.placement_mode != "replace_source_subtitle" and overlaps_source:
                source_overlap_count += 1
            source_visibility_not_evaluated += 1
            clipped = (
                bbox["left"] < 0
                or bbox["right"] > self.config.video_width
                or bbox["top"] < 0
                or bbox["bottom"] > self.config.video_height
            )
            if clipped:
                clipping_violations.append(index)
            caption_reports.append(
                {
                    "caption_index": index,
                    "strategy": strategy,
                    "selected_source_bbox": source_bbox,
                    "selected_source_boxes": [source_bbox] if source_bbox else [],
                    "ignored_boxes": [],
                    "reliability_score": region.confidence if region else 0.0,
                    "mask_area_ratio": round(mask_area_ratio, 6),
                    "target_text_bbox": bbox,
                    "bbox_temporal_consistency": (
                        "single_region_estimate" if region else "unavailable"
                    ),
                    "fallback_reason": fallback_reason,
                    "rendered_bbox": bbox,
                    "clipped": clipped,
                    "safe_zone_violation": outside_safe_zone,
                    "mask_applied": placement.mask_bbox is not None,
                    "source_region_bbox": source_bbox,
                    "mask_bbox": mask,
                    "mask_covers_source_region": mask_covers_source,
                    "source_subtitle_visible": "not_evaluated",
                    "used_full_width_band": used_full_width_band,
                    "is_fallback": placement.is_fallback,
                }
            )
        contrast_ratio = self._contrast_ratio(
            style.primary_color, style.outline_color
        )
        unexpected_source_overlaps = (
            source_overlap_count
            if self.config.placement_mode == "avoid_source_subtitle"
            else 0
        )
        warnings = []
        if safe_zone_violations:
            warnings.append("placement_outside_safe_zone")
        if clipping_violations:
            warnings.append("rendered_subtitle_clipping")
        if unexpected_source_overlaps:
            warnings.append("source_subtitle_overlap")
        if self.config.placement_mode == "replace_source_subtitle" and source_overlap_count:
            warnings.append("invalid_source_subtitle_mask")
        if any(
            caption["strategy"] == "full_width_fallback"
            for caption in caption_reports
        ):
            warnings.append("production_replacement_conditional_full_width_fallback")
        if any(
            caption["strategy"] == "precise_bbox"
            and caption["mask_area_ratio"] >= 0.12
            for caption in caption_reports
        ):
            warnings.append("precise_mask_area_too_large")
        if any(
            caption["strategy"] == "local_band"
            and caption["mask_area_ratio"] >= 0.20
            for caption in caption_reports
        ):
            warnings.append("local_band_area_too_large")
        if contrast_ratio < 4.5:
            warnings.append("insufficient_text_contrast")
        return {
            "placement_mode": self.config.placement_mode,
            "play_res_x": self.config.video_width,
            "play_res_y": self.config.video_height,
            "caption_count": len(placements),
            "fallback_count": fallback_count,
            "safe_zone_violations": safe_zone_violations,
            "clipping_violations": clipping_violations,
            "source_overlap_count": source_overlap_count,
            "source_visibility_not_evaluated_count": source_visibility_not_evaluated,
            "contrast_ratio": round(contrast_ratio, 2),
            "minimum_contrast_ratio": 4.5,
            "warnings": warnings,
            "captions": caption_reports,
            "final_placement_verdict": "PASS" if not warnings else "FAIL",
            "quality_gate_status": "PASS" if not warnings else "FAIL",
        }

    def _fits_below(self, source_bottom: float) -> bool:
        return (
            source_bottom + self.config.padding_px + self.config.estimated_height_px
            <= self._safe_bottom
        )

    def _fits_above(self, source_top: float) -> bool:
        return (
            source_top - self.config.padding_px - self.config.estimated_height_px
            >= self._safe_top
        )

    @property
    def _safe_top(self) -> float:
        return self.config.safe_zone_top * self.config.video_height

    @property
    def _safe_bottom(self) -> float:
        return self.config.safe_zone_bottom * self.config.video_height

    def _clamp_x(self, x: int) -> int:
        return max(
            int(self.config.safe_zone_left * self.config.video_width),
            min(x, int(self.config.safe_zone_right * self.config.video_width)),
        )

    def _clamp_center_y(self, y: int) -> int:
        half_height = self.config.estimated_height_px / 2
        return int(
            max(
                self._safe_top + half_height,
                min(y, self._safe_bottom - half_height),
            )
        )

    def _resolve_collisions(
        self,
        subtitle: Segment,
        placement: SubtitlePlacement,
        previous_subtitles: List[Segment],
        previous_placements: List[SubtitlePlacement],
    ) -> SubtitlePlacement:
        for previous_subtitle, previous_placement in zip(
            previous_subtitles, previous_placements
        ):
            if not self._times_overlap(subtitle, previous_subtitle):
                continue
            if not self._placements_overlap(placement, previous_placement):
                continue
            if placement.alignment == 8:
                placement.y += self.config.estimated_height_px
                if (
                    placement.y + self.config.estimated_height_px
                    > self._safe_bottom
                ):
                    return SubtitlePlacement(
                        alignment=self.config.fallback_alignment,
                        margin_v=self.config.fallback_margin_v,
                    )
            else:
                placement.y -= self.config.estimated_height_px
                if (
                    placement.y - self.config.estimated_height_px
                    < self._safe_top
                ):
                    return SubtitlePlacement(
                        alignment=self.config.fallback_alignment,
                        margin_v=self.config.fallback_margin_v,
                    )
        return placement

    @staticmethod
    def _times_overlap(left: Segment, right: Segment) -> bool:
        return left.start < right.end and left.end > right.start

    def _placements_overlap(
        self,
        left: SubtitlePlacement,
        right: SubtitlePlacement,
    ) -> bool:
        if (
            left.x is None
            or left.y is None
            or right.x is None
            or right.y is None
            or abs(left.x - right.x) >= self.config.estimated_height_px
        ):
            return False
        left_top, left_bottom = self._vertical_bounds(left)
        right_top, right_bottom = self._vertical_bounds(right)
        return left_top < right_bottom and left_bottom > right_top

    def _vertical_bounds(self, placement: SubtitlePlacement) -> Tuple[float, float]:
        if placement.alignment == 5:
            half_height = self.config.estimated_height_px / 2
            return placement.y - half_height, placement.y + half_height
        if placement.alignment == 8:
            return placement.y, placement.y + self.config.estimated_height_px
        return placement.y - self.config.estimated_height_px, placement.y

    def _overlaps_region(
        self, placement: SubtitlePlacement, region: SubtitleRegion
    ) -> bool:
        source_left = region.x * self.config.video_width
        source_right = (region.x + region.width) * self.config.video_width
        source_top = region.y * self.config.video_height
        source_bottom = (region.y + region.height) * self.config.video_height
        top, bottom = self._vertical_bounds(placement)
        return (
            source_left <= placement.x <= source_right
            and top < source_bottom
            and bottom > source_top
        )

    @classmethod
    def _contrast_ratio(cls, foreground: str, background: str) -> float:
        lighter, darker = sorted(
            (cls._relative_luminance(foreground), cls._relative_luminance(background)),
            reverse=True,
        )
        return (lighter + 0.05) / (darker + 0.05)

    @staticmethod
    def _relative_luminance(color: str) -> float:
        value = color.lstrip("#")
        if len(value) != 6:
            return 0.0
        channels = [int(value[index:index + 2], 16) / 255 for index in (0, 2, 4)]
        linear = [
            channel / 12.92
            if channel <= 0.04045
            else ((channel + 0.055) / 1.055) ** 2.4
            for channel in channels
        ]
        return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]
