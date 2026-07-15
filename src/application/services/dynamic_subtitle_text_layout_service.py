from dataclasses import dataclass, field
from typing import Mapping, Protocol, Sequence

from src.application.ports.subtitle_text_measurer import SubtitleTextMeasurer
from src.application.services.translation_quality_gate import (
    TranslationQualityResult,
)
from src.application.services.dynamic_subtitle_timeline_service import (
    TimelineMappingResult,
)

BBox = Mapping[str, float]

@dataclass(frozen=True)
class DynamicSubtitleLayoutConfig:
    max_lines: int = 2
    minimum_font_size: int = 16
    horizontal_padding: int = 12
    vertical_padding: int = 8
    line_spacing_ratio: float = 1.15
    maximum_width_ratio: float = 1.0
    maximum_height_ratio: float = 1.0

    def __post_init__(self) -> None:
        if self.max_lines < 1:
            raise ValueError("max_lines must be >= 1")
        if self.minimum_font_size < 1:
            raise ValueError("minimum_font_size must be >= 1")
        for name in ("horizontal_padding", "vertical_padding"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0")
        if self.line_spacing_ratio <= 0:
            raise ValueError("line_spacing_ratio must be positive")
        if not 0 < self.maximum_width_ratio <= 1:
            raise ValueError("maximum_width_ratio must be in (0, 1]")
        if not 0 < self.maximum_height_ratio <= 1:
            raise ValueError("maximum_height_ratio must be in (0, 1]")

@dataclass(frozen=True)
class DynamicSubtitleLayoutResult:
    interval_id: str
    layout_status: str
    layout_reason: str
    lines: tuple[str, ...]
    font_size: int | None
    position_x: float | None
    position_y: float | None
    text_width: float | None
    text_height: float | None
    source_bbox: BBox | None
    expanded_cover_bbox: BBox | None

FIT = "FIT"
REVIEW_REQUIRED = "REVIEW_REQUIRED"
SKIPPED = "SKIPPED"

class DynamicSubtitleTextLayoutService:
    """Pure Application service that calculates text layout within a cover bbox.

    Given a TimelineMappingResult, frame geometry, and subtitle style,
    determines whether the translated target text fits in the expanded cover
    region, and if so, computes the optimal font size, line wrapping,
    and centered position.
    """

    def __init__(
        self,
        measurer: SubtitleTextMeasurer,
        config: DynamicSubtitleLayoutConfig | None = None,
    ) -> None:
        self._measurer = measurer
        self._config = config or DynamicSubtitleLayoutConfig()

    def layout(
        self,
        mapping: TimelineMappingResult,
        frame_width: int,
        frame_height: int,
    ) -> DynamicSubtitleLayoutResult:
        if mapping.safety_reason:
            return self._build(mapping, SKIPPED, mapping.safety_reason)
        if mapping.mapping_status != "mapped":
            return self._build(mapping, REVIEW_REQUIRED, "mapping_not_eligible")
        target = mapping.target_text.strip()
        if not target:
            return self._build(mapping, REVIEW_REQUIRED, "empty_target_text")
        if mapping.expanded_cover_bbox is None:
            return self._build(mapping, REVIEW_REQUIRED, "missing_cover_bbox")
        if frame_width <= 0 or frame_height <= 0:
            return self._build(mapping, REVIEW_REQUIRED, "invalid_frame_geometry")
        bbox = mapping.expanded_cover_bbox
        if not (bbox["right"] > bbox["left"] and bbox["bottom"] > bbox["top"]):
            return self._build(mapping, REVIEW_REQUIRED, "missing_cover_bbox")
        if (
            bbox["left"] < 0
            or bbox["top"] < 0
            or bbox["right"] > frame_width
            or bbox["bottom"] > frame_height
        ):
            return self._build(mapping, REVIEW_REQUIRED, "cover_bbox_out_of_frame")
        cover_w = bbox["right"] - bbox["left"]
        cover_h = bbox["bottom"] - bbox["top"]
        max_text_w = cover_w * self._config.maximum_width_ratio - 2 * self._config.horizontal_padding
        max_text_h = cover_h * self._config.maximum_height_ratio - 2 * self._config.vertical_padding
        base_style = self._style_from_mapping(mapping)
        style_font = base_style["font_size"]
        bold = base_style["bold"]
        spacing = base_style["spacing"]
        font_family = base_style["font_family"]
        best = None
        start_font = max(style_font, self._config.minimum_font_size)
        for font_size in range(start_font, self._config.minimum_font_size - 1, -1):
            result = self._try_fit(
                target, font_size, max_text_w, max_text_h,
                font_family, bold, spacing,
            )
            if result is not None:
                best = (font_size, result)
                break
        if best is None:
            return self._build(mapping, REVIEW_REQUIRED, "target_text_does_not_fit")
        font_size, (lines, line_widths) = best
        text_w = max(line_widths)
        text_h = self._line_block_height(len(lines), font_size)
        center_x = (bbox["left"] + bbox["right"]) / 2
        center_y = (bbox["top"] + bbox["bottom"]) / 2
        return DynamicSubtitleLayoutResult(
            interval_id=mapping.interval_id,
            layout_status=FIT,
            layout_reason="fit",
            lines=tuple(lines),
            font_size=font_size,
            position_x=center_x,
            position_y=center_y,
            text_width=text_w,
            text_height=text_h,
            source_bbox=mapping.source_bbox,
            expanded_cover_bbox=mapping.expanded_cover_bbox,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build(
        self,
        mapping: TimelineMappingResult,
        status: str,
        reason: str,
    ) -> DynamicSubtitleLayoutResult:
        return DynamicSubtitleLayoutResult(
            interval_id=mapping.interval_id,
            layout_status=status,
            layout_reason=reason,
            lines=(),
            font_size=None,
            position_x=None,
            position_y=None,
            text_width=None,
            text_height=None,
            source_bbox=mapping.source_bbox,
            expanded_cover_bbox=mapping.expanded_cover_bbox,
        )

    def _style_from_mapping(self, mapping: TimelineMappingResult) -> dict:
        meta = mapping.metadata or {}
        return {
            "font_size": meta.get("font_size", 24),
            "bold": meta.get("bold", False),
            "spacing": meta.get("spacing", 0.0),
            "font_family": meta.get("font_family", "Arial"),
        }

    def _try_fit(
        self,
        text: str,
        font_size: int,
        max_w: float,
        max_h: float,
        font_family: str,
        bold: bool,
        spacing: float,
    ) -> tuple[list[str], list[float]] | None:
        # 1 line
        w1, h1 = self._measurer.measure(
            text, font_family=font_family, font_size=font_size,
            bold=bold, spacing=spacing,
        )
        if w1 <= max_w and h1 <= max_h:
            return [text], [w1]
        if self._config.max_lines < 2:
            return None
        # 2+ lines balanced/greedy
        lines = self._wrap(text, font_size, max_w, font_family, bold, spacing)
        if lines is None or len(lines) > self._config.max_lines:
            return None
        total_h = self._line_block_height(len(lines), font_size)
        if total_h > max_h:
            return None
        widths = []
        for line in lines:
            lw, _ = self._measurer.measure(
                line, font_family=font_family, font_size=font_size,
                bold=bold, spacing=spacing,
            )
            if lw > max_w:
                return None
            widths.append(lw)
        return lines, widths

    def _wrap(
        self,
        text: str,
        font_size: int,
        max_w: float,
        font_family: str,
        bold: bool,
        spacing: float,
    ) -> list[str] | None:
        words = text.split()
        if not words:
            return None
        if len(words) == 1:
            w, _ = self._measurer.measure(
                words[0], font_family=font_family, font_size=font_size,
                bold=bold, spacing=spacing,
            )
            return [words[0]] if w <= max_w else None
        # Try balanced 2-line split
        if self._config.max_lines >= 2:
            best = self._split_two_balanced(
                words, font_size, max_w, font_family, bold, spacing,
            )
            if best is not None:
                return list(best)
        # Greedy wrap for more lines
        lines = []
        current: list[str] = []
        for word in words:
            trial = " ".join(current + [word])
            if current:
                tw, _ = self._measurer.measure(
                    trial, font_family=font_family, font_size=font_size,
                    bold=bold, spacing=spacing,
                )
                if tw > max_w:
                    lines.append(" ".join(current))
                    current = [word]
                else:
                    current.append(word)
            else:
                current.append(word)
        if current:
            lines.append(" ".join(current))
        if len(lines) > self._config.max_lines:
            return None
        for line in lines:
            lw, _ = self._measurer.measure(
                line, font_family=font_family, font_size=font_size,
                bold=bold, spacing=spacing,
            )
            if lw > max_w:
                return None
        return lines

    def _split_two_balanced(
        self,
        words: list[str],
        font_size: int,
        max_w: float,
        font_family: str,
        bold: bool,
        spacing: float,
    ) -> tuple[str, str] | None:
        n = len(words)
        best: tuple[str, str] | None = None
        best_score = float("inf")
        for i in range(1, n):
            line1 = " ".join(words[:i])
            line2 = " ".join(words[i:])
            w1, _ = self._measurer.measure(
                line1, font_family=font_family, font_size=font_size,
                bold=bold, spacing=spacing,
            )
            w2, _ = self._measurer.measure(
                line2, font_family=font_family, font_size=font_size,
                bold=bold, spacing=spacing,
            )
            if w1 > max_w or w2 > max_w:
                continue
            score = max(w1, w2)
            if score < best_score:
                best_score = score
                best = (line1, line2)
        return best

    def _line_block_height(self, num_lines: int, font_size: int) -> float:
        if num_lines == 0:
            return 0.0
        return num_lines * font_size * self._config.line_spacing_ratio

    def _measure(
        self,
        text: str,
        font_size: int,
        font_family: str,
        bold: bool,
        spacing: float,
    ) -> tuple[float, float]:
        return self._measurer.measure(
            text,
            font_family=font_family,
            font_size=font_size,
            bold=bold,
            spacing=spacing,
        )
