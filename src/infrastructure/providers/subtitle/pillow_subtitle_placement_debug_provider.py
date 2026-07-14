import json
import os
import subprocess
from typing import List, Optional, Tuple

from PIL import Image, ImageDraw

from src.application.ports.subtitle_placement_debug_provider import (
    SubtitlePlacementDebugProvider,
)
from src.domain.entities.frame import Frame
from src.domain.entities.subtitle_placement import SubtitlePlacement
from src.domain.entities.subtitle_region import SubtitleRegion
from src.shared.exceptions import ProviderError


class PillowSubtitlePlacementDebugProvider(SubtitlePlacementDebugProvider):
    def generate(
        self,
        frames: List[Frame],
        regions: List[SubtitleRegion],
        placements: List[SubtitlePlacement],
        before_path: str,
        overlay_path: str,
        after_path: str,
        regions_path: str,
        placements_path: str,
        rendered_video_path: Optional[str] = None,
    ) -> None:
        if not frames:
            raise ProviderError("Cannot generate subtitle placement debug artifacts without frames.")

        source_frame = self._select_source_frame(frames, regions)
        try:
            with Image.open(source_frame.image_path) as source:
                image = source.convert("RGB")
        except Exception as e:
            raise ProviderError(
                f"Failed to open debug source frame {source_frame.image_path}: {e}"
            ) from e

        try:
            self._ensure_parent(before_path)
            image.save(before_path, format="JPEG")

            overlay = image.copy()
            self._draw_regions(overlay, regions, source_frame.timestamp)
            self._draw_placements(overlay, placements)
            self._ensure_parent(overlay_path)
            overlay.save(overlay_path, format="JPEG")

            self._ensure_parent(after_path)
            if not rendered_video_path or not os.path.exists(rendered_video_path):
                raise ProviderError(
                    "Cannot generate after_render.jpg because rendered output.mp4 is missing."
                )
            extraction_method = self._extract_rendered_frame(
                rendered_video_path,
                source_frame.timestamp,
                after_path,
            )
            self._write_json(regions_path, self._serialize_regions(regions))
            self._write_json(placements_path, self._serialize_placements(placements))
            self._write_json(
                os.path.join(os.path.dirname(before_path), "debug_metadata.json"),
                {
                    "selected_timestamp": source_frame.timestamp,
                    "source_frame_path": source_frame.image_path,
                    "rendered_video_path": rendered_video_path,
                    "extraction_method": extraction_method,
                    "before_render_size": os.path.getsize(before_path),
                    "after_render_size": os.path.getsize(after_path),
                    "validation_notes": (
                        "before_render.jpg and overlay_debug.jpg use the selected "
                        "source Frame; after_render.jpg is extracted from output.mp4 "
                        "at the same timestamp."
                    ),
                },
            )
        except Exception as e:
            raise ProviderError(f"Failed to generate subtitle placement debug artifacts: {e}") from e

    @staticmethod
    def _select_source_frame(
        frames: List[Frame],
        regions: List[SubtitleRegion],
    ) -> Frame:
        if regions:
            region_timestamp = regions[0].start_time
            for frame in frames:
                if frame.timestamp == region_timestamp:
                    return frame
        return frames[0]

    @staticmethod
    def _extract_rendered_frame(
        rendered_video_path: str,
        timestamp: float,
        output_path: str,
    ) -> str:
        command = [
            "ffmpeg",
            "-i",
            rendered_video_path,
            "-ss",
            str(timestamp),
            "-frames:v",
            "1",
            "-q:v",
            "2",
            "-y",
            output_path,
        ]
        try:
            subprocess.run(command, capture_output=True, check=True, timeout=60)
        except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            raise ProviderError(
                f"Failed to extract rendered debug frame from {rendered_video_path}: {e}"
            ) from e
        if not os.path.exists(output_path):
            raise ProviderError(
                f"FFmpeg did not produce rendered debug frame at {output_path}"
            )
        return "ffmpeg_output_seek"

    @staticmethod
    def _draw_regions(
        image: Image.Image,
        regions: List[SubtitleRegion],
        timestamp: float,
    ) -> None:
        draw = ImageDraw.Draw(image)
        image_width, image_height = image.size
        for region in regions:
            if region.start_time <= timestamp <= region.end_time:
                left = region.x * image_width
                top = region.y * image_height
                right = (region.x + region.width) * image_width
                bottom = (region.y + region.height) * image_height
                draw.rectangle((left, top, right, bottom), outline="red", width=3)
                draw.text(
                    (left, max(0, top - 12)),
                    f"confidence={region.confidence:.2f}",
                    fill="red",
                )

    @staticmethod
    def _draw_placements(
        image: Image.Image,
        placements: List[SubtitlePlacement],
    ) -> None:
        draw = ImageDraw.Draw(image)
        image_width, image_height = image.size
        for placement in placements:
            if placement.mask_bbox is not None:
                mask = placement.mask_bbox
                mask_box = (
                    mask["left"] * image_width / 1920,
                    mask["top"] * image_height / 1080,
                    mask["right"] * image_width / 1920,
                    mask["bottom"] * image_height / 1080,
                )
                draw.rectangle(mask_box, fill=(80, 80, 80), outline="gray", width=3)
            if placement.rendered_bbox is not None:
                bbox = placement.rendered_bbox
                rendered_box = (
                    bbox["left"] * image_width / 1920,
                    bbox["top"] * image_height / 1080,
                    bbox["right"] * image_width / 1920,
                    bbox["bottom"] * image_height / 1080,
                )
                draw.rectangle(rendered_box, outline="blue", width=3)
            x, y = PillowSubtitlePlacementDebugProvider._placement_coordinates(
                placement,
                image_width,
                image_height,
            )
            is_fallback = placement.x is None or placement.y is None
            color = "orange" if is_fallback else "blue"
            marker_size = 8
            draw.line((x - marker_size, y, x + marker_size, y), fill=color, width=3)
            draw.line((x, y - marker_size, x, y + marker_size), fill=color, width=3)
            label = f"Vietnamese placement: {PillowSubtitlePlacementDebugProvider._strategy(placement)}"
            if is_fallback:
                label += " (global style fallback)"
            draw.text((x + marker_size, y + marker_size), label, fill=color)

    @staticmethod
    def _serialize_regions(regions: List[SubtitleRegion]) -> list:
        return [
            {
                "timestamp": region.start_time,
                "normalized_coordinates": {
                    "x": region.x,
                    "y": region.y,
                    "width": region.width,
                    "height": region.height,
                },
                "confidence": region.confidence,
                "provider": "subtitle_region_detector",
            }
            for region in regions
        ]

    @staticmethod
    def _serialize_placements(placements: List[SubtitlePlacement]) -> list:
        return [
            {
                "alignment": placement.alignment,
                "ass_position": (
                    {"x": placement.x, "y": placement.y}
                    if placement.x is not None and placement.y is not None
                    else None
                ),
                "strategy": PillowSubtitlePlacementDebugProvider._strategy(placement),
                "fallback_reason": (
                    "No region detected or region confidence below threshold."
                    if placement.x is None or placement.y is None
                    else None
                ),
                "rendered_bbox": placement.rendered_bbox,
                "mask_bbox": placement.mask_bbox,
            }
            for placement in placements
        ]

    @staticmethod
    def _strategy(placement: SubtitlePlacement) -> str:
        if placement.x is None or placement.y is None:
            return "global_style_fallback"
        if placement.alignment == 8:
            return "below_source"
        if placement.alignment == 2:
            return "above_source"
        if placement.alignment == 5:
            return "override"
        return "positioned"

    @staticmethod
    def _placement_coordinates(
        placement: SubtitlePlacement,
        image_width: int,
        image_height: int,
    ) -> Tuple[float, float]:
        if placement.x is not None and placement.y is not None:
            return (
                placement.x * image_width / 1920,
                placement.y * image_height / 1080,
            )

        margin_v = placement.margin_v or 0
        return (
            image_width / 2,
            image_height - margin_v * image_height / 1080,
        )

    @staticmethod
    def _ensure_parent(path: str) -> None:
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)

    def _write_json(self, path: str, data: list) -> None:
        self._ensure_parent(path)
        with open(path, "w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False)
