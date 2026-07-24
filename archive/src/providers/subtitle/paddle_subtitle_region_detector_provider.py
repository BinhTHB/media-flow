import os
import json
import time
import math
import re
from collections.abc import Sequence
from typing import List

from PIL import Image

from src.application.ports.subtitle_region_detector_provider import (
    SubtitleRegionDetectorProvider,
)
from src.domain.entities.frame import Frame
from src.domain.entities.subtitle_region import SubtitleRegion
from src.shared.exceptions import ProviderError
from src.infrastructure.runtime.ocr_device_policy import validate_ocr_device


def transform_polygon(
    polygon: list[list[float]],
    transform: str,
    input_width: float,
    input_height: float,
    output_width: float | None = None,
    output_height: float | None = None,
) -> list[list[float]]:
    output_width = output_width or input_width
    output_height = output_height or input_height
    scale_x, scale_y = output_width / input_width, output_height / input_height
    points = []
    for x, y in polygon:
        if transform == "vertical_flip":
            x, y = x, input_height - y
        elif transform == "horizontal_flip":
            x, y = input_width - x, y
        elif transform == "rotate180":
            x, y = input_width - x, input_height - y
        elif transform == "transpose":
            x, y = y, x
        elif transform != "identity":
            raise ValueError(f"Unsupported coordinate transform: {transform}")
        points.append([float(x * scale_x), float(y * scale_y)])
    return points


def polygon_bbox(polygon: list[list[float]]) -> dict[str, float]:
    return {
        "left": min(point[0] for point in polygon),
        "top": min(point[1] for point in polygon),
        "right": max(point[0] for point in polygon),
        "bottom": max(point[1] for point in polygon),
    }


class PaddleOCRSubtitleRegionDetectorProvider(SubtitleRegionDetectorProvider):
    def __init__(
        self,
        language: str = "ch",
        use_gpu: bool = True,
        min_confidence: float = 0.70,
    ):
        self.language = language
        self.use_gpu = use_gpu
        self.min_confidence = min_confidence
        self._ocr = None

    def _get_ocr(self):
        if self._ocr is None:
            if self.use_gpu:
                validate_ocr_device("gpu")
            try:
                import paddleocr
            except ImportError as e:
                raise ProviderError("paddleocr is not installed") from e
            try:
                if int(getattr(paddleocr, "__version__", "2").split(".", 1)[0]) >= 3:
                    self._ocr = paddleocr.PaddleOCR(
                        lang=self.language,
                        device="gpu" if self.use_gpu else "cpu",
                        enable_mkldnn=False,
                        use_doc_orientation_classify=False,
                        use_doc_unwarping=False,
                        use_textline_orientation=False,
                    )
                else:
                    self._ocr = paddleocr.PaddleOCR(
                        use_angle_cls=True,
                        lang=self.language,
                        use_gpu=self.use_gpu,
                        show_log=False,
                    )
            except Exception as e:
                raise ProviderError(f"PaddleOCR initialization failed: {e}") from e
        return self._ocr

    def detect_regions(self, frames: List[Frame]) -> List[SubtitleRegion]:
        ocr = self._get_ocr()
        regions = []

        for frame in frames:
            # Check if parent directory is named "frames" to enable checkpointing
            parent_dir_name = os.path.basename(os.path.dirname(frame.image_path))
            use_checkpoint = (parent_dir_name == "frames")

            checkpoint_path = None
            frame_name = None
            if use_checkpoint:
                job_dir = os.path.dirname(os.path.dirname(frame.image_path))
                if not job_dir or job_dir == ".":
                    debug_dir = os.path.join("output", "debug")
                else:
                    debug_dir = os.path.join(job_dir, "debug")
                region_frames_dir = os.path.join(debug_dir, "subtitle_region_frames")
                os.makedirs(region_frames_dir, exist_ok=True)
                
                frame_name = os.path.splitext(os.path.basename(frame.image_path))[0]
                frame_match = re.search(r"(\d+)$", frame_name)
                frame_index = int(frame_match.group(1)) if frame_match else None
                checkpoint_path = os.path.join(region_frames_dir, f"{frame_name}.json")

                # Checkpoint resume/reuse
                if os.path.exists(checkpoint_path):
                    try:
                        with open(checkpoint_path, "r", encoding="utf-8") as f:
                            cp = json.load(f)
                        if "error" not in cp or cp["error"] is None:
                            for r_dict in cp.get("regions", []):
                                regions.append(
                                    SubtitleRegion(
                                        start_time=r_dict["start_time"],
                                        end_time=r_dict["end_time"],
                                        x=r_dict["x"],
                                        y=r_dict["y"],
                                        width=r_dict["width"],
                                        height=r_dict["height"],
                                        confidence=r_dict["confidence"],
                                    )
                                )
                            continue
                    except Exception:
                        pass

            try:
                with Image.open(frame.image_path) as image:
                    image_width, image_height = image.size
            except FileNotFoundError as e:
                raise ProviderError(f"Frame image not found: {frame.image_path}") from e
            except Exception as e:
                raise ProviderError(
                    f"Failed to load frame image {frame.image_path}: {e}"
                ) from e

            start_time = time.monotonic()
            error_msg = None
            frame_regions = []
            
            try:
                try:
                    result = ocr.ocr(frame.image_path, cls=True)
                except TypeError:
                    result = ocr.ocr(frame.image_path)
            except Exception as e:
                error_msg = str(e)
                duration = time.monotonic() - start_time
                if use_checkpoint and checkpoint_path:
                    checkpoint_data = {
                        "frame_id": frame_name,
                        "frame_path": frame.image_path,
                        "frame_timestamp": frame.timestamp,
                        "provider_name": "paddle_region_detector",
                        "processing_duration": duration,
                        "error": error_msg,
                        "regions": [],
                    }
                    try:
                        with open(checkpoint_path, "w", encoding="utf-8") as f:
                            json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)
                    except Exception:
                        pass
                raise ProviderError(
                    f"PaddleOCR detection failed for frame {frame.image_path}: {e}"
                ) from e

            duration = time.monotonic() - start_time

            provider_angle = 0
            ocr_input_width, ocr_input_height = image_width, image_height
            if result and isinstance(result[0], dict):
                page = result[0]
                preprocessing = page.get("doc_preprocessor_res") or {}
                provider_angle = int(preprocessing.get("angle") or 0)
                input_image = preprocessing.get("output_img")
                if getattr(input_image, "shape", None) is not None:
                    ocr_input_height, ocr_input_width = input_image.shape[:2]
                result = [[
                    (polygon.tolist(), (text, float(confidence)))
                    for polygon, text, confidence in zip(
                        page.get("rec_polys", []),
                        page.get("rec_texts", []),
                        page.get("rec_scores", []),
                    )
                ]]

            if (
                not result
                or not isinstance(result, Sequence)
                or isinstance(result, (str, bytes))
                or not isinstance(result[0], Sequence)
                or isinstance(result[0], (str, bytes))
            ):
                if use_checkpoint and checkpoint_path:
                    checkpoint_data = {
                        "frame_id": frame_name,
                        "frame_path": frame.image_path,
                        "frame_timestamp": frame.timestamp,
                        "provider_name": "paddle_region_detector",
                        "processing_duration": duration,
                        "error": None,
                        "regions": [],
                    }
                    try:
                        with open(checkpoint_path, "w", encoding="utf-8") as f:
                            json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)
                    except Exception:
                        pass
                continue

            boxes = []
            detailed_boxes = []
            for line in result[0]:
                if not isinstance(line, Sequence) or isinstance(line, (str, bytes)):
                    raise ProviderError("PaddleOCR detection returned an invalid line shape.")
                if len(line) != 2:
                    raise ProviderError("PaddleOCR detection returned an invalid line shape.")

                box, text_confidence = line
                if (
                    not isinstance(box, list)
                    or len(box) != 4
                    or not isinstance(text_confidence, Sequence)
                    or isinstance(text_confidence, (str, bytes))
                    or len(text_confidence) != 2
                ):
                    raise ProviderError("PaddleOCR detection returned an invalid line shape.")

                text, confidence = text_confidence
                if not isinstance(text, str) or not text.strip():
                    continue
                if (
                    not isinstance(confidence, (int, float))
                    or isinstance(confidence, bool)
                    or not 0 <= confidence <= 1
                ):
                    continue

                raw_coordinates = []
                valid_box = True
                for point in box:
                    if (
                        not isinstance(point, Sequence)
                        or isinstance(point, (str, bytes))
                        or len(point) != 2
                        or any(
                            not isinstance(value, (int, float))
                            or isinstance(value, bool)
                            or not math.isfinite(value)
                            for value in point
                        )
                    ):
                        valid_box = False
                        break
                    raw_coordinates.append([float(point[0]), float(point[1])])
                if not valid_box:
                    continue
                coordinate_transform = (
                    "rotate180" if provider_angle == 180 else "identity"
                )
                coordinates = transform_polygon(
                    raw_coordinates,
                    coordinate_transform,
                    ocr_input_width,
                    ocr_input_height,
                    image_width,
                    image_height,
                )

                min_x = max(0, min(image_width, min(point[0] for point in coordinates)))
                max_x = max(0, min(image_width, max(point[0] for point in coordinates)))
                min_y = max(0, min(image_height, min(point[1] for point in coordinates)))
                max_y = max(0, min(image_height, max(point[1] for point in coordinates)))
                x = min_x / image_width
                y = min_y / image_height
                width = (max_x - min_x) / image_width
                height = (max_y - min_y) / image_height

                if width <= 0 or height <= 0:
                    continue
                ignored_reason = None
                if confidence < self.min_confidence:
                    ignored_reason = "low_confidence"
                elif width < 0.015 or height < 0.008:
                    ignored_reason = "isolated_tiny_box"
                elif height > width * 1.5:
                    ignored_reason = "vertical_box"
                elif min_x <= 1 or min_y <= 1 or max_x >= image_width - 1:
                    ignored_reason = "border_artifact"
                box_data = {
                    "text": text.strip(),
                    "confidence": float(confidence),
                    "frame_image_path": frame.image_path,
                    "frame_width": image_width,
                    "frame_height": image_height,
                    "ocr_input_image_path": frame.image_path,
                    "ocr_input_width": ocr_input_width,
                    "ocr_input_height": ocr_input_height,
                    "raw_polygon_from_provider": raw_coordinates,
                    "raw_bbox_from_provider": polygon_bbox(raw_coordinates),
                    "coordinate_transform_applied": coordinate_transform,
                    "transformed_polygon": coordinates,
                    "transformed_bbox": polygon_bbox(coordinates),
                    "final_bbox_coordinate_space": "original_frame_pixels",
                    "polygon": coordinates,
                    "bbox": {
                        "left": float(min_x),
                        "top": float(min_y),
                        "right": float(max_x),
                        "bottom": float(max_y),
                    },
                    "width": float(max_x - min_x),
                    "height": float(max_y - min_y),
                    "x_center": float((min_x + max_x) / 2),
                    "y_center": float((min_y + max_y) / 2),
                    "area": float((max_x - min_x) * (max_y - min_y)),
                    "source": "paddleocr",
                    "normalized_bbox": {
                        "x": x,
                        "y": y,
                        "width": width,
                        "height": height,
                    },
                    "ignored_reason": ignored_reason,
                }
                detailed_boxes.append(box_data)
                if ignored_reason:
                    continue
                boxes.append((x, y, width, height, confidence, box_data))

            if boxes:
                groups = []
                for box in sorted(boxes, key=lambda item: item[1] + item[3] / 2):
                    center_y = box[1] + box[3] / 2
                    group = next(
                        (
                            candidate
                            for candidate in groups
                            if abs(center_y - candidate["center_y"])
                            <= max(box[3], candidate["max_height"]) * 2.5
                        ),
                        None,
                    )
                    if group is None:
                        groups.append({
                            "boxes": [box],
                            "center_y": center_y,
                            "max_height": box[3],
                        })
                    else:
                        group["boxes"].append(box)
                        group["center_y"] = sum(
                            item[1] + item[3] / 2 for item in group["boxes"]
                        ) / len(group["boxes"])
                        group["max_height"] = max(
                            item[3] for item in group["boxes"]
                        )
                selected_group = max(
                    groups,
                    key=lambda group: (
                        max(item[0] + item[2] for item in group["boxes"])
                        - min(item[0] for item in group["boxes"])
                    )
                    * sum(item[4] for item in group["boxes"])
                    / len(group["boxes"])
                    * (2.5 if group["center_y"] >= 0.60 else 1.0),
                )
                selected_boxes = selected_group["boxes"]
                selected_ids = {id(box[5]) for box in selected_boxes}
                for box in boxes:
                    if id(box[5]) not in selected_ids:
                        box[5]["ignored_reason"] = "lower_subtitle_block_score"
                x = min(box[0] for box in selected_boxes)
                y = min(box[1] for box in selected_boxes)
                right = max(box[0] + box[2] for box in selected_boxes)
                bottom = max(box[1] + box[3] for box in selected_boxes)
                r = SubtitleRegion(
                    start_time=frame.timestamp,
                    end_time=frame.timestamp + 2.0,
                    x=x,
                    y=y,
                    width=right - x,
                    height=bottom - y,
                    confidence=max(box[4] for box in selected_boxes),
                )
                regions.append(r)
                frame_regions.append({
                    "start_time": r.start_time,
                    "end_time": r.end_time,
                    "x": r.x,
                    "y": r.y,
                    "width": r.width,
                    "height": r.height,
                    "confidence": r.confidence,
                })

            if use_checkpoint and checkpoint_path:
                checkpoint_data = {
                    "frame_id": frame_name,
                    "frame_index": frame_index,
                    "frame_path": frame.image_path,
                    "source_image_path": frame.image_path,
                    "frame_timestamp": frame.timestamp,
                    "frame_width": image_width,
                    "frame_height": image_height,
                    "provider_name": "paddle_region_detector",
                    "processing_duration": duration,
                    "error": None,
                    "regions": frame_regions,
                    "coordinate_space": "source_frame_pixels",
                    "provider": {
                        "name": "paddle_region_detector",
                        "version": getattr(__import__("paddleocr"), "__version__", None),
                        "device": "gpu" if self.use_gpu else "cpu",
                        "language": self.language,
                        "document_orientation_angle": provider_angle,
                    },
                    "boxes": detailed_boxes,
                }
                try:
                    with open(checkpoint_path, "w", encoding="utf-8") as f:
                        json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)
                except Exception:
                    pass

        return regions