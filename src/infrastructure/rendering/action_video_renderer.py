"""
Action Video Renderer

Executes a RenderingPlan on an input video file using OpenCV pixel manipulation
and FFmpeg encoding/audio-muxing.

Actions supported:
- blur: Real pixel Gaussian blur over frame bounding boxes.
- solid_cover: Solid rectangle fill over bounding boxes.
- replace_image: Alpha-composited PNG/JPEG image overlay fitted to bounding boxes.
- replace_text: Vietnamese Unicode text rendering via PIL.
- preserve: No pixel modification.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.domain.entities.action_plan import RenderingPlan, TrackActionPlan
from src.domain.entities.processing_policy import ActionType

logger = logging.getLogger(__name__)


def _hex_to_bgr(hex_color: str) -> Tuple[int, int, int]:
    hex_color = hex_color.lstrip("#")
    if len(hex_color) == 6:
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        return (b, g, r)
    return (0, 0, 0)


class ActionVideoRenderer:
    def __init__(self, font_path: Optional[str] = None):
        self.font_path = font_path

    def render(
        self,
        input_video_path: str,
        rendering_plan: RenderingPlan,
        output_video_path: str,
        ffmpeg_encoder: str = "auto",
        nvenc_preset: str = "p5",
        nvenc_cq: int = 23,
    ) -> str:
        output_path = Path(output_video_path).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temp_dir = output_path.parent / f"_temp_render_{rendering_plan.job_id}"
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_video = temp_dir / "rendered_no_audio.mp4"

        # Index frame actions by frame_index for fast frame-by-frame lookup
        actions_by_frame: Dict[int, List[Tuple[TrackActionPlan, Tuple[int, int, int, int]]]] = {}
        for action in rendering_plan.actions:
            if action.action_type == ActionType.PRESERVE.value:
                continue
            for fw in action.frame_windows:
                fi = fw.frame_index
                x1, y1, x2, y2 = [int(round(v)) for v in fw.bbox_xyxy]
                # Apply padding
                padding = int(action.params.get("padding_px", 4))
                x1 = max(0, x1 - padding)
                y1 = max(0, y1 - padding)
                x2 = min(rendering_plan.video_width, x2 + padding)
                y2 = min(rendering_plan.video_height, y2 + padding)
                if x2 > x1 and y2 > y1:
                    actions_by_frame.setdefault(fi, []).append((action, (x1, y1, x2, y2)))

        cap = cv2.VideoCapture(str(input_video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open input video for rendering: {input_video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or rendering_plan.fps or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or rendering_plan.video_width
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or rendering_plan.video_height

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(temp_video), fourcc, fps, (width, height))

        frame_idx = 0
        loaded_image_assets: Dict[str, np.ndarray] = {}

        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            if frame_idx in actions_by_frame:
                frame_actions = actions_by_frame[frame_idx]
                for action_plan, (x1, y1, x2, y2) in frame_actions:
                    act_type = action_plan.action_type

                    if act_type == ActionType.BLUR.value:
                        strength = float(action_plan.params.get("blur_strength", 25.0))
                        ksize = int(round(strength))
                        if ksize % 2 == 0:
                            ksize += 1
                        ksize = max(3, ksize)
                        roi = frame[y1:y2, x1:x2]
                        if roi.size > 0:
                            blurred_roi = cv2.GaussianBlur(roi, (ksize, ksize), 0)
                            frame[y1:y2, x1:x2] = blurred_roi

                    elif act_type == ActionType.SOLID_COVER.value:
                        color_hex = action_plan.params.get("solid_color_hex", "#000000")
                        bgr = _hex_to_bgr(color_hex)
                        frame[y1:y2, x1:x2] = bgr

                    elif act_type == ActionType.REPLACE_IMAGE.value:
                        img_path = action_plan.replacement_image_path
                        if img_path and os.path.exists(img_path):
                            if img_path not in loaded_image_assets:
                                asset = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
                                loaded_image_assets[img_path] = asset
                            asset = loaded_image_assets[img_path]

                            if asset is not None:
                                target_w = x2 - x1
                                target_h = y2 - y1
                                resized = cv2.resize(asset, (target_w, target_h), interpolation=cv2.INTER_AREA)

                                if resized.shape[2] == 4:
                                    # Alpha compositing
                                    alpha = (resized[:, :, 3] / 255.0) * float(action_plan.params.get("opacity", 1.0))
                                    for c in range(3):
                                        frame[y1:y2, x1:x2, c] = (
                                            alpha * resized[:, :, c] + (1.0 - alpha) * frame[y1:y2, x1:x2, c]
                                        )
                                else:
                                    frame[y1:y2, x1:x2] = resized[:, :, :3]

                    elif act_type == ActionType.REPLACE_TEXT.value:
                        text = action_plan.replacement_text
                        if text:
                            # 1. Blur or cover background region
                            roi = frame[y1:y2, x1:x2]
                            if roi.size > 0:
                                blurred = cv2.GaussianBlur(roi, (25, 25), 0)
                                frame[y1:y2, x1:x2] = blurred

                            # 2. Render text with PIL for Unicode Vietnamese support
                            pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                            draw = ImageDraw.Draw(pil_img)

                            box_h = y2 - y1
                            font_size = max(16, int(box_h * 0.6))
                            try:
                                font = ImageFont.truetype(self.font_path or "arial.ttf", font_size)
                            except OSError:
                                font = ImageFont.load_default()

                            # Compute text box position
                            bbox_text = draw.textbbox((0, 0), text, font=font)
                            text_w = bbox_text[2] - bbox_text[0]
                            text_h = bbox_text[3] - bbox_text[1]

                            tx = x1 + max(0, (x2 - x1 - text_w) // 2)
                            ty = y1 + max(0, (y2 - y1 - text_h) // 2)

                            # Outline
                            stroke_w = max(1, font_size // 12)
                            draw.text((tx, ty), text, font=font, fill=(255, 255, 255), stroke_width=stroke_w, stroke_fill=(0, 0, 0))

                            frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

            writer.write(frame)
            frame_idx += 1

        cap.release()
        writer.release()

        # Audio Muxing via FFmpeg
        cmd = [
            "ffmpeg",
            "-y",
            "-i", str(temp_video),
            "-i", str(input_video_path),
            "-c:v", "copy",
            "-c:a", "copy",
            "-map", "0:v:0",
            "-map", "1:a:0?",
            str(output_path),
        ]

        if ffmpeg_encoder == "nvenc" or (ffmpeg_encoder == "auto" and cv2.cuda.getCudaEnabledDeviceCount() > 0):
            # Fast re-encode with NVENC if requested/available
            cmd = [
                "ffmpeg",
                "-y",
                "-i", str(temp_video),
                "-i", str(input_video_path),
                "-c:v", "h264_nvenc",
                "-preset", nvenc_preset,
                "-cq", str(nvenc_cq),
                "-c:a", "copy",
                "-map", "0:v:0",
                "-map", "1:a:0?",
                str(output_path),
            ]

        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as exc:
            # Fallback to simple stream copy / CPU ffmpeg
            cmd_fallback = [
                "ffmpeg",
                "-y",
                "-i", str(temp_video),
                "-i", str(input_video_path),
                "-c:v", "libx264",
                "-preset", "fast",
                "-c:a", "copy",
                "-map", "0:v:0",
                "-map", "1:a:0?",
                str(output_path),
            ]
            subprocess.run(cmd_fallback, capture_output=True, text=True, check=True)

        # Clean temp directory
        try:
            shutil.rmtree(temp_dir)
        except OSError:
            pass

        return str(output_path)
