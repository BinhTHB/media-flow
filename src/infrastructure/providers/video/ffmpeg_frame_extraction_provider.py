import os
import subprocess
from typing import List

import cv2

from src.application.ports.frame_extraction_provider import FrameExtractionProvider
from src.domain.entities.frame import Frame
from src.shared.exceptions import ProviderError


class FFmpegFrameExtractionProvider(FrameExtractionProvider):
    def _check_ffmpeg(self):
        try:
            subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        except (FileNotFoundError, subprocess.CalledProcessError) as e:
            raise ProviderError("FFmpeg is not installed or not found in PATH.") from e

    def _get_video_dimensions(self, video_path: str) -> tuple[int, int]:
        try:
            result = subprocess.run(
                ["ffprobe", "-v", "error", "-select_streams", "v:0",
                 "-show_entries", "stream=width,height",
                 "-of", "csv=s=x:p=0", video_path],
                capture_output=True, check=True, text=True, timeout=30,
            )
            width, height = result.stdout.strip().split("x")
            return int(width), int(height)
        except (ValueError, FileNotFoundError, subprocess.CalledProcessError) as exc:
            raise ProviderError(f"FFprobe could not read video dimensions: {video_path}") from exc

    def _validate_frame_files(self, output_dir: str, filenames: List[str],
                               expected_w: int, expected_h: int):
        for fname in filenames:
            path = os.path.join(output_dir, fname)
            img = cv2.imread(path)
            if img is None:
                raise ProviderError(f"Could not decode frame file: {path}")
            h, w = img.shape[:2]
            if w != expected_w or h != expected_h:
                raise ProviderError(
                    f"Frame resolution mismatch: got {w}x{h}, expected {expected_w}x{expected_h}. "
                    f"Possibly wrong video decoded. Frame: {fname}"
                )
            if float(img.std()) < 1.0:
                raise ProviderError(
                    f"Frame appears blank (std={img.std():.2f}). Video decode failed or wrong video. "
                    f"Frame: {fname}"
                )

    def extract_frames(self, video_path: str, interval: float, output_dir: str) -> List[Frame]:
        if not os.path.exists(video_path):
            raise ProviderError(f"Input video file not found: {video_path}")
        self._check_ffmpeg()

        video_w, video_h = self._get_video_dimensions(video_path)

        os.makedirs(output_dir, exist_ok=True)
        cmd = [
            "ffmpeg", "-i", video_path,
            "-vf", f"fps=1/{interval}",
            "-q:v", "2",
            "-y", os.path.join(output_dir, "frame_%04d.jpg")
        ]
        try:
            subprocess.run(cmd, capture_output=True, check=True, timeout=120)
        except subprocess.CalledProcessError as e:
            raise ProviderError(f"FFmpeg frame extraction failed: {e.stderr.decode('utf-8', errors='ignore')}") from e

        files = sorted([f for f in os.listdir(output_dir) if f.startswith("frame_") and f.endswith(".jpg")])
        if not files:
            raise ProviderError("FFmpeg produced zero frames from video")

        self._validate_frame_files(output_dir, files, video_w, video_h)

        frames = []
        for i, f in enumerate(files):
            timestamp = (float(i) + 0.5) * interval
            frames.append(Frame(timestamp=timestamp, image_path=os.path.join(output_dir, f)))
        return frames