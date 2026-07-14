import subprocess
import os
from typing import List
from src.application.ports.frame_extraction_provider import FrameExtractionProvider
from src.domain.entities.frame import Frame
from src.shared.exceptions import ProviderError

class FFmpegFrameExtractionProvider(FrameExtractionProvider):
    def _check_ffmpeg(self):
        try:
            subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        except (FileNotFoundError, subprocess.CalledProcessError) as e:
            raise ProviderError("FFmpeg is not installed or not found in PATH.") from e

    def extract_frames(self, video_path: str, interval: float, output_dir: str) -> List[Frame]:
        if not os.path.exists(video_path):
            raise ProviderError(f"Input video file not found: {video_path}")
        self._check_ffmpeg()
        os.makedirs(output_dir, exist_ok=True)
        
        pattern = os.path.join(output_dir, "frame_%04d.jpg")
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
        frames = []
        for i, f in enumerate(files):
            timestamp = (float(i) + 0.5) * interval
            frames.append(Frame(timestamp=timestamp, image_path=os.path.join(output_dir, f)))
        return frames
