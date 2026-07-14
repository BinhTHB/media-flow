import os
from typing import List
from PIL import Image
from src.application.ports.frame_extraction_provider import FrameExtractionProvider
from src.domain.entities.frame import Frame

class MockFrameExtractionProvider(FrameExtractionProvider):
    def extract_frames(self, video_path: str, interval: float, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        frames = []
        for i in range(2):
            path = os.path.join(output_dir, f"frame_{i:04d}.jpg")
            Image.new("RGB", (1920, 1080), color=(128, 128, 128)).save(
                path, format="JPEG"
            )
            frames.append(Frame(timestamp=float(i) * interval, image_path=path))
        return frames
