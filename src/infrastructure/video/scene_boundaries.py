"""Small-frame visual cut detector; provides frame indices, never subtitle labels."""
import cv2
import numpy as np
from src.core.contracts import SceneBoundaryPort


class FrameDifferenceSceneDetector(SceneBoundaryPort):
    def detect_cuts(self, video_path: str) -> list[int]:
        capture = cv2.VideoCapture(video_path)
        if not capture.isOpened():
            raise RuntimeError("Cannot open source for scene analysis")
        cuts, previous, index = [], None, 0
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                small = cv2.resize(frame, (64, 36)).astype(np.float32)
                if previous is not None:
                    difference = np.abs(small - previous).mean() / 255.0
                    # Deliberately conservative: gradual fades remain one scene.
                    if difference >= 0.28:
                        cuts.append(index)
                previous = small
                index += 1
        finally:
            capture.release()
        return cuts
