from abc import ABC, abstractmethod
from typing import List


class DubbingProvider(ABC):
    @abstractmethod
    def merge_timeline(
        self,
        clips: List[dict],
        output_path: str,
        duration: float,
    ) -> str:
        pass

    @abstractmethod
    def render_dubbed_video(
        self,
        video_path: str,
        dubbed_audio_path: str,
        output_path: str,
        original_audio_mode: str,
        original_audio_volume: float,
    ) -> str:
        pass
