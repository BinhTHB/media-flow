from abc import ABC, abstractmethod

class VideoProvider(ABC):
    @abstractmethod
    def get_dimensions(self, video_path: str) -> tuple[int, int]:
        pass

    @abstractmethod
    def get_duration(self, video_path: str) -> float:
        pass

    @abstractmethod
    def extract_audio(self, video_path: str, output_path: str | None = None) -> str:
        pass

    @abstractmethod
    def render_video(
        self,
        video_path: str,
        subtitle_path: str,
        output_path: str | None = None,
        style: dict | None = None
    ) -> str:
        pass
