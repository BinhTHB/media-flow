from typing import Any
from src.application.ports.video_provider import VideoProvider

class RenderVideoUseCase:
    def __init__(self, video_provider: VideoProvider):
        self.video_provider = video_provider

    def execute(
        self,
        video_path: str,
        subtitle_path: str,
        output_path: str | None = None,
        style: dict[str, Any] | None = None,
    ) -> str:
        return self.video_provider.render_video(
            video_path=video_path,
            subtitle_path=subtitle_path,
            output_path=output_path,
            style=style,
        )
