from src.application.ports.video_provider import VideoProvider


class GetVideoDurationUseCase:
    def __init__(self, video_provider: VideoProvider):
        self.video_provider = video_provider

    def execute(self, video_path: str) -> float:
        return self.video_provider.get_duration(video_path)
