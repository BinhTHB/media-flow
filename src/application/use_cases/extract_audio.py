from src.application.ports.video_provider import VideoProvider

class ExtractAudioUseCase:
    def __init__(self, video_provider: VideoProvider):
        self.video_provider = video_provider

    def execute(self, video_path: str) -> str:
        return self.video_provider.extract_audio(video_path)
