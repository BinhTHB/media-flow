import os
from src.application.ports.video_provider import VideoProvider

class MockVideoProvider(VideoProvider):
    def get_dimensions(self, video_path: str) -> tuple[int, int]:
        return 1920, 1080

    def get_duration(self, video_path: str) -> float:
        return 30.0

    def extract_audio(self, video_path: str, output_path: str | None = None) -> str:
        if output_path:
            audio_path = output_path
        else:
            base, _ = os.path.splitext(video_path)
            audio_path = f"{base}_extracted.wav"
        
        dirname = os.path.dirname(audio_path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
            
        with open(audio_path, "w", encoding="utf-8") as f:
            f.write("mock audio")
        return audio_path

    def render_video(
        self,
        video_path: str,
        subtitle_path: str,
        output_path: str | None = None,
        style: dict | None = None
    ) -> str:
        if output_path:
            render_path = output_path
        else:
            base, ext = os.path.splitext(video_path)
            render_path = f"{base}_translated{ext}"
            
        dirname = os.path.dirname(render_path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
            
        with open(render_path, "w", encoding="utf-8") as f:
            f.write("mock rendered video")
        return render_path
