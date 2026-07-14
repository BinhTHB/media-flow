from src.application.ports.artifact_provider import ArtifactProvider
from src.application.ports.dubbing_provider import DubbingProvider
from src.domain.entities.tts_config import TTSConfig


class MuxDubbedVideoUseCase:
    def __init__(
        self,
        dubbing_provider: DubbingProvider,
        artifact_provider: ArtifactProvider,
    ) -> None:
        self.dubbing_provider = dubbing_provider
        self.artifact_provider = artifact_provider

    def execute(
        self,
        job_id: str,
        video_path: str,
        dubbed_audio_path: str,
        config: TTSConfig,
    ) -> str:
        return self.dubbing_provider.render_dubbed_video(
            video_path,
            dubbed_audio_path,
            self.artifact_provider.dubbed_video_path(job_id),
            config.original_audio_mode,
            config.original_audio_volume,
        )
