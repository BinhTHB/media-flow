from dataclasses import dataclass, field
from typing import List
from src.domain.entities.feature_config import FeatureConfig
from src.domain.entities.job_status import JobStatus
from src.domain.entities.segment import Segment
from src.domain.entities.tts_config import TTSConfig
from src.domain.entities.stage_control_config import StageControlConfig
from src.shared.exceptions import ValidationError

@dataclass
class VideoJob:
    job_id: str
    input_video: str
    source_language: str = "auto"
    speech_language: str = "auto"
    target_language: str = "vi"
    features: FeatureConfig = field(default_factory=FeatureConfig)
    segments: List[Segment] = field(default_factory=list)
    audio_path: str = ""
    output_video_path: str = ""
    subtitle_path: str = ""
    subtitle_content: str = ""
    status: JobStatus = JobStatus.CREATED
    error_message: str = ""
    started_at: float = 0.0
    completed_at: float = 0.0
    tts_config: TTSConfig = field(default_factory=TTSConfig)
    dubbed_audio_path: str = ""
    dubbed_video_path: str = ""
    final_timeline_id: str = ""
    final_render_manifest_path: str = ""
    stage_controls: StageControlConfig = field(default_factory=StageControlConfig)

    def __post_init__(self):
        if not self.job_id:
            raise ValidationError("VideoJob.job_id cannot be empty.")
        if not self.input_video:
            raise ValidationError("VideoJob.input_video cannot be empty.")
        if self.source_language not in ("zh", "en", "auto"):
            raise ValidationError(f"VideoJob.source_language must be 'zh', 'en', or 'auto', got '{self.source_language}'.")
        if self.speech_language not in ("zh", "en", "vi", "auto"):
            raise ValidationError(
                "VideoJob.speech_language must be 'zh', 'en', 'vi', or 'auto', "
                f"got '{self.speech_language}'."
            )
        if self.target_language != "vi":
            raise ValidationError(f"VideoJob.target_language must be 'vi', got '{self.target_language}'.")
        if isinstance(self.status, str):
            self.status = JobStatus(self.status)

    def to_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "source_language": self.source_language,
            "speech_language": self.speech_language,
            "target_language": self.target_language,
            "input_video": self.input_video,
            "features": self.features.to_dict(),
            "segments": [s.to_dict() for s in self.segments],
            "status": self.status.value,
            "error_message": self.error_message,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "tts_config": self.tts_config.__dict__,
            "dubbed_audio_path": self.dubbed_audio_path,
            "dubbed_video_path": self.dubbed_video_path,
            "final_timeline_id": self.final_timeline_id,
            "final_render_manifest_path": self.final_render_manifest_path,
            "stage_controls": self.stage_controls.to_dict(),
        }
