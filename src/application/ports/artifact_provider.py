from abc import ABC, abstractmethod
from typing import Any

class ArtifactProvider(ABC):
    @abstractmethod
    def audio_path(self, job_id: str) -> str:
        pass

    @abstractmethod
    def transcript_path(self, job_id: str) -> str:
        pass

    @abstractmethod
    def translated_path(self, job_id: str) -> str:
        pass

    @abstractmethod
    def normalized_path(self, job_id: str) -> str:
        pass

    @abstractmethod
    def segmented_path(self, job_id: str) -> str:
        pass

    @abstractmethod
    def subtitle_path(self, job_id: str) -> str:
        pass

    @abstractmethod
    def subtitle_ass_path(self, job_id: str) -> str:
        pass

    @abstractmethod
    def video_path(self, job_id: str) -> str:
        pass

    @abstractmethod
    def video_output_path(self, job_id: str) -> str:
        pass

    @abstractmethod
    def tts_segments_dir(self, job_id: str) -> str:
        pass

    @abstractmethod
    def tts_manifest_path(self, job_id: str) -> str:
        pass

    @abstractmethod
    def dubbed_audio_path(self, job_id: str) -> str:
        pass

    @abstractmethod
    def dubbed_video_path(self, job_id: str) -> str:
        pass

    @abstractmethod
    def final_timeline_path(self, job_id: str) -> str:
        pass

    @abstractmethod
    def final_render_manifest_path(self, job_id: str) -> str:
        pass

    @abstractmethod
    def placement_quality_report_path(self, job_id: str) -> str:
        pass

    @abstractmethod
    def log_path(self, job_id: str) -> str:
        pass

    @abstractmethod
    def frames_dir(self, job_id: str) -> str:
        pass

    @abstractmethod
    def debug_dir(self, job_id: str) -> str:
        pass

    @abstractmethod
    def debug_before_path(self, job_id: str) -> str:
        pass

    @abstractmethod
    def debug_overlay_path(self, job_id: str) -> str:
        pass

    @abstractmethod
    def debug_after_path(self, job_id: str) -> str:
        pass

    @abstractmethod
    def debug_regions_json_path(self, job_id: str) -> str:
        pass

    @abstractmethod
    def debug_placements_json_path(self, job_id: str) -> str:
        pass

    @abstractmethod
    def debug_metadata_json_path(self, job_id: str) -> str:
        pass

    @abstractmethod
    def write_json(self, path: str, data: Any, overwrite: bool = True) -> None:
        pass

    @abstractmethod
    def read_json(self, path: str) -> Any:
        pass

    @abstractmethod
    def write_text(self, path: str, content: str, overwrite: bool = True) -> None:
        pass
