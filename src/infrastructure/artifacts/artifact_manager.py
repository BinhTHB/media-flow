import json
import os
from typing import Any
from src.application.ports.artifact_provider import ArtifactProvider
from src.shared.exceptions import ProviderError

class ArtifactManager(ArtifactProvider):
    def __init__(self, base_dir: str = "output"):
        self.base_dir = base_dir

    def job_dir(self, job_id: str) -> str:
        path = os.path.join(self.base_dir, job_id)
        os.makedirs(path, exist_ok=True)
        return path

    def _path(self, job_id: str, *parts: str) -> str:
        path = os.path.join(self.job_dir(job_id), *parts)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        return path

    def audio_path(self, job_id: str) -> str:
        return self._path(job_id, "audio", "audio.wav")

    def transcript_path(self, job_id: str) -> str:
        return self._path(job_id, "transcripts", "transcript.json")

    def translated_path(self, job_id: str) -> str:
        return self._path(job_id, "transcripts", "translated.json")

    def normalized_path(self, job_id: str) -> str:
        return self._path(job_id, "transcripts", "normalized.json")

    def segmented_path(self, job_id: str) -> str:
        return self._path(job_id, "transcripts", "segmented.json")

    def subtitle_path(self, job_id: str) -> str:
        return self._path(job_id, "subtitles", "subtitle.srt")

    def subtitle_ass_path(self, job_id: str) -> str:
        return self._path(job_id, "subtitles", "subtitle.ass")

    def video_path(self, job_id: str) -> str:
        return self.video_output_path(job_id)

    def video_output_path(self, job_id: str) -> str:
        return self._path(job_id, "videos", "output.mp4")

    def tts_segments_dir(self, job_id: str) -> str:
        path = os.path.join(self.job_dir(job_id), "tts_segments")
        os.makedirs(path, exist_ok=True)
        return path

    def tts_manifest_path(self, job_id: str) -> str:
        return self._path(job_id, "tts_manifest.json")

    def dubbed_audio_path(self, job_id: str) -> str:
        return self._path(job_id, "audio", "dubbed_audio.wav")

    def dubbed_video_path(self, job_id: str) -> str:
        return self._path(job_id, "videos", "output_dubbed.mp4")

    def final_timeline_path(self, job_id: str) -> str:
        return self._path(job_id, "debug", "final_timeline.json")

    def final_render_manifest_path(self, job_id: str) -> str:
        return self._path(job_id, "debug", "final_render_manifest.json")

    def placement_quality_report_path(self, job_id: str) -> str:
        return self._path(job_id, "debug", "placement_quality_report.json")

    def precise_ocr_boxes_path(self, job_id: str) -> str:
        return self._path(job_id, "debug", "precise_subtitle_localization.json")

    def adaptive_subtitle_region_report_path(self, job_id: str) -> str:
        return self._path(job_id, "debug", "adaptive_subtitle_region_report.json")

    def log_path(self, job_id: str) -> str:
        return self._path(job_id, "logs", "job.log")

    def frames_dir(self, job_id: str) -> str:
        path = os.path.join(self.job_dir(job_id), "frames")
        os.makedirs(path, exist_ok=True)
        return path

    def debug_dir(self, job_id: str) -> str:
        path = os.path.join(self.job_dir(job_id), "debug")
        os.makedirs(path, exist_ok=True)
        return path

    def debug_before_path(self, job_id: str) -> str:
        return os.path.join(self.debug_dir(job_id), "before_render.jpg")

    def debug_overlay_path(self, job_id: str) -> str:
        return os.path.join(self.debug_dir(job_id), "overlay_debug.jpg")

    def debug_after_path(self, job_id: str) -> str:
        return os.path.join(self.debug_dir(job_id), "after_render.jpg")

    def debug_regions_json_path(self, job_id: str) -> str:
        return os.path.join(self.debug_dir(job_id), "subtitle_regions.json")

    def debug_placements_json_path(self, job_id: str) -> str:
        return os.path.join(self.debug_dir(job_id), "subtitle_placements.json")

    def debug_metadata_json_path(self, job_id: str) -> str:
        return os.path.join(self.debug_dir(job_id), "debug_metadata.json")

    def _guard_overwrite(self, path: str, overwrite: bool) -> None:
        if os.path.exists(path) and not overwrite:
            raise ProviderError(f"Artifact already exists: {path}")

    def write_json(self, path: str, data: Any, overwrite: bool = True) -> None:
        self._guard_overwrite(path, overwrite)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def read_json(self, path: str) -> Any:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def write_text(self, path: str, content: str, overwrite: bool = True) -> None:
        self._guard_overwrite(path, overwrite)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
