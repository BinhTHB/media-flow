import os
from typing import Sequence

from src.application.ports.artifact_provider import ArtifactProvider
from src.application.ports.dubbing_provider import DubbingProvider
from src.application.ports.tts_provider import TTSProvider
from src.domain.entities.tts_config import TTSConfig
from src.shared.exceptions import ProviderError


class GenerateDubbedAudioUseCase:
    def __init__(
        self,
        tts_provider: TTSProvider,
        dubbing_provider: DubbingProvider,
        artifact_provider: ArtifactProvider,
    ) -> None:
        self.tts_provider = tts_provider
        self.dubbing_provider = dubbing_provider
        self.artifact_provider = artifact_provider

    def execute(
        self,
        job_id: str,
        segments: Sequence[object],
        duration: float,
        config: TTSConfig,
        timeline_id: str = "",
    ) -> str:
        manifest = []
        if not segments:
            raise ProviderError("Narration timeline is empty.")
        segment_dir = self.artifact_provider.tts_segments_dir(job_id)
        for index, segment in enumerate(segments, start=1):
            segment_id = getattr(segment, "segment_id", getattr(segment, "id", ""))
            text = (
                getattr(segment, "translated_text", "") or getattr(segment, "text", "")
            ).strip()
            if not text:
                raise ProviderError(f"Narration segment has empty text: {segment_id}")
            audio_path = os.path.join(segment_dir, f"{index:04d}_{segment_id}.mp3")
            try:
                self.tts_provider.synthesize(text, audio_path, config.voice, config.rate)
            except ProviderError as exc:
                manifest.append(
                    {"id": segment_id, "start": segment.start, "end": segment.end,
                     "text": text, "status": "failed", "error": str(exc)}
                )
                self.artifact_provider.write_json(
                    self.artifact_provider.tts_manifest_path(job_id),
                    {"timeline_id": timeline_id, "segments": manifest},
                )
                raise ProviderError(f"TTS segment failed: {segment_id}") from exc
            if not os.path.exists(audio_path) or os.path.getsize(audio_path) == 0:
                manifest.append(
                    {"id": segment_id, "start": segment.start, "end": segment.end,
                     "text": text, "status": "failed", "error": "missing_or_empty_audio"}
                )
                self.artifact_provider.write_json(
                    self.artifact_provider.tts_manifest_path(job_id),
                    {"timeline_id": timeline_id, "segments": manifest},
                )
                raise ProviderError(f"TTS segment is missing or empty: {segment_id}")
            manifest.append(
                {"id": segment_id, "start": segment.start, "end": segment.end,
                 "text": text, "audio_path": audio_path, "status": "ready"}
            )
        self.artifact_provider.write_json(
            self.artifact_provider.tts_manifest_path(job_id),
            {"timeline_id": timeline_id, "segments": manifest},
        )
        clips = [entry for entry in manifest if entry["status"] == "ready"]
        if not clips:
            raise ProviderError("All TTS segment syntheses failed.")
        return self.dubbing_provider.merge_timeline(
            clips, self.artifact_provider.dubbed_audio_path(job_id), duration
        )
