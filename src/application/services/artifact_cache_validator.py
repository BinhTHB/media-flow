import hashlib
import json
import os
from typing import Any, Dict


class ArtifactCacheValidator:
    SCHEMA_VERSION = 1

    def signature(
        self,
        video_path: str,
        frame_interval: float,
        provider_name: str,
        provider_version: str,
        language: str,
        device: str,
        duration: float,
        resolution: str,
    ) -> Dict[str, Any]:
        frame_config = json.dumps(
            {"interval": frame_interval}, sort_keys=True, separators=(",", ":")
        )
        return {
            "schema_version": self.SCHEMA_VERSION,
            "input_video_sha256": self._sha256(video_path),
            "frame_sampling_config_sha256": hashlib.sha256(
                frame_config.encode("utf-8")
            ).hexdigest(),
            "provider_name": provider_name,
            "provider_version": provider_version,
            "language": language,
            "device": device,
            "video_duration": duration,
            "video_resolution": resolution,
        }

    @staticmethod
    def is_valid(cached: Dict[str, Any], current: Dict[str, Any]) -> bool:
        return cached == current

    @staticmethod
    def _sha256(path: str) -> str:
        if not os.path.exists(path):
            return f"missing:{os.path.abspath(path)}"
        digest = hashlib.sha256()
        with open(path, "rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
