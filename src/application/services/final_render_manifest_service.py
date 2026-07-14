import hashlib
import os
from datetime import datetime, timezone
from typing import Dict

from src.shared.exceptions import ValidationError


class FinalRenderManifestService:
    REQUIRED_ARTIFACTS = (
        "subtitle_srt",
        "subtitle_ass",
        "placement_quality_report",
        "rendered_video",
        "tts_manifest",
        "dubbed_audio",
        "dubbed_video",
    )

    def build(self, job_id: str, timeline_id: str, artifacts: Dict[str, str]) -> dict:
        missing = [key for key in self.REQUIRED_ARTIFACTS if key not in artifacts]
        if missing:
            raise ValidationError(f"Final render artifacts missing: {', '.join(missing)}")
        entries = {
            key: self._artifact_entry(path, timeline_id)
            for key, path in artifacts.items()
        }
        return {
            "schema_version": 1,
            "job_id": job_id,
            "timeline_id": timeline_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "artifacts": entries,
        }

    def validate(self, manifest: dict, timeline_id: str) -> None:
        if manifest.get("timeline_id") != timeline_id:
            raise ValidationError("Final render manifest timeline ID mismatch.")
        artifacts = manifest.get("artifacts", {})
        for key in self.REQUIRED_ARTIFACTS:
            entry = artifacts.get(key)
            if not entry or entry.get("timeline_id") != timeline_id:
                raise ValidationError(f"Final render artifact is stale or missing: {key}")
            path = entry["path"]
            if not os.path.exists(path) or os.path.getsize(path) == 0:
                raise ValidationError(f"Final render artifact is missing or empty: {key}")
            if self._sha256(path) != entry["sha256"]:
                raise ValidationError(f"Final render artifact changed after manifest: {key}")

    def _artifact_entry(self, path: str, timeline_id: str) -> dict:
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            raise ValidationError(f"Final render artifact is missing or empty: {path}")
        stat = os.stat(path)
        return {
            "path": path,
            "timeline_id": timeline_id,
            "size": stat.st_size,
            "modified_ns": stat.st_mtime_ns,
            "sha256": self._sha256(path),
        }

    @staticmethod
    def _sha256(path: str) -> str:
        digest = hashlib.sha256()
        with open(path, "rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
