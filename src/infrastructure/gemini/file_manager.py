from __future__ import annotations

import os
import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.infrastructure.gemini.contracts import GEMINI_API_BASE, GEMINI_API_VERSION, GeminiFileClient
from src.infrastructure.gemini.exceptions import GeminiFileError


@dataclass
class GeminiFileHandle:
    name: str
    uri: str = ""
    mime_type: Optional[str] = None
    size_bytes: int = 0
    uploaded: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


class FileManager:
    """Shared Files API manager with content-addressed in-process upload reuse."""

    def __init__(
        self,
        client: Optional[GeminiFileClient] = None,
        *,
        enabled: bool = False,
        poll_interval_seconds: float = 2.0,
        poll_timeout_seconds: float = 180.0,
    ) -> None:
        self._client = client
        self._enabled = enabled
        self._poll_interval_seconds = poll_interval_seconds
        self._poll_timeout_seconds = poll_timeout_seconds
        self._by_sha256: Dict[tuple[str, int, str], GeminiFileHandle] = {}
        self.upload_count = 0
        self.reuse_count = 0

    @property
    def supports_inline(self) -> bool:
        return True

    @property
    def supports_remote_upload(self) -> bool:
        return self._enabled and self._client is not None

    @property
    def enabled(self) -> bool:
        return self._enabled

    def upload_file(self, local_path: str, *, mime_type: Optional[str] = None) -> GeminiFileHandle:
        if not self.supports_remote_upload:
            raise GeminiFileError("Gemini Files API is not enabled.")
        digest = self._sha256(local_path)
        cache_key = (digest, self._safe_size(local_path), mime_type or "application/octet-stream")
        existing = self._by_sha256.get(cache_key)
        if existing is not None:
            verified = self._verify_remote(existing)
            self._by_sha256[cache_key] = verified
            self.reuse_count += 1
            verified.metadata["reuse_count"] = int(verified.metadata.get("reuse_count", 0)) + 1
            return verified
        try:
            result = self._client.upload_file(local_path, mime_type=mime_type)
        except Exception as exc:
            raise GeminiFileError(f"File upload failed for '{local_path}': {exc}") from exc
        metadata = result.get("file", result)
        if not isinstance(metadata, dict) or not metadata.get("name"):
            raise GeminiFileError("Gemini Files API upload response omitted file name.")
        metadata = self._poll_until_active(metadata)
        self.upload_count += 1
        handle = GeminiFileHandle(
            name=str(metadata["name"]),
            uri=str(metadata.get("uri", "")),
            mime_type=metadata.get("mimeType", mime_type),
            size_bytes=int(metadata.get("sizeBytes", self._safe_size(local_path)) or 0),
            uploaded=True,
            metadata={
                "sha256": digest,
                "size_bytes": int(metadata.get("sizeBytes", self._safe_size(local_path)) or 0),
                "mime_type": metadata.get("mimeType", mime_type),
                "upload_timestamp": time.time(),
                "state": metadata.get("state", "ACTIVE"),
                "reuse_count": 0,
            },
        )
        self._by_sha256[cache_key] = handle
        return handle

    def delete_file(self, file_name: str) -> None:
        if not self.supports_remote_upload:
            return
        try:
            self._client.delete_file(file_name)
        except Exception as exc:
            raise GeminiFileError(f"File delete failed for '{file_name}': {exc}") from exc
        self._by_sha256 = {
            key: handle for key, handle in self._by_sha256.items()
            if handle.name != file_name
        }

    def list_files(self) -> List[GeminiFileHandle]:
        return list(self._by_sha256.values())

    def _verify_remote(self, handle: GeminiFileHandle) -> GeminiFileHandle:
        if self._client is None:
            raise GeminiFileError("Gemini Files API client is missing for remote verification.")
        try:
            result = self._client.get_file(handle.name)
        except Exception as exc:
            raise GeminiFileError(f"Remote file verification failed for '{handle.name}': {exc}") from exc
        metadata = result.get("file", result)
        if not isinstance(metadata, dict):
            raise GeminiFileError(f"Remote file verification response invalid for '{handle.name}'.")
        state = str(metadata.get("state", "ACTIVE") or "ACTIVE").upper()
        if state == "FAILED":
            raise GeminiFileError(f"Remote file '{handle.name}' has failed state on verification.")
        if state != "ACTIVE":
            metadata = self._poll_until_active(metadata)
        handle.metadata["state"] = metadata.get("state", "ACTIVE")
        handle.uri = str(metadata.get("uri", handle.uri))
        return handle

    def _poll_until_active(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        if self._client is None:
            raise GeminiFileError("Gemini Files API client is missing.")
        name = str(metadata["name"])
        deadline = time.monotonic() + self._poll_timeout_seconds
        while True:
            state = str(metadata.get("state", "ACTIVE") or "ACTIVE").upper()
            if state == "ACTIVE":
                return metadata
            if state == "FAILED":
                raise GeminiFileError(f"Gemini remote file '{name}' processing failed.")
            if time.monotonic() >= deadline:
                raise GeminiFileError(f"Timed out waiting for Gemini remote file '{name}'.")
            try:
                result = self._client.get_file(name)
            except Exception as exc:
                raise GeminiFileError(f"Could not poll Gemini remote file '{name}': {exc}") from exc
            metadata = result.get("file", result)
            if not isinstance(metadata, dict):
                raise GeminiFileError("Gemini Files API poll response was invalid.")
            time.sleep(self._poll_interval_seconds)

    @staticmethod
    def _sha256(path: str) -> str:
        try:
            digest = hashlib.sha256()
            with open(path, "rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            return digest.hexdigest()
        except OSError as exc:
            raise GeminiFileError(f"Could not hash '{path}': {exc}") from exc

    @staticmethod
    def _safe_size(path: str) -> int:
        try:
            return os.path.getsize(path)
        except OSError:
            return 0
