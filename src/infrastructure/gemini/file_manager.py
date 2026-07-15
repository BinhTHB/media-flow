from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.infrastructure.gemini.contracts import GeminiFileClient
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
    """No-op-safe file manager scaffold.

    Without an injected client (or when disabled) it returns local-safe handles
    and performs no network I/O. This keeps OCR/ASR file flows safe before a real
    Gemini File API client is wired in.
    """

    def __init__(self, client: Optional[GeminiFileClient] = None, *, enabled: bool = False) -> None:
        self._client = None
        self._enabled = False

    @property
    def supports_inline(self) -> bool:
        return True

    @property
    def supports_remote_upload(self) -> bool:
        return False

    @property
    def enabled(self) -> bool:
        return self._enabled

    def upload_file(self, local_path: str, *, mime_type: Optional[str] = None) -> GeminiFileHandle:
        if not self.supports_remote_upload:
            return GeminiFileHandle(
                name=os.path.basename(local_path),
                uri="",
                mime_type=mime_type,
                size_bytes=self._safe_size(local_path),
                uploaded=False,
            )
        try:
            result = self._client.upload_file(local_path, mime_type=mime_type)
        except Exception as exc:
            raise GeminiFileError(f"File upload failed for '{local_path}': {exc}") from exc
        return GeminiFileHandle(
            name=result.get("name", os.path.basename(local_path)),
            uri=result.get("uri", ""),
            mime_type=result.get("mimeType", mime_type),
            size_bytes=int(result.get("sizeBytes", 0) or 0),
            uploaded=True,
            metadata=result,
        )

    def delete_file(self, file_name: str) -> None:
        if not self.supports_remote_upload:
            return
        try:
            self._client.delete_file(file_name)
        except Exception as exc:
            raise GeminiFileError(f"File delete failed for '{file_name}': {exc}") from exc

    def list_files(self) -> List[GeminiFileHandle]:
        if not self._enabled:
            return []
        return []

    @staticmethod
    def _safe_size(path: str) -> int:
        try:
            return os.path.getsize(path)
        except OSError:
            return 0
