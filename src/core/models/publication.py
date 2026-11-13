"""Publication domain entities.

One finished video (``PublishItem``) is published to one platform per
attempt; every attempt lands in the state store as a ``PublishRecord``.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

#: YouTube "Entertainment" category — neutral default for re-uploads.
DEFAULT_CATEGORY_ID = "24"

#: Safe default: automation must not publish publicly without explicit config.
DEFAULT_PRIVACY = "private"


def item_id_for(video_path: str) -> str:
    """Stable identity for one source file: sha256 of path + size + mtime.

    Content hashing multi-GB videos on every scan would dominate the poll
    loop; path+size+mtime is stable enough for dedupe across worker runs.
    """
    p = Path(video_path)
    stat = p.stat()
    digest = hashlib.sha256(
        f"{str(p.resolve()).lower()}|{stat.st_size}|{int(stat.st_mtime)}".encode("utf-8")
    ).hexdigest()
    return digest[:16]


def fallback_metadata(video_path: str, transcript: str = "") -> "VideoMetadata":
    """Filename-derived metadata: the safe default when no generator runs.

    A spoken transcript, when present, is a better title/description than the
    output filename (``DDapV_Tz3RY-reup-translate-dub``).
    """
    stem = Path(video_path).stem
    lines = [part.strip() for part in (transcript or "").split("|") if part.strip()]
    title = (lines[0] if lines else stem)[:100]
    description = (" ".join(lines) if lines else stem)[:5000]
    tags = ["vietnam"] if lines else []
    return VideoMetadata(title=title, description=description, tags=tags)


@dataclass
class VideoMetadata:
    title: str
    description: str = ""
    tags: List[str] = field(default_factory=list)
    category_id: str = DEFAULT_CATEGORY_ID
    privacy: str = DEFAULT_PRIVACY
    language: str = "vi"

    def validate(self) -> None:
        if not self.title or not self.title.strip():
            raise ValueError("VideoMetadata.title must not be empty")
        if self.privacy not in {"public", "unlisted", "private"}:
            raise ValueError(f"VideoMetadata.privacy invalid: {self.privacy!r}")
        if len(self.title) > 100:
            raise ValueError("VideoMetadata.title exceeds 100 characters")


@dataclass
class PublishItem:
    """One video queued for publication to one platform."""

    item_id: str
    video_path: str
    platform: str
    metadata: VideoMetadata


@dataclass
class PublishRecord:
    """Outcome of publishing one item to one platform."""

    item_id: str
    video_path: str
    platform: str
    status: str  # PUBLISHED | FAILED
    platform_video_id: Optional[str] = None
    url: Optional[str] = None
    error: Optional[str] = None
    attempts: int = 1
    published_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )
