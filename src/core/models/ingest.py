"""Ingest domain models: where a video comes from, and what we got.

Ingest is the step before everything else — turning a URL (or a list of them)
into a local file the pipeline can process. These models describe the request
and its outcome; they carry no downloader-specific types.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

#: Supported source families. Kept as a closed set so routing is explicit
#: rather than "whatever the extractor guesses".
SOURCE_PLATFORMS = {
    "youtube",
    "tiktok",
    "douyin",
    "bilibili",
    "instagram",
    "facebook",
    "twitter",
    "generic",
}

#: Quality ceilings. A re-upload never benefits from a 4K master: the source
#: is usually already compressed, and bigger files only slow every later stage.
QUALITY_HEIGHTS = {"best": None, "1080": 1080, "720": 720, "480": 480}


@dataclass
class SourceSpec:
    """One thing to ingest."""

    url: str
    #: Optional override; otherwise inferred from the URL host.
    platform: str = ""
    quality: str = "1080"
    #: Download subtitles when the platform offers them (cheap, and better than
    #: OCR for sources that ship real captions).
    fetch_subtitles: bool = False
    #: Trim to the first N seconds after download (0 = keep whole).
    trim_seconds: float = 0.0
    #: Free-form label recorded in the manifest for the operator.
    note: str = ""

    def __post_init__(self) -> None:
        if not self.url or not self.url.strip():
            raise ValueError("SourceSpec.url must not be empty")
        if self.quality not in QUALITY_HEIGHTS:
            allowed = ", ".join(sorted(QUALITY_HEIGHTS))
            raise ValueError(f"SourceSpec.quality must be one of: {allowed}")
        if float(self.trim_seconds) < 0:
            raise ValueError("SourceSpec.trim_seconds must not be negative")

    @property
    def max_height(self) -> Optional[int]:
        return QUALITY_HEIGHTS[self.quality]

    def to_dict(self) -> Dict[str, object]:
        return {
            "url": self.url,
            "platform": self.platform or infer_platform(self.url),
            "quality": self.quality,
            "fetch_subtitles": bool(self.fetch_subtitles),
            "trim_seconds": float(self.trim_seconds),
            "note": self.note,
        }


@dataclass
class IngestResult:
    """Outcome of ingesting one source."""

    source: SourceSpec
    status: str  # OK | FAILED | SKIPPED
    video_path: Optional[str] = None
    subtitle_path: Optional[str] = None
    title: str = ""
    duration_seconds: float = 0.0
    width: int = 0
    height: int = 0
    #: Stable identity of the downloaded media, used to avoid re-ingesting the
    #: same source twice.
    content_id: str = ""
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.status == "OK" and bool(self.video_path)

    def to_dict(self) -> Dict[str, object]:
        return {
            "url": self.source.url,
            "platform": self.source.platform or infer_platform(self.source.url),
            "status": self.status,
            "video_path": self.video_path,
            "subtitle_path": self.subtitle_path,
            "title": self.title,
            "duration_seconds": round(float(self.duration_seconds), 3),
            "width": int(self.width),
            "height": int(self.height),
            "content_id": self.content_id,
            "error": self.error,
            "note": self.source.note,
        }


@dataclass
class IngestBatch:
    """All results from one ingest run."""

    results: List[IngestResult] = field(default_factory=list)

    def add(self, result: IngestResult) -> None:
        self.results.append(result)

    @property
    def ok_count(self) -> int:
        return sum(1 for r in self.results if r.ok)

    @property
    def failed_count(self) -> int:
        return sum(1 for r in self.results if r.status == "FAILED")

    def to_dict(self) -> Dict[str, object]:
        return {
            "total": len(self.results),
            "ok": self.ok_count,
            "failed": self.failed_count,
            "results": [r.to_dict() for r in self.results],
        }


def infer_platform(url: str) -> str:
    """Map a URL host onto a known platform name.

    Explicit and small on purpose: guessing from page content is how ingest
    tools end up downloading the wrong thing.
    """
    lowered = url.lower()
    hosts = (
        ("youtube.com", "youtube"),
        ("youtu.be", "youtube"),
        ("tiktok.com", "tiktok"),
        ("douyin.com", "douyin"),
        ("bilibili.com", "bilibili"),
        ("instagram.com", "instagram"),
        ("facebook.com", "facebook"),
        ("fb.watch", "facebook"),
        ("twitter.com", "twitter"),
        ("x.com", "twitter"),
    )
    for needle, platform in hosts:
        if needle in lowered:
            return platform
    return "generic"
