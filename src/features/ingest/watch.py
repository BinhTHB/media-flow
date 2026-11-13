"""Turn a source list into the video URLs a poll should download.

A line names a channel, a playlist, or one video. Optional ``visual`` and
``audio`` flags say which reup stages to run after the download. Listing and
downloading stay outside this module: it only parses the file and drops URLs
the manifest already holds.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

from src.core.models.ingest import SourceSpec

_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}


@dataclass(frozen=True)
class WatchSource:
    """One line of the source file."""

    url: str
    visual: Optional[bool] = None
    audio: Optional[bool] = None


def parse_watch_text(text: str) -> list[WatchSource]:
    """Parse a source file. Blank lines and ``#`` comments are ignored."""
    sources: list[WatchSource] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        sources.append(_parse_line(line))
    return sources


def plan_downloads(
    entries: Iterable[str],
    known_ok: Iterable[str],
    *,
    quality: str = "1080",
    fetch_subtitles: bool = False,
    trim_seconds: float = 0.0,
    note: str = "",
) -> list[SourceSpec]:
    """Keep listed video URLs that are not already downloaded."""
    done = set(known_ok)
    planned: list[SourceSpec] = []
    seen: set[str] = set()
    for url in entries:
        if not url or url in done or url in seen:
            continue
        seen.add(url)
        planned.append(
            SourceSpec(
                url=url,
                quality=quality,
                fetch_subtitles=fetch_subtitles,
                trim_seconds=trim_seconds,
                note=note,
            )
        )
    return planned


def _parse_line(line: str) -> WatchSource:
    tokens = line.split()
    url = tokens[0]
    visual: Optional[bool] = None
    audio: Optional[bool] = None
    for token in tokens[1:]:
        if "=" not in token:
            raise ValueError(f"Nguồn không hợp lệ: {token}")
        key, value = token.split("=", 1)
        flag = _flag(value)
        if key == "visual":
            visual = flag
        elif key == "audio":
            audio = flag
        else:
            raise ValueError(f"Nguồn không hợp lệ: {token}")
    if not url.startswith(("http://", "https://")):
        raise ValueError(f"Nguồn không hợp lệ: {url}")
    return WatchSource(url=url, visual=visual, audio=audio)


def _flag(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered in _TRUE:
        return True
    if lowered in _FALSE:
        return False
    raise ValueError(f"Nguồn không hợp lệ: {value}")
