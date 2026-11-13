"""yt-dlp implementation of :class:`DownloaderPort`.

yt-dlp is the de-facto extractor for every platform a re-upload tool cares
about (YouTube, TikTok, Douyin, Bilibili, Instagram, ...). It is imported
lazily so MediaFlow's core never depends on it and a missing install is a
clear configuration error.

Error mapping is the important part: yt-dlp raises a zoo of exception types.
Only genuine network/rate-limit failures become ``TransientIngestError``;
private, geo-blocked, or removed videos stay permanent so the batch does not
spin on them.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, Optional

from src.core.contracts import DownloaderPort, SourceCatalogPort, TransientIngestError
from src.core.models.ingest import IngestResult, SourceSpec

logger = logging.getLogger(__name__)

#: Filename template. Keeping the id in the name makes two downloads of the
#: same video distinguishable without opening the manifest.
OUTPUT_TEMPLATE = "%(id)s.%(ext)s"

#: yt-dlp error fragments that mean "try again later".
_TRANSIENT_MARKERS = (
    "timed out",
    "timeout",
    "connection reset",
    "connection aborted",
    "temporary failure",
    "http error 429",
    "http error 500",
    "http error 502",
    "http error 503",
    "http error 504",
    "too many requests",
    "rate limit",
    "unable to download webpage",
)

#: yt-dlp error fragments that are permanent for this URL.
_PERMANENT_MARKERS = (
    "private video",
    "video unavailable",
    "has been removed",
    "not available in your country",
    "geo restricted",
    "sign in to confirm",
    "age-restricted",
    "unsupported url",
    "no video formats found",
)


class YtDlpDownloader(DownloaderPort, SourceCatalogPort):
    def __init__(
        self,
        *,
        cookie_file: Optional[str] = None,
        cookies_from_browser: Optional[str] = None,
        proxy: Optional[str] = None,
        socket_timeout: int = 30,
        retries: int = 3,
        quiet: bool = True,
    ) -> None:
        self._cookie_file = cookie_file
        self._cookies_from_browser = cookies_from_browser
        self._proxy = proxy
        self._socket_timeout = int(socket_timeout)
        self._retries = int(retries)
        self._quiet = bool(quiet)

    # -- options ------------------------------------------------------------

    def _build_options(self, source: SourceSpec, output_dir: str) -> Dict[str, Any]:
        options: Dict[str, Any] = {
            "outtmpl": str(Path(output_dir) / OUTPUT_TEMPLATE),
            "format": self._format_selector(source),
            "merge_output_format": "mp4",
            "quiet": self._quiet,
            "no_warnings": self._quiet,
            "noprogress": self._quiet,
            "socket_timeout": self._socket_timeout,
            "retries": self._retries,
            # Prefer a single mp4 so later stages never remux unnecessarily.
            "prefer_ffmpeg": True,
            "overwrites": False,
        }
        if self._cookie_file:
            options["cookiefile"] = self._cookie_file
        if self._cookies_from_browser:
            options["cookiesfrombrowser"] = (self._cookies_from_browser,)
        if self._proxy:
            options["proxy"] = self._proxy
        if source.fetch_subtitles:
            options.update(
                {
                    "writesubtitles": True,
                    "writeautomaticsub": True,
                    "subtitleslangs": ["vi", "en", "zh-Hans", "zh", "all"],
                    "subtitlesformat": "srt/vtt/best",
                }
            )
        if source.trim_seconds > 0:
            # Download-time trimming avoids fetching a 40-minute source to keep
            # its first 30 seconds.
            options["download_ranges"] = _first_seconds(source.trim_seconds)
            options["force_keyframes_at_cuts"] = True
        return options

    @staticmethod
    def _format_selector(source: SourceSpec) -> str:
        max_height = source.max_height
        if max_height is None:
            return "bv*+ba/b"
        # Cap the height but still prefer the best available under the cap.
        return f"bv*[height<={max_height}]+ba/b[height<={max_height}]/b"

    # -- port ---------------------------------------------------------------

    def download(self, source: SourceSpec, output_dir: str) -> IngestResult:
        try:
            from yt_dlp import YoutubeDL
            from yt_dlp.utils import DownloadError
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "yt-dlp is required for ingest. Install it with `pip install yt-dlp`."
            ) from exc

        Path(output_dir).mkdir(parents=True, exist_ok=True)
        options = self._build_options(source, output_dir)

        logger.info("Ingest: downloading %s (%s)", source.url, source.platform or "auto")
        try:
            with YoutubeDL(options) as ydl:
                info = ydl.extract_info(source.url, download=True)
                if info is None:
                    raise RuntimeError("yt-dlp returned no media info")
                # Playlists: take the first entry rather than guessing.
                if "entries" in info:
                    entries = [e for e in (info.get("entries") or []) if e]
                    if not entries:
                        raise RuntimeError("playlist contained no downloadable entries")
                    info = entries[0]
                path = ydl.prepare_filename(info)
        except DownloadError as exc:
            raise _map_download_error(source.url, str(exc)) from exc

        resolved = _resolve_output(Path(path), Path(output_dir))
        if resolved is None:
            raise RuntimeError(f"download reported success but no file exists for {source.url}")

        return IngestResult(
            source=source,
            status="OK",
            video_path=str(resolved),
            subtitle_path=_find_subtitle(resolved),
            title=str(info.get("title") or ""),
            duration_seconds=float(info.get("duration") or 0.0),
            width=int(info.get("width") or 0),
            height=int(info.get("height") or 0),
            content_id=str(info.get("id") or ""),
        )

    def list_entries(self, url: str, *, limit: int) -> list[str]:
        """List video URLs for a channel, playlist, or single video. No download."""
        try:
            from yt_dlp import YoutubeDL
            from yt_dlp.utils import DownloadError
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "yt-dlp is required for ingest. Install it with `pip install yt-dlp`."
            ) from exc

        options: Dict[str, Any] = {
            "extract_flat": "in_playlist",
            "playlistend": max(1, int(limit)),
            "quiet": self._quiet,
            "no_warnings": self._quiet,
            "skip_download": True,
            "socket_timeout": self._socket_timeout,
        }
        if self._cookie_file:
            options["cookiefile"] = self._cookie_file
        if self._cookies_from_browser:
            options["cookiesfrombrowser"] = (self._cookies_from_browser,)
        if self._proxy:
            options["proxy"] = self._proxy
        try:
            with YoutubeDL(options) as ydl:
                info = ydl.extract_info(url, download=False)
        except DownloadError as exc:
            raise _map_download_error(url, str(exc)) from exc
        return entry_urls(info, url, max(1, int(limit)))


def entry_urls(info: Optional[Dict[str, Any]], page_url: str, limit: int) -> list[str]:
    """Pull video URLs out of a yt-dlp info dict, newest first, capped.

    A flat playlist entry often carries only an id. YouTube and Bilibili ids
    become watch URLs. Anything else is kept only when the entry already has
    an http URL, so a bare id is never guessed into the wrong site.
    """
    if not info or limit <= 0:
        return []
    entries = info.get("entries")
    if not entries:
        single = _http_url(info.get("webpage_url") or info.get("original_url") or page_url)
        return [single] if single else []
    found: list[str] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        url = _entry_url(entry, page_url)
        if url and url not in found:
            found.append(url)
        if len(found) >= limit:
            break
    return found


def _entry_url(entry: Dict[str, Any], page_url: str) -> Optional[str]:
    direct = _http_url(entry.get("webpage_url") or entry.get("url"))
    if direct:
        return direct
    video_id = str(entry.get("id") or "").strip()
    if not video_id:
        return None
    host = page_url.lower()
    if "youtube.com" in host or "youtu.be" in host:
        return f"https://www.youtube.com/watch?v={video_id}"
    if "bilibili.com" in host:
        return f"https://www.bilibili.com/video/{video_id}"
    return None


def _http_url(value: Any) -> Optional[str]:
    text = str(value or "").strip()
    if text.startswith(("http://", "https://")):
        return text
    return None


def _first_seconds(seconds: float):
    """Build the yt-dlp ``download_ranges`` callable for a head trim."""
    limit = float(seconds)

    def _range(info_dict, ydl):  # noqa: ARG001 - yt-dlp passes both
        return [{"start_time": 0.0, "end_time": limit}]

    return _range


def _resolve_output(prepared: Path, output_dir: Path) -> Optional[Path]:
    """Find the real downloaded file.

    ``prepare_filename`` guesses the extension before merging, so a merged
    download can land as ``.mkv`` or ``.mp4`` regardless of the guess.
    """
    if prepared.is_file():
        return prepared
    stem = prepared.with_suffix("")
    for candidate in sorted(output_dir.glob(f"{stem.name}.*")):
        if candidate.suffix.lower() in {".mp4", ".mkv", ".webm", ".mov", ".flv"}:
            return candidate
    return None


def _find_subtitle(video: Path) -> Optional[str]:
    for candidate in sorted(video.parent.glob(f"{video.stem}.*")):
        if candidate.suffix.lower() in {".srt", ".vtt", ".ass"}:
            return str(candidate)
    return None


def _map_download_error(url: str, message: str) -> Exception:
    """Translate a yt-dlp error string into a project-level exception."""
    lowered = message.lower()
    for marker in _PERMANENT_MARKERS:
        if marker in lowered:
            return RuntimeError(f"permanent ingest failure for {url}: {message[:300]}")
    for marker in _TRANSIENT_MARKERS:
        if marker in lowered:
            return TransientIngestError(f"transient ingest failure for {url}: {message[:300]}")
    # Unknown errors are treated as permanent: retrying an unclassified failure
    # mostly just multiplies the wait time.
    return RuntimeError(f"ingest failed for {url}: {message[:300]}")
