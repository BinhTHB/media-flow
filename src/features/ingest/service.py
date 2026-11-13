"""Ingest use case: remote source -> local file the pipeline can process.

Responsibilities kept here (not in the downloader):

- **retry policy** — TransientIngestError backs off; anything else is permanent
- **idempotency** — a manifest keyed by source URL records what was already
  downloaded, so re-running a batch does not re-fetch gigabytes
- **artifact layout** — one directory per run, with a stable manifest

The downloader stays a dumb provider: give it a SourceSpec and a directory, get
an IngestResult back.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

from src.core.contracts import DownloaderPort, TransientIngestError
from src.core.models.ingest import (
    IngestBatch,
    IngestResult,
    SourceSpec,
    infer_platform,
)

logger = logging.getLogger(__name__)

MANIFEST_NAME = "manifest.json"


class IngestService:
    def __init__(
        self,
        downloader: DownloaderPort,
        *,
        max_attempts: int = 3,
        backoff_base_seconds: float = 5.0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.downloader = downloader
        self.max_attempts = max(1, int(max_attempts))
        self.backoff_base_seconds = float(backoff_base_seconds)
        self._sleep = sleep

    # -- single source ------------------------------------------------------

    def ingest_one(
        self, source: SourceSpec, output_dir: str, *, overwrite: bool = False
    ) -> IngestResult:
        root = Path(output_dir)
        root.mkdir(parents=True, exist_ok=True)
        manifest = _Manifest(root / MANIFEST_NAME)

        if not overwrite:
            previous = manifest.lookup(source.url)
            if previous and previous.get("status") == "OK":
                path = previous.get("video_path")
                if path and Path(path).is_file():
                    logger.info("Ingest: %s already downloaded; skipping.", source.url)
                    return IngestResult(
                        source=source,
                        status="SKIPPED",
                        video_path=path,
                        subtitle_path=previous.get("subtitle_path"),
                        title=str(previous.get("title", "")),
                        duration_seconds=float(previous.get("duration_seconds", 0.0)),
                        width=int(previous.get("width", 0) or 0),
                        height=int(previous.get("height", 0) or 0),
                        content_id=str(previous.get("content_id", "")),
                    )

        last_error: Optional[str] = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                result = self.downloader.download(source, str(root))
                logger.info(
                    "Ingest: %s -> %s (%s) on attempt %d",
                    source.url, result.video_path, result.status, attempt,
                )
                manifest.record(result)
                return result
            except TransientIngestError as exc:
                last_error = f"transient: {exc}"
                if attempt < self.max_attempts:
                    delay = self.backoff_base_seconds * (2 ** (attempt - 1))
                    logger.warning(
                        "Ingest: %s transient failure (attempt %d/%d), retrying in %.0fs: %s",
                        source.url, attempt, self.max_attempts, delay, exc,
                    )
                    self._sleep(delay)
            except Exception as exc:  # permanent: no retry
                last_error = f"permanent: {exc}"
                logger.error("Ingest: %s failed permanently: %s", source.url, exc)
                break

        failed = IngestResult(source=source, status="FAILED", error=last_error)
        manifest.record(failed)
        return failed

    # -- batch --------------------------------------------------------------

    def ingest_many(
        self,
        sources: Iterable[SourceSpec],
        output_dir: str,
        *,
        overwrite: bool = False,
    ) -> IngestBatch:
        batch = IngestBatch()
        for source in sources:
            batch.add(self.ingest_one(source, output_dir, overwrite=overwrite))
        return batch

    def known_ok_urls(self, output_dir: str) -> set[str]:
        """Video URLs this directory already holds."""
        return _Manifest(Path(output_dir) / MANIFEST_NAME).ok_urls()


class _Manifest:
    """Small JSON ledger of what this ingest directory already contains.

    Rewritten atomically on every record so an interrupted run leaves a valid
    file rather than a half-written one.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._entries: Dict[str, Dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        if not self._path.is_file():
            return
        try:
            payload = json.loads(self._path.read_text(encoding="utf-8"))
            if isinstance(payload, dict) and isinstance(payload.get("sources"), dict):
                self._entries = {
                    str(k): dict(v)
                    for k, v in payload["sources"].items()
                    if isinstance(v, dict)
                }
        except (OSError, ValueError) as exc:
            logger.warning("Ingest manifest %s unreadable (%s); starting fresh.", self._path, exc)
            self._entries = {}

    def lookup(self, url: str) -> Optional[Dict[str, Any]]:
        return self._entries.get(url)

    def ok_urls(self) -> set[str]:
        """URLs whose file is still on disk."""
        found: set[str] = set()
        for url, entry in self._entries.items():
            path = entry.get("video_path")
            if entry.get("status") == "OK" and path and Path(str(path)).is_file():
                found.add(url)
        return found

    def record(self, result: IngestResult) -> None:
        self._entries[result.source.url] = result.to_dict()
        self._save()

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"version": 1, "sources": self._entries}
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self._path)
