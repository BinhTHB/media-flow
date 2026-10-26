"""JSON-file state store for publication records.

Persisted under ``outputs/publishing/state.json`` by default; the worker is
single-process so plain read-modify-write is sufficient (no locking)."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

from src.core.models.publication import PublishRecord

logger = logging.getLogger(__name__)


class JsonPublicationStore:
    def __init__(self, state_path: str) -> None:
        self._path = Path(state_path)
        self._records: Dict[str, List[dict]] = {}
        self._load()

    def _load(self) -> None:
        if not self._path.is_file():
            return
        try:
            payload = json.loads(self._path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                self._records = {k: list(v) for k, v in payload.items() if isinstance(v, list)}
        except (OSError, ValueError) as exc:
            # A corrupt state file must not wedge the whole worker: start
            # fresh and let the next successful publish rewrite it.
            logger.warning("Publish store %s unreadable (%s); starting empty.", self._path, exc)
            self._records = {}

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._records, indent=2), encoding="utf-8")
        tmp.replace(self._path)

    def is_published(self, item_id: str, platform: str) -> bool:
        for record in self._records.get(item_id, []):
            if record.get("platform") == platform and record.get("status") == "PUBLISHED":
                return True
        return False

    def add(self, record: PublishRecord) -> None:
        self._records.setdefault(record.item_id, []).append(asdict(record))
        self._save()
