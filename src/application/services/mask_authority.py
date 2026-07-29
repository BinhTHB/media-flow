"""Track-scoped authority for classifier-backed subtitle mask rendering."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Mapping


class DecisionStore:
    """Append-only, per-run decision table keyed by ``track_id``."""

    def __init__(self, run_id: str, path: str | Path | None = None) -> None:
        self.run_id = run_id
        self.path = Path(path) if path else None
        self._records: dict[str, dict[str, Any]] = {}
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text("", encoding="utf-8")

    def record(
        self,
        *,
        track_id: str,
        model: str,
        is_subtitle: bool,
        response_ref: str,
        cache_hit: bool,
    ) -> dict[str, Any]:
        if not track_id:
            raise ValueError("track_id is required for a subtitle decision")
        if track_id in self._records:
            raise ValueError(f"Duplicate subtitle decision for track_id: {track_id}")
        record = {
            "run_id": self.run_id,
            "track_id": track_id,
            "is_subtitle": bool(is_subtitle),
            "model": model,
            "response_ref": response_ref,
            "cache_hit": bool(cache_hit),
        }
        self._records[track_id] = record
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        return record

    def get(self, track_id: str | None) -> dict[str, Any] | None:
        if not track_id:
            return None
        return self._records.get(track_id)

    def permits_mask(self, track_id: str | None) -> bool:
        """Return true only for a current-run positive decision with raw output."""
        record = self.get(track_id)
        response_ref = record.get("response_ref") if record else None
        response_path = self.path.parent / response_ref if self.path and response_ref else None
        return bool(
            record
            and record["run_id"] == self.run_id
            and record["is_subtitle"] is True
            and response_path
            and response_path.is_file()
        )

    def positive_count(self) -> int:
        return sum(record["is_subtitle"] is True for record in self._records.values())

    def is_for_run(self, run_id: str) -> bool:
        return self.run_id == run_id


def has_verified_mask_provenance(
    region: Any, store: DecisionStore, expected_run_id: str | None = None
) -> bool:
    """Validate a mask region only through its track decision."""
    valid = bool(
        (expected_run_id is None or store.is_for_run(expected_run_id))
        and store.permits_mask(getattr(region, "track_id", None))
    )
    if not valid:
        logging.getLogger(__name__).warning(
            "LOUD-WARN [MaskAuthority]: discarded mask for unverified track_id=%s",
            getattr(region, "track_id", None),
        )
    return valid
