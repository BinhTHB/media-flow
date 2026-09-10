"""Durable, bounded, non-secret job state under outputs/jobs/<job_id>."""
from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
import threading
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

_SECRET_KEY = re.compile(r"(api[_-]?key|token|secret|password|authorization|credential)", re.I)
_SAFE_JOB_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_TERMINAL = {"SUCCESS", "PARTIAL", "FAILED", "BLOCKED"}


def redact_secrets(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): "[REDACTED]" if _SECRET_KEY.search(str(key)) else redact_secrets(item) for key, item in value.items()}
    if isinstance(value, list):
        return [redact_secrets(item) for item in value]
    return value


class JobStore:
    """Atomic JSON store with process-local claims and deterministic cleanup."""

    _claims_lock = threading.Lock()
    _claims: set[str] = set()

    def __init__(self, base_dir: str | Path = "outputs/jobs", *, max_events: int = 200) -> None:
        self.base_dir = Path(base_dir)
        self.max_events = max(1, max_events)
        self._lock = threading.RLock()

    def job_dir(self, job_id: str) -> Path:
        if not _SAFE_JOB_ID.fullmatch(job_id):
            raise ValueError("job_id must contain only letters, numbers, '.', '_' or '-'.")
        return self.base_dir / job_id

    def _claim_key(self, job_id: str) -> str:
        return str(self.job_dir(job_id).resolve())

    def claim(self, job_id: str) -> bool:
        key = self._claim_key(job_id)
        with self._claims_lock:
            if key in self._claims:
                return False
            self._claims.add(key)
            return True

    def release(self, job_id: str) -> None:
        with self._claims_lock:
            self._claims.discard(self._claim_key(job_id))

    def is_claimed(self, job_id: str) -> bool:
        with self._claims_lock:
            return self._claim_key(job_id) in self._claims

    def request_cancel(self, job_id: str) -> None:
        with self._lock:
            self._atomic_write(self.job_dir(job_id) / "cancel.json", {"requested": True})

    def cancel_requested(self, job_id: str) -> bool:
        value = self._read(self.job_dir(job_id) / "cancel.json", {})
        return bool(value.get("requested")) if isinstance(value, dict) else False

    def _atomic_write(self, path: Path, payload: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        handle, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as stream:
                json.dump(redact_secrets(deepcopy(payload)), stream, ensure_ascii=False, indent=2)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp_name, path)
        except Exception:
            try:
                os.unlink(temp_name)
            except OSError:
                pass
            raise

    @staticmethod
    def _read(path: Path, fallback: Any = None) -> Any:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return fallback

    def save_state(self, job_id: str, state: dict[str, Any]) -> None:
        with self._lock:
            self._atomic_write(self.job_dir(job_id) / "state.json", state)

    def append_event(self, job_id: str, event: dict[str, Any]) -> None:
        with self._lock:
            path = self.job_dir(job_id) / "events.json"
            events = self._read(path, [])
            events.append(event)
            self._atomic_write(path, events[-self.max_events:])

    def save_result(self, job_id: str, result: dict[str, Any]) -> None:
        with self._lock:
            self._atomic_write(self.job_dir(job_id) / "result.json", result)

    def get(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            directory = self.job_dir(job_id)
            state = self._read(directory / "state.json")
            if state is None:
                return None
            state["events"] = self._read(directory / "events.json", [])
            result = self._read(directory / "result.json")
            if result is not None:
                state["result"] = result
            return state

    def list(self) -> list[dict[str, Any]]:
        if not self.base_dir.is_dir():
            return []
        jobs = []
        for directory in self.base_dir.iterdir():
            state = self._read(directory / "state.json") if directory.is_dir() else None
            if isinstance(state, dict):
                jobs.append(state)
        return sorted(jobs, key=lambda item: float(item.get("updated_at", item.get("created_at", 0))), reverse=True)

    def cleanup(
        self,
        *,
        max_age_seconds: float | None = None,
        max_jobs: int | None = None,
        max_total_bytes: int | None = None,
        now: float | None = None,
    ) -> list[str]:
        """Remove oldest terminal, unclaimed jobs until all configured quotas pass."""
        current_time = time.time() if now is None else now
        candidates: list[tuple[float, int, str, Path]] = []
        for state in reversed(self.list()):
            job_id = str(state.get("id", ""))
            if not job_id or str(state.get("status")) not in _TERMINAL or self.is_claimed(job_id):
                continue
            directory = self.job_dir(job_id)
            size = sum(path.stat().st_size for path in directory.rglob("*") if path.is_file())
            updated = float(state.get("updated_at", state.get("created_at", 0)))
            candidates.append((updated, size, job_id, directory))

        all_jobs = self.list()
        total_bytes = sum(
            path.stat().st_size for path in self.base_dir.rglob("*") if path.is_file()
        ) if self.base_dir.is_dir() else 0
        removed: list[str] = []
        remaining_jobs = len(all_jobs)
        for updated, size, job_id, directory in candidates:
            expired = max_age_seconds is not None and current_time - updated > max_age_seconds
            over_count = max_jobs is not None and remaining_jobs > max_jobs
            over_bytes = max_total_bytes is not None and total_bytes > max_total_bytes
            if not (expired or over_count or over_bytes):
                continue
            shutil.rmtree(directory)
            removed.append(job_id)
            remaining_jobs -= 1
            total_bytes -= size
        return removed
