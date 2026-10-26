"""Filesystem session store for cookie-based publishers.

A browser session (cookie jar + storage state) is the credential a
cookie-based publisher uses instead of an OAuth token. This store keeps one
file per account under a session root and a small JSON index of metadata, so
the worker can tell *which* accounts are still logged in without opening a
browser.

Session files are secrets: they are written with restrictive permissions where
the platform allows it, and never logged. Only the index (paths, timestamps,
validity) is safe to print.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from src.core.contracts import SessionStorePort
from src.core.models.account import AccountIdentity, SessionState

logger = logging.getLogger(__name__)

INDEX_NAME = "sessions.json"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class FileSessionStore(SessionStorePort):
    """One session file per account, plus a metadata index."""

    def __init__(self, root: str) -> None:
        self._root = Path(root)
        self._index_path = self._root / INDEX_NAME
        self._index: Dict[str, Dict[str, Any]] = {}
        self._load_index()

    # -- index --------------------------------------------------------------

    def _load_index(self) -> None:
        if not self._index_path.is_file():
            return
        try:
            payload = json.loads(self._index_path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                self._index = {
                    str(k): dict(v) for k, v in payload.items() if isinstance(v, dict)
                }
        except (OSError, ValueError) as exc:
            # A corrupt index must not block publishing: rebuild on next save.
            logger.warning("Session index %s unreadable (%s); starting empty.", self._index_path, exc)
            self._index = {}

    def _save_index(self) -> None:
        self._root.mkdir(parents=True, exist_ok=True)
        tmp = self._index_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._index, indent=2), encoding="utf-8")
        tmp.replace(self._index_path)

    # -- port ---------------------------------------------------------------

    def load(self, account: AccountIdentity) -> Optional[str]:
        entry = self._index.get(account.key)
        if not entry:
            return None
        if not entry.get("valid", True):
            return None
        path = Path(str(entry.get("session_path", "")))
        if not path.is_absolute():
            path = self._root / path
        return str(path) if path.is_file() else None

    def save(self, account: AccountIdentity, session_path: str) -> str:
        self._root.mkdir(parents=True, exist_ok=True)
        source = Path(session_path)
        if not source.is_file():
            raise FileNotFoundError(session_path)

        # Keep every session inside the store root so paths stay portable and
        # one account cannot point at another's file by accident.
        target = self._root / f"{_safe_name(account.key)}.json"
        if source.resolve() != target.resolve():
            target.write_bytes(source.read_bytes())
        _restrict_permissions(target)

        existing = self._index.get(account.key, {})
        state = SessionState(
            account_key=account.key,
            session_path=target.name,
            created_at=str(existing.get("created_at") or _utc_now()),
            last_used_at=_utc_now(),
            valid=True,
            note="",
        )
        self._index[account.key] = state.to_dict()
        self._save_index()
        logger.info("Session: stored session for %s", account.key)
        return str(target)

    def invalidate(self, account: AccountIdentity, note: str = "") -> None:
        entry = self._index.get(account.key)
        if not entry:
            return
        entry["valid"] = False
        entry["note"] = note
        entry["last_used_at"] = _utc_now()
        self._index[account.key] = entry
        self._save_index()
        logger.warning("Session: invalidated %s (%s)", account.key, note or "no reason")

    # -- introspection (safe to log) ---------------------------------------

    def touch(self, account: AccountIdentity) -> None:
        entry = self._index.get(account.key)
        if entry:
            entry["last_used_at"] = _utc_now()
            self._index[account.key] = entry
            self._save_index()

    def list_states(self) -> list[dict]:
        return [dict(v) for v in self._index.values()]


def _safe_name(key: str) -> str:
    """Turn ``platform:account`` into a filesystem-safe stem."""
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in key)


def _restrict_permissions(path: Path) -> None:
    """Best-effort owner-only permissions for a secret file."""
    try:
        os.chmod(path, 0o600)
    except OSError:
        # Windows and some filesystems ignore POSIX modes; the file still lives
        # under a gitignored directory, so this is defence in depth, not the
        # primary protection.
        pass
