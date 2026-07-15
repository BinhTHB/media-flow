from __future__ import annotations

import hashlib
import json
import os
import threading
from typing import Any, Dict, Optional

from src.infrastructure.gemini.exceptions import GeminiCacheError


def _default_clock() -> float:
    import time
    return time.time()


class ResponseCache:
    """Disk-backed JSON response cache with atomic writes. Never stores secrets."""

    def __init__(self, cache_dir: Optional[str] = None, namespace: str = "default",
                 enabled: bool = True, ttl_seconds: Optional[float] = None,
                 clock=None) -> None:
        self.cache_dir = cache_dir
        self.namespace = namespace
        self.enabled = bool(enabled and cache_dir)
        self.ttl_seconds = ttl_seconds
        self._clock = clock or _default_clock
        self._lock = threading.RLock()
        if self.enabled:
            os.makedirs(self.cache_dir, exist_ok=True)

    @property
    def path(self) -> Optional[str]:
        if not self.enabled or not self.cache_dir:
            return None
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in self.namespace)
        return os.path.join(self.cache_dir, f"{safe}.json")

    def make_key(self, **parts: Any) -> str:
        canonical = json.dumps(parts, sort_keys=True, default=str, ensure_ascii=False)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        if not self.enabled:
            return None
        with self._lock:
            data = self._load()
        entry = data.get("entries", {}).get(key)
        if entry is None:
            return None
        if self._is_expired(entry):
            return None
        return entry.get("value")

    def set(self, key: str, value: Dict[str, Any]) -> None:
        if not self.enabled:
            return
        with self._lock:
            data = self._load()
            data.setdefault("entries", {})[key] = {"value": value, "stored_at": self._clock()}
            self._save(data)

    def invalidate(self, key: str) -> None:
        if not self.enabled:
            return
        with self._lock:
            data = self._load()
            data.get("entries", {}).pop(key, None)
            self._save(data)

    def clear(self) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._save({"entries": {}})

    def _load(self) -> Dict[str, Any]:
        path = self.path
        if not path:
            return {"entries": {}}
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if not isinstance(data, dict):
                return {"entries": {}}
            return data
        except FileNotFoundError:
            return {"entries": {}}
        except (json.JSONDecodeError, OSError):
            return {"entries": {}}

    def _save(self, data: Dict[str, Any]) -> None:
        path = self.path
        if not path:
            raise GeminiCacheError("ResponseCache is disabled; cannot write cache.")
        tmp = f"{path}.{os.getpid()}.{threading.get_ident()}.tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2)
        os.replace(tmp, path)

    def _is_expired(self, entry: Dict[str, Any]) -> bool:
        if self.ttl_seconds is None:
            return False
        stored_at = entry.get("stored_at")
        if stored_at is None:
            return False
        return (self._clock() - float(stored_at)) > self.ttl_seconds
