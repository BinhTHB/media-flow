from __future__ import annotations

import threading
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional

from src.infrastructure.gemini.contracts import (
    MODE_ROLE_PRIMARY,
    MODE_ROLE_ESCALATION,
)


@dataclass
class UsageRecord:
    model: str
    role: str = MODE_ROLE_PRIMARY
    prompt_tokens: int = 0
    candidates_tokens: int = 0
    total_tokens: int = 0
    request_id: Optional[str] = None
    cache_hit: bool = False
    retry_count: int = 0
    is_escalation: bool = False
    source_kind: Optional[str] = None
    prompt_version: Optional[str] = None
    schema_version: Optional[str] = None
    api_surface: Optional[str] = None
    media_sha256: Optional[str] = None


class UsageRecorder:
    """Thread-safe usage recorder. Stores only safe provenance metadata (no secrets)."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._records: List[UsageRecord] = []
        self._by_model: Dict[str, Dict[str, int]] = defaultdict(lambda: {
            "requests": 0,
            "prompt_tokens": 0,
            "candidates_tokens": 0,
            "total_tokens": 0,
            "cache_hits": 0,
            "retries": 0,
            "escalations": 0,
            "rate_limits": 0,
        })

    def record(self, record: UsageRecord) -> None:
        with self._lock:
            self._records.append(record)
            agg = self._by_model[record.model]
            agg["requests"] += 1
            agg["prompt_tokens"] += record.prompt_tokens
            agg["candidates_tokens"] += record.candidates_tokens
            agg["total_tokens"] += record.total_tokens
            if record.cache_hit:
                agg["cache_hits"] += 1
            if record.retry_count:
                agg["retries"] += record.retry_count
            if record.is_escalation:
                agg["escalations"] += 1

    def record_rate_limit(self, model: str) -> None:
        with self._lock:
            self._by_model[model]["rate_limits"] += 1

    def totals(self) -> Dict[str, int]:
        with self._lock:
            out: Dict[str, int] = {
                "requests": 0,
                "prompt_tokens": 0,
                "candidates_tokens": 0,
                "total_tokens": 0,
                "cache_hits": 0,
                "retries": 0,
                "escalations": 0,
                "rate_limits": 0,
            }
            for agg in self._by_model.values():
                for key in out:
                    out[key] += agg[key]
            return out

    def by_model(self, model: str) -> Dict[str, int]:
        with self._lock:
            return dict(self._by_model.get(model, {}))

    def all(self) -> List[UsageRecord]:
        with self._lock:
            return list(self._records)

    def reset(self) -> None:
        with self._lock:
            self._records.clear()
            self._by_model.clear()
