from __future__ import annotations

import json
import os
import threading
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Deque, Dict, Optional, Tuple
from zoneinfo import ZoneInfo

from src.infrastructure.gemini.exceptions import GeminiSchedulerError, ProviderRateLimitError

Clock = Callable[[], float]
Sleeper = Callable[[float], None]
RandomSource = Callable[[], float]
DayKey = Callable[[], str]

WINDOW_SECONDS = 60.0


class RequestScheduler:
    """Shared scheduler with an optional persistent generate-content ledger."""

    def __init__(self, *, rpm_cap: int = 60, tpm_cap: Optional[int] = None,
                 rpd_cap: Optional[int] = None,
                 max_concurrency: int = 1, max_retries: int = 5,
                 backoff_initial_seconds: float = 2.0, backoff_max_seconds: float = 60.0,
                 respect_retry_after: bool = True,
                 clock: Optional[Clock] = None, sleeper: Optional[Sleeper] = None,
                 random_source: Optional[RandomSource] = None,
                 day_key: Optional[DayKey] = None,
                 ledger_path: Optional[str] = None,
                 min_request_spacing_seconds: float = 0.0) -> None:
        if rpm_cap < 1:
            raise GeminiSchedulerError("rpm_cap must be >= 1")
        if max_retries < 0:
            raise GeminiSchedulerError("max_retries must be >= 0")
        if rpd_cap is not None and rpd_cap < 1:
            raise GeminiSchedulerError("rpd_cap must be >= 1")
        if min_request_spacing_seconds < 0:
            raise GeminiSchedulerError("min_request_spacing_seconds must be non-negative")
        self.rpm_cap = rpm_cap
        self.tpm_cap = tpm_cap
        self.rpd_cap = rpd_cap
        self.max_concurrency = max(1, int(max_concurrency))
        self.max_retries = max_retries
        self.backoff_initial_seconds = backoff_initial_seconds
        self.backoff_max_seconds = backoff_max_seconds
        self.respect_retry_after = respect_retry_after
        self._clock = clock or _default_clock
        self._sleeper = sleeper or _default_sleeper
        self._random = random_source or _default_random
        self._day_key = day_key or _pacific_day_key
        self._ledger_path = Path(ledger_path) if ledger_path else None
        self.min_request_spacing_seconds = min_request_spacing_seconds
        self._lock = threading.RLock()
        self._request_times: Deque[float] = deque()
        self._token_events: Deque[Tuple[float, int]] = deque()
        self._count_tokens_times: Deque[float] = deque()
        self._upload_times: Deque[float] = deque()
        self._active = 0
        self.retry_count = 0
        self.rate_limit_count = 0
        self.request_count = 0
        self.rpd_count = 0
        self._rpd_day = self._day_key()
        self.request_count_by_model: Dict[str, int] = {}
        self.escalation_count = 0
        self._load_ledger()

    def _load_ledger(self) -> None:
        if self._ledger_path is None:
            return
        try:
            payload = json.loads(self._ledger_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        if payload.get("clock") != "epoch_seconds":
            return
        now = self._clock()
        self._request_times = self._timestamps(payload.get("generate_request_times"), now)
        self._token_events = self._tokens(payload.get("generate_token_events"), now)
        self._count_tokens_times = self._timestamps(payload.get("count_tokens_times"), now)
        self._upload_times = self._timestamps(payload.get("upload_times"), now)

    @staticmethod
    def _timestamps(values: Any, now: float) -> Deque[float]:
        return deque(
            value for value in values or []
            if isinstance(value, (int, float)) and value > now - WINDOW_SECONDS
        )

    @staticmethod
    def _tokens(values: Any, now: float) -> Deque[Tuple[float, int]]:
        return deque(
            (timestamp, tokens) for timestamp, tokens in values or []
            if isinstance(timestamp, (int, float)) and isinstance(tokens, int)
            and timestamp > now - WINDOW_SECONDS
        )

    def _refresh_ledger(self) -> None:
        if self._ledger_path is not None:
            self._load_ledger()

    def _save_ledger(self) -> None:
        if self._ledger_path is None:
            return
        payload = {
            "clock": "epoch_seconds",
            "generate_request_times": list(self._request_times),
            "generate_token_events": list(self._token_events),
            "count_tokens_times": list(self._count_tokens_times),
            "upload_times": list(self._upload_times),
        }
        try:
            self._ledger_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self._ledger_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(payload), encoding="utf-8")
            os.replace(temporary, self._ledger_path)
        except OSError:
            pass

    def acquire(self, tokens: int = 0, *, model: Optional[str] = None,
                is_escalation: bool = False) -> None:
        while True:
            with self._lock:
                self._refresh_ledger()
                now = self._clock()
                self._reset_rpd_if_needed()
                if self.rpd_cap is not None and self.rpd_count >= self.rpd_cap:
                    raise ProviderRateLimitError("Gemini daily request cap reached.")
                self._purge(now)
                wait = self._compute_wait(now, tokens)
                if wait <= 0:
                    self._request_times.append(now)
                    if tokens:
                        self._token_events.append((now, tokens))
                    self._active += 1
                    self.request_count += 1
                    self.rpd_count += 1
                    if model:
                        self.request_count_by_model[model] = self.request_count_by_model.get(model, 0) + 1
                    if is_escalation:
                        self.escalation_count += 1
                    self._save_ledger()
                    return
                sleep_for = wait
            self._sleeper(sleep_for)

    def release(self) -> None:
        with self._lock:
            if self._active > 0:
                self._active -= 1

    def record_count_tokens(self) -> None:
        with self._lock:
            self._refresh_ledger()
            now = self._clock()
            self._purge(now)
            self._count_tokens_times.append(now)
            self._save_ledger()

    def record_upload(self) -> None:
        with self._lock:
            self._refresh_ledger()
            now = self._clock()
            self._purge(now)
            self._upload_times.append(now)
            self._save_ledger()

    def on_rate_limited(self, retry_after: Optional[float] = None) -> float:
        with self._lock:
            self.rate_limit_count += 1
        delay = self.compute_retry_delay(self.retry_count, retry_after)
        self._sleeper(delay)
        with self._lock:
            self.retry_count += 1
        return delay

    def compute_retry_delay(self, attempt: int, retry_after: Optional[float] = None) -> float:
        if self.respect_retry_after and retry_after is not None:
            return float(retry_after)
        base = self.backoff_initial_seconds * (2 ** attempt)
        base = min(base, self.backoff_max_seconds)
        return base + self._random() * 0.1 * base

    def should_retry(self, attempt: int) -> bool:
        return attempt < self.max_retries

    def ledger_state(self, tokens: int = 0) -> Dict[str, Any]:
        with self._lock:
            self._refresh_ledger()
            now = self._clock()
            self._reset_rpd_if_needed()
            self._purge(now)
            return {
                "rolling_requests": len(self._request_times),
                "rolling_tokens": sum(value for _, value in self._token_events),
                "count_tokens_requests": len(self._count_tokens_times),
                "upload_requests": len(self._upload_times),
                "configured_rpm": self.rpm_cap,
                "configured_tpm": self.tpm_cap,
                "configured_rpd": self.rpd_cap,
                "minimum_request_spacing_seconds": self.min_request_spacing_seconds,
                "next_eligible_wait_seconds": self._compute_wait(now, tokens),
                "last_request_age_seconds": (
                    now - self._request_times[-1] if self._request_times else None
                ),
            }

    def _compute_wait(self, now: float, tokens: int) -> float:
        wait = 0.0
        if self._request_times:
            wait = max(
                wait,
                self._request_times[-1] + self.min_request_spacing_seconds - now,
            )
        if len(self._request_times) >= self.rpm_cap:
            wait = max(wait, self._request_times[0] + WINDOW_SECONDS - now)
        if self._active >= self.max_concurrency:
            wait = max(wait, 0.01)
        if self.tpm_cap is not None:
            deficit = sum(value for _, value in self._token_events) + tokens - self.tpm_cap
            for timestamp, value in self._token_events:
                if deficit <= 0:
                    break
                deficit -= value
                wait = max(wait, timestamp + WINDOW_SECONDS - now)
        return max(wait, 0.0)

    def _purge(self, now: float) -> None:
        cutoff = now - WINDOW_SECONDS
        for events in (self._request_times, self._count_tokens_times, self._upload_times):
            while events and events[0] <= cutoff:
                events.popleft()
        while self._token_events and self._token_events[0][0] <= cutoff:
            self._token_events.popleft()

    def _reset_rpd_if_needed(self) -> None:
        day = self._day_key()
        if day != self._rpd_day:
            self._rpd_day = day
            self.rpd_count = 0
            self.request_count_by_model.clear()
            self.escalation_count = 0


def _default_clock() -> float:
    import time
    return time.time()


def _default_sleeper(seconds: float) -> None:
    import time
    time.sleep(seconds)


def _default_random() -> float:
    import random
    return random.random()


def _pacific_day_key() -> str:
    return datetime.now(ZoneInfo("America/Los_Angeles")).date().isoformat()
