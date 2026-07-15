from __future__ import annotations

import threading
from collections import deque
from typing import Callable, Deque, Optional, Tuple

from src.infrastructure.gemini.exceptions import GeminiSchedulerError

Clock = Callable[[], float]
Sleeper = Callable[[float], None]
RandomSource = Callable[[], float]

WINDOW_SECONDS = 60.0


class RequestScheduler:
    """Deterministic, in-process request scheduler shared across Gemini providers.

    Enforces RPM/TPM/concurrency caps and computes retry backoff. The clock,
    sleeper and random source are injectable so behavior is fully deterministic
    and testable without real sleeps or wall-clock dependence.
    """

    def __init__(self, *, rpm_cap: int = 60, tpm_cap: Optional[int] = None,
                 max_concurrency: int = 1, max_retries: int = 5,
                 backoff_initial_seconds: float = 2.0, backoff_max_seconds: float = 60.0,
                 respect_retry_after: bool = True,
                 clock: Optional[Clock] = None, sleeper: Optional[Sleeper] = None,
                 random_source: Optional[RandomSource] = None) -> None:
        if rpm_cap < 1:
            raise GeminiSchedulerError("rpm_cap must be >= 1")
        if max_retries < 0:
            raise GeminiSchedulerError("max_retries must be >= 0")
        self.rpm_cap = rpm_cap
        self.tpm_cap = tpm_cap
        self.max_concurrency = max(1, int(max_concurrency))
        self.max_retries = max_retries
        self.backoff_initial_seconds = backoff_initial_seconds
        self.backoff_max_seconds = backoff_max_seconds
        self.respect_retry_after = respect_retry_after
        self._clock = clock or _default_clock
        self._sleeper = sleeper or _default_sleeper
        self._random = random_source or _default_random
        self._lock = threading.RLock()
        self._request_times: Deque[float] = deque()
        self._token_events: Deque[Tuple[float, int]] = deque()
        self._active = 0
        self.retry_count = 0
        self.rate_limit_count = 0
        self.request_count = 0

    def acquire(self, tokens: int = 0) -> None:
        while True:
            with self._lock:
                now = self._clock()
                self._purge(now)
                wait = self._compute_wait(now, tokens)
                if wait <= 0:
                    self._request_times.append(now)
                    if tokens:
                        self._token_events.append((now, tokens))
                    self._active += 1
                    self.request_count += 1
                    return
                sleep_for = wait
            self._sleeper(sleep_for)

    def release(self) -> None:
        with self._lock:
            if self._active > 0:
                self._active -= 1

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
        jitter = self._random() * 0.1 * base
        return base + jitter

    def should_retry(self, attempt: int) -> bool:
        return attempt < self.max_retries

    def _compute_wait(self, now: float, tokens: int) -> float:
        wait = 0.0
        if len(self._request_times) >= self.rpm_cap:
            oldest = self._request_times[0]
            wait = max(wait, oldest + WINDOW_SECONDS - now)
        if self._active >= self.max_concurrency:
            wait = max(wait, 0.01)
        if self.tpm_cap is not None:
            used = sum(t for _, t in self._token_events)
            if used + tokens > self.tpm_cap:
                deficit = used + tokens - self.tpm_cap
                freed = 0.0
                for ts, t in self._token_events:
                    freed = ts + WINDOW_SECONDS - now
                    deficit -= t
                    if deficit <= 0:
                        break
                wait = max(wait, max(freed, 0.0))
        return max(wait, 0.0)

    def _purge(self, now: float) -> None:
        cutoff = now - WINDOW_SECONDS
        while self._request_times and self._request_times[0] <= cutoff:
            self._request_times.popleft()
        while self._token_events and self._token_events[0][0] <= cutoff:
            self._token_events.popleft()


def _default_clock() -> float:
    import time
    return time.monotonic()


def _default_sleeper(seconds: float) -> None:
    import time
    time.sleep(seconds)


def _default_random() -> float:
    import random
    return random.random()
