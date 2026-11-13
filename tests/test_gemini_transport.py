"""Regression tests: transient Gemini transport errors must surface as the
retryable GeminiNetworkError. A real dubbing job (outputs/jobs/job_v2_1790572244)
failed with a raw read TimeoutError that bypassed the client retry loop even
though the scheduler allowed 5 retries."""

import urllib.error
from unittest.mock import patch

import pytest

from src.infrastructure.gemini.client import GeminiClient, _UrllibTransport
from src.infrastructure.gemini.contracts import GeminiRequest
from src.infrastructure.gemini.exceptions import GeminiNetworkError
from src.infrastructure.gemini.request_scheduler import RequestScheduler


class _FakeResponse:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def read(self) -> bytes:
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


def test_read_timeout_wraps_as_retryable_network_error():
    def raise_read_timeout(req, timeout):
        raise TimeoutError("The read operation timed out")

    with patch("urllib.request.urlopen", raise_read_timeout):
        with pytest.raises(GeminiNetworkError, match="read operation timed out"):
            _UrllibTransport().send("POST", "https://example.invalid", headers={}, data=b"{}")


def test_url_error_still_wraps_reason():
    def raise_url_error(req, timeout):
        raise urllib.error.URLError(OSError("connection refused"))

    with patch("urllib.request.urlopen", raise_url_error):
        with pytest.raises(GeminiNetworkError, match="connection refused"):
            _UrllibTransport().send("POST", "https://example.invalid", headers={}, data=b"{}")


def test_client_retries_wrapped_transient_errors():
    class _FlakyTransport:
        def __init__(self) -> None:
            self.attempts = 0

        def send(self, method, url, *, headers=None, data=None, timeout=180.0):
            self.attempts += 1
            if self.attempts < 3:
                raise GeminiNetworkError("Gemini request failed: The read operation timed out")
            return {
                "candidates": [{"content": {"parts": [{"text": "ok"}]}}],
                "usageMetadata": {},
            }

    transport = _FlakyTransport()
    client = GeminiClient(
        api_key="test-key",
        transport=transport,
        scheduler=RequestScheduler(sleeper=lambda seconds: None),
    )

    resp = client.generate_content(GeminiRequest(contents=[{"parts": [{"text": "hi"}]}]))

    assert resp.text == "ok"
    assert transport.attempts == 3


def test_retry_after_cannot_stall_a_dub_for_an_hour():
    scheduler = RequestScheduler(backoff_max_seconds=60.0, random_source=lambda: 0.0)
    assert scheduler.compute_retry_delay(0, retry_after=3600) == 60.0
