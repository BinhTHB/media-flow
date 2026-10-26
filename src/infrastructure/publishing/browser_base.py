"""Base class for cookie/session based browser publishers.

Why a browser instead of an API: the official Content Posting APIs gate re-upload
behind an app audit, cap posts per creator per day, and in TikTok's case
explicitly forbid "an app that copies arbitrary contents from other platforms".
A session-based publisher drives the real web UI with a logged-in profile, which
is what consumer re-upload tools actually do.

Design constraints that matter here:

- **Playwright is optional.** It is imported lazily inside ``_session()`` so the
  rest of MediaFlow (and CI) never needs a browser. A missing install raises a
  clear configuration error, not an ImportError from a random module.
- **Session files are secrets.** They are read from the session store, never
  logged, and a failed login invalidates them so the next run asks for a fresh
  one instead of silently retrying a dead cookie.
- **No magic waits.** Every step waits for a concrete condition with a timeout,
  because ``sleep(5)`` is how these tools rot.
"""

from __future__ import annotations

import logging
import time
from abc import abstractmethod
from pathlib import Path
from typing import Any, Optional

from src.core.contracts import (
    PublicationPort,
    SessionStorePort,
    TransientPublishError,
)
from src.core.models.account import AccountIdentity
from src.core.models.publication import PublishItem, PublishRecord

logger = logging.getLogger(__name__)

#: Default per-step ceiling. Uploads of a 60s vertical video on a slow link take
#: far longer than this; subclasses override ``upload_timeout``.
DEFAULT_STEP_TIMEOUT_MS = 30_000


class BrowserSessionExpired(RuntimeError):
    """The stored session is no longer logged in and must be refreshed."""


class BrowserPublisher(PublicationPort):
    """Drive one platform's web upload UI with a stored browser session."""

    platform_name = "browser"

    def __init__(
        self,
        account: AccountIdentity,
        session_store: SessionStorePort,
        *,
        headless: bool = True,
        step_timeout_ms: int = DEFAULT_STEP_TIMEOUT_MS,
        upload_timeout_ms: int = 600_000,
        slow_mo_ms: int = 0,
        user_agent: Optional[str] = None,
    ) -> None:
        self.account = account
        self._session_store = session_store
        self._headless = bool(headless)
        self._step_timeout_ms = max(1_000, int(step_timeout_ms))
        self._upload_timeout_ms = max(self._step_timeout_ms, int(upload_timeout_ms))
        self._slow_mo_ms = max(0, int(slow_mo_ms))
        self._user_agent = user_agent
        self.platform_name = account.platform

    # -- subclass hooks -----------------------------------------------------

    @abstractmethod
    def _open_upload_page(self, page: Any) -> None:
        """Navigate to the upload page and wait until it is interactive."""

    @abstractmethod
    def _submit(self, page: Any, item: PublishItem) -> Optional[str]:
        """Fill the form, submit, and return the platform's post id if known."""

    def _is_logged_in(self, page: Any) -> bool:
        """Return False when the page shows a login wall.

        Default heuristic looks for a password field; subclasses with a stable
        marker (avatar, account menu) should override.
        """
        try:
            return page.locator("input[type=password]").count() == 0
        except Exception:
            return True

    # -- template method ----------------------------------------------------

    def publish(self, item: PublishItem) -> PublishRecord:
        session_path = self._session_store.load(self.account)
        if not session_path:
            raise TransientPublishError(
                f"no stored session for {self.account.key}; run the login helper first"
            )

        with self._session(session_path) as page:
            self._open_upload_page(page)
            if not self._is_logged_in(page):
                self._session_store.invalidate(self.account, "login wall on upload page")
                raise TransientPublishError(
                    f"session for {self.account.key} is expired; re-run the login helper"
                )
            platform_video_id = self._submit(page, item)

        self._session_store.touch(self.account)
        return PublishRecord(
            item_id=item.item_id,
            video_path=item.video_path,
            platform=self.platform_name,
            status="PUBLISHED",
            platform_video_id=platform_video_id,
            url=self._post_url(platform_video_id),
        )

    def _post_url(self, platform_video_id: Optional[str]) -> Optional[str]:
        return None

    # -- browser plumbing ---------------------------------------------------

    def _session(self, session_path: str):
        """Yield a ready Playwright page using the stored session."""
        return _BrowserPage(
            session_path=session_path,
            headless=self._headless,
            slow_mo_ms=self._slow_mo_ms,
            user_agent=self._user_agent,
            step_timeout_ms=self._step_timeout_ms,
            upload_timeout_ms=self._upload_timeout_ms,
        )

    # -- helpers for subclasses --------------------------------------------

    def _wait_for_file_chooser(self, page: Any, video_path: str) -> None:
        """Set the file on the next file chooser the page opens."""
        video = Path(video_path)
        if not video.is_file():
            raise FileNotFoundError(video_path)
        with page.expect_file_chooser(timeout=self._step_timeout_ms) as chooser_info:
            page.evaluate("() => document.querySelector('input[type=file]')?.click()")
        chooser_info.value.set_files(str(video))


class _BrowserPage:
    """Context manager that owns a Playwright browser/context/page triple.

    Kept separate from the publisher so the publisher logic reads as a flow
    rather than a resource-management ladder.
    """

    def __init__(
        self,
        *,
        session_path: str,
        headless: bool,
        slow_mo_ms: int,
        user_agent: Optional[str],
        step_timeout_ms: int,
        upload_timeout_ms: int,
    ) -> None:
        self._session_path = session_path
        self._headless = headless
        self._slow_mo_ms = slow_mo_ms
        self._user_agent = user_agent
        self._step_timeout_ms = step_timeout_ms
        self._upload_timeout_ms = upload_timeout_ms
        self._playwright = None
        self._browser = None
        self._context = None
        self._page = None

    def __enter__(self):
        try:
            from playwright.sync_api import sync_playwright
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "playwright is required for session-based publishing. Install it with "
                "`pip install playwright` and then `playwright install chromium`."
            ) from exc

        self._playwright = sync_playwright().start()
        self._browser = self._playwright.chromium.launch(
            headless=self._headless, slow_mo=self._slow_mo_ms or None
        )
        context_args: dict[str, Any] = {"storage_state": self._session_path}
        if self._user_agent:
            context_args["user_agent"] = self._user_agent
        self._context = self._browser.new_context(**context_args)
        self._context.set_default_timeout(self._step_timeout_ms)
        self._page = self._context.new_page()
        self._page.set_default_timeout(self._step_timeout_ms)
        self._page.set_default_navigation_timeout(self._upload_timeout_ms)
        return self._page

    def __exit__(self, exc_type, exc, tb):
        # Persist the refreshed session on success so the cookie jar keeps
        # sliding forward instead of expiring on a fixed clock.
        try:
            if exc_type is None and self._context is not None:
                self._context.storage_state(path=self._session_path)
        except Exception as save_exc:  # never mask the original failure
            logger.debug("Browser: could not persist session state (%s)", save_exc)
        finally:
            for closer in (self._context, self._browser):
                try:
                    if closer is not None:
                        closer.close()
                except Exception:
                    pass
            if self._playwright is not None:
                try:
                    self._playwright.stop()
                except Exception:
                    pass
        return False


def wait_until(predicate, *, timeout_ms: int, interval_ms: int = 250) -> None:
    """Poll ``predicate`` until it returns truthy or the timeout elapses.

    Small helper for subclass flows that need to wait on page state Playwright
    has no first-class locator for.
    """
    deadline = time.monotonic() + (timeout_ms / 1000.0)
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(interval_ms / 1000.0)
    raise TimeoutError(f"condition not met within {timeout_ms}ms")
