"""One-time browser login helper: create a stored session for a publisher.

Re-upload automation needs a logged-in browser profile, and a human has to do
that login once (password, 2FA, captcha). This helper opens a real browser,
waits for the operator to finish logging in, then writes the cookie jar into
the session store.

    python -m src.interfaces.publish_login --platform tiktok --account acc1

The resulting session file is a credential: it lives under the gitignored
session directory and is never printed.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from typing import Optional, Sequence

from src.core.models.account import AccountIdentity
from src.infrastructure.publishing.session_store import FileSessionStore

logger = logging.getLogger("src.interfaces.publish_login")

DEFAULT_SESSION_DIR = "outputs/publishing/sessions"

#: Where each platform's login lands, so the helper can tell the operator what
#: to expect and can detect a successful login.
LOGIN_URLS = {
    "tiktok": "https://www.tiktok.com/login",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.interfaces.publish_login",
        description="Create a stored browser session for a publication target.",
    )
    parser.add_argument("--platform", required=True, choices=sorted(LOGIN_URLS))
    parser.add_argument("--account", required=True, help="Account label, e.g. acc1")
    parser.add_argument("--sessions", default=DEFAULT_SESSION_DIR,
                        help=f"Session directory (default: {DEFAULT_SESSION_DIR})")
    parser.add_argument("--timeout", type=float, default=300.0,
                        help="Seconds to wait for a completed login (default: 300)")
    parser.add_argument("--poll", type=float, default=2.0,
                        help="Login-state poll interval in seconds (default: 2)")
    return parser


def login(
    platform: str,
    account_id: str,
    session_dir: str,
    *,
    timeout_seconds: float = 300.0,
    poll_seconds: float = 2.0,
) -> str:
    """Open a browser, wait for a human login, persist the session."""
    try:
        from playwright.sync_api import sync_playwright
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "playwright is required for the login helper. Install it with "
            "`pip install playwright` and then `playwright install chromium`."
        ) from exc

    url = LOGIN_URLS.get(platform)
    if not url:
        raise ValueError(f"No login URL for platform {platform!r}")

    store = FileSessionStore(session_dir)
    account = AccountIdentity(
        platform=platform,
        account_id=account_id,
        credential_ref=f"{platform.upper()}_{account_id.upper()}_SESSION",
    )

    staging = store._root / f".login-{_safe(account.key)}.json"  # noqa: SLF001
    store._root.mkdir(parents=True, exist_ok=True)  # noqa: SLF001

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=False)
        context = browser.new_context()
        page = context.new_page()
        page.goto(url, wait_until="domcontentloaded")
        print(f"Log in to {platform} as '{account_id}' in the opened window.")
        print(f"Waiting up to {timeout_seconds:.0f}s for a completed login...")

        deadline = time.monotonic() + timeout_seconds
        logged_in = False
        while time.monotonic() < deadline:
            if _looks_logged_in(page):
                logged_in = True
                break
            time.sleep(max(0.5, poll_seconds))

        if not logged_in:
            context.close()
            browser.close()
            raise TimeoutError(
                f"no login detected within {timeout_seconds:.0f}s; nothing was saved"
            )

        context.storage_state(path=str(staging))
        context.close()
        browser.close()

    saved = store.save(account, str(staging))
    staging.unlink(missing_ok=True)
    print(f"Session stored for {account.key}.")
    return saved


def _looks_logged_in(page) -> bool:
    """A login wall still showing a password field means "not yet"."""
    try:
        return page.locator("input[type=password]").count() == 0
    except Exception:
        return False


def _safe(key: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in key)


def main(argv: Optional[Sequence[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = build_parser().parse_args(argv)
    try:
        login(
            args.platform, args.account, args.sessions,
            timeout_seconds=args.timeout, poll_seconds=args.poll,
        )
    except (RuntimeError, ValueError, TimeoutError) as exc:
        logger.error("Login failed: %s", exc)
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
