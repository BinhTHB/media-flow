"""Unattended publish worker: queue directory -> platforms, zero human input.

    python -m src.interfaces.publish_worker --queue outputs/publish_queue \
        --target file:E:/reup/delivered --target youtube --watch

Targets come from two families:

- **Official API** (``youtube``): credentials from the environment
  (``YOUTUBE_CLIENT_ID`` / ``YOUTUBE_CLIENT_SECRET`` / ``YOUTUBE_REFRESH_TOKEN``,
  obtained once via the OAuth consent screen).
- **Browser session** (``browser:PLATFORM:ACCOUNT``): drives the real web
  upload UI with a stored cookie jar. Sessions live in the session store and
  are created with ``--login``.

Re-upload variation (speed/crop/mirror/pitch) is applied per attempt when
``--variation`` is not ``off``. The worker is idempotent: processed videos are
tracked in ``outputs/publishing/state.json`` and never re-published.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from typing import Optional, Sequence

from src.core.contracts import PublicationPort, SessionStorePort
from src.core.models.account import AccountIdentity
from src.features.publishing.service import PublishService
from src.features.publishing.variation import VariationPolicy
from src.infrastructure.publishing.file_target import FileTargetPublication
from src.infrastructure.publishing.publication_store import JsonPublicationStore
from src.infrastructure.publishing.session_store import FileSessionStore
from src.infrastructure.publishing.youtube import YouTubePublication

logger = logging.getLogger("src.interfaces.publish_worker")

DEFAULT_STATE_PATH = os.path.join("outputs", "publishing", "state.json")
DEFAULT_SESSION_DIR = os.path.join("outputs", "publishing", "sessions")


def build_browser_publisher(
    account: AccountIdentity,
    session_store: SessionStorePort,
    *,
    headless: bool = True,
) -> PublicationPort:
    """Return the session-based publisher for a platform.

    The registry is explicit rather than dynamic: a new platform is a new
    module plus one line here, which keeps import cost (and Playwright) lazy.
    """
    platform = account.platform.lower()
    if platform == "tiktok":
        from src.infrastructure.publishing.tiktok_browser import TikTokBrowserPublisher

        return TikTokBrowserPublisher(account, session_store, headless=headless)
    raise ValueError(
        f"No browser publisher for platform {account.platform!r}; "
        "supported: tiktok"
    )


def build_targets(
    specs: Sequence[str],
    session_store: Optional[SessionStorePort] = None,
    *,
    headless: bool = True,
) -> list[PublicationPort]:
    """Build publication targets from CLI specs.

    ``spec`` forms:
    - ``file:DIR`` — local delivery (offline, deterministic)
    - ``youtube`` — official Data API v3 upload (OAuth refresh token)
    - ``browser:PLATFORM:ACCOUNT`` — session-based web upload; requires a
      stored session for that account (see ``--login``)
    """
    targets: list[PublicationPort] = []
    for spec in specs:
        if spec.startswith("file:"):
            targets.append(FileTargetPublication(spec[len("file:") :]))
        elif spec == "youtube":
            targets.append(
                YouTubePublication(
                    client_id=os.environ["YOUTUBE_CLIENT_ID"],
                    client_secret=os.environ["YOUTUBE_CLIENT_SECRET"],
                    refresh_token=os.environ["YOUTUBE_REFRESH_TOKEN"],
                )
            )
        elif spec.startswith("browser:"):
            targets.append(_build_browser_target(spec, session_store, headless=headless))
        else:
            raise ValueError(
                f"Unknown publish target: {spec!r} "
                "(use 'youtube', 'file:DIR', or 'browser:PLATFORM:ACCOUNT')"
            )
    return targets


def _build_browser_target(
    spec: str,
    session_store: Optional[SessionStorePort],
    *,
    headless: bool,
) -> PublicationPort:
    """Resolve ``browser:PLATFORM:ACCOUNT`` into a session-based publisher."""
    parts = spec.split(":", 2)
    if len(parts) != 3 or not parts[1] or not parts[2]:
        raise ValueError(
            f"Malformed browser target {spec!r}; expected 'browser:PLATFORM:ACCOUNT'"
        )
    _, platform, account_id = parts
    if session_store is None:
        raise ValueError("browser targets require a session store")

    account = AccountIdentity(
        platform=platform,
        account_id=account_id,
        credential_ref=f"{platform.upper()}_{account_id.upper()}_SESSION",
    )
    return build_browser_publisher(account, session_store, headless=headless)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.interfaces.publish_worker",
        description="Publish finished videos to platforms, fully unattended.",
    )
    parser.add_argument("--queue", default=os.path.join("outputs", "publish_queue"),
                        help="Directory watched for new videos (default: outputs/publish_queue)")
    parser.add_argument("--target", action="append", required=True,
                        help="Publication target: 'youtube', 'file:DIR', or "
                             "'browser:PLATFORM:ACCOUNT' (repeatable)")
    parser.add_argument("--state", default=DEFAULT_STATE_PATH,
                        help="Publication state store path")
    parser.add_argument("--sessions", default=DEFAULT_SESSION_DIR,
                        help="Directory holding stored browser sessions")
    parser.add_argument("--variation", default="balanced",
                        choices=["off", "light", "balanced", "aggressive"],
                        help="Re-upload variation level (default: balanced)")
    parser.add_argument("--no-headless", action="store_true",
                        help="Show the browser window (debugging browser targets)")
    parser.add_argument("--watch", action="store_true",
                        help="Keep polling instead of one shot")
    parser.add_argument("--interval", type=float, default=60.0,
                        help="Poll interval in seconds for --watch (default: 60)")
    parser.add_argument("--max-attempts", type=int, default=5)
    parser.add_argument("--backoff-base", type=float, default=30.0)
    parser.add_argument("--no-metadata", action="store_true",
                        help="Disable AI metadata generation; use filename/sidecar only")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = build_parser().parse_args(argv)

    session_store = FileSessionStore(args.sessions)

    try:
        targets = build_targets(
            args.target, session_store, headless=not args.no_headless
        )
    except (ValueError, KeyError) as exc:
        logger.error("Target configuration error: %s", exc)
        return 2

    metadata_generator = None
    if not args.no_metadata and os.getenv("GEMINI_API_KEY"):
        from src.infrastructure.publishing.gemini_metadata import GeminiMetadataGenerator

        metadata_generator = GeminiMetadataGenerator()
        logger.info("Publish: AI metadata generation enabled.")

    variation_provider = None
    if args.variation != "off":
        from src.infrastructure.video.variation import FFmpegVariationAdapter

        variation_provider = FFmpegVariationAdapter()
        logger.info("Publish: variation level '%s' enabled.", args.variation)

    service = PublishService(
        targets=targets,
        metadata_generator=metadata_generator,
        store=JsonPublicationStore(args.state),
        max_attempts=args.max_attempts,
        backoff_base_seconds=args.backoff_base,
        variation_policy=VariationPolicy(level=args.variation) if variation_provider else None,
        variation_provider=variation_provider,
    )

    os.makedirs(args.queue, exist_ok=True)
    while True:
        try:
            records = service.process_queue_dir(args.queue)
        except FileNotFoundError as exc:
            logger.error("Queue directory missing: %s", exc)
            return 2
        failed = [r for r in records if r.status == "FAILED"]
        if records:
            logger.info(
                "Publish: run finished — %d record(s), %d failed.",
                len(records), len(failed),
            )
        if not args.watch:
            return 1 if failed else 0
        time.sleep(max(1.0, args.interval))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
