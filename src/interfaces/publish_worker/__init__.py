"""Unattended publish worker: queue directory -> platforms, zero human input.

    python -m src.interfaces.publish_worker --queue outputs/publish_queue \
        --target file:E:/reup/delivered --target youtube --watch

Targets are wired from the environment (YouTube: YOUTUBE_CLIENT_ID /
YOUTUBE_CLIENT_SECRET / YOUTUBE_REFRESH_TOKEN — obtained once via the OAuth
consent screen; see docs/PUBLISHING.md). The worker is idempotent: processed
videos are tracked in outputs/publishing/state.json and never re-published.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from typing import Optional, Sequence

from src.core.contracts import PublicationPort
from src.features.publishing.service import PublishService
from src.infrastructure.publishing.file_target import FileTargetPublication
from src.infrastructure.publishing.publication_store import JsonPublicationStore
from src.infrastructure.publishing.youtube import YouTubePublication

logger = logging.getLogger("src.interfaces.publish_worker")

DEFAULT_STATE_PATH = os.path.join("outputs", "publishing", "state.json")


def build_targets(specs: Sequence[str]) -> list[PublicationPort]:
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
        else:
            raise ValueError(f"Unknown publish target: {spec!r} (use 'youtube' or 'file:DIR')")
    return targets


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.interfaces.publish_worker",
        description="Publish finished videos to platforms, fully unattended.",
    )
    parser.add_argument("--queue", default=os.path.join("outputs", "publish_queue"),
                        help="Directory watched for new videos (default: outputs/publish_queue)")
    parser.add_argument("--target", action="append", required=True,
                        help="Publication target: 'youtube' or 'file:DIR' (repeatable)")
    parser.add_argument("--state", default=DEFAULT_STATE_PATH,
                        help="Publication state store path")
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

    try:
        targets = build_targets(args.target)
    except (ValueError, KeyError) as exc:
        logger.error("Target configuration error: %s", exc)
        return 2

    metadata_generator = None
    if not args.no_metadata and os.getenv("GEMINI_API_KEY"):
        from src.infrastructure.publishing.gemini_metadata import GeminiMetadataGenerator

        metadata_generator = GeminiMetadataGenerator()
        logger.info("Publish: AI metadata generation enabled.")

    service = PublishService(
        targets=targets,
        metadata_generator=metadata_generator,
        store=JsonPublicationStore(args.state),
        max_attempts=args.max_attempts,
        backoff_base_seconds=args.backoff_base,
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
