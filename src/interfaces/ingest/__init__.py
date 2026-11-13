"""Ingest CLI: turn a list of URLs into local videos.

    python -m src.interfaces.ingest --url https://youtu.be/xxx --url ...
    python -m src.interfaces.ingest --urls-file links.txt --out input/downloads
    python -m src.interfaces.ingest --url ... --then-process   # chain into the pipeline

``--then-process`` runs the normal MediaFlow pipeline over each downloaded file,
so one command goes from URL to finished video.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import List, Optional, Sequence

from src.core.models.ingest import SourceSpec
from src.features.ingest.service import IngestService
from src.features.ingest.watch import WatchSource, parse_watch_text, plan_downloads
from src.infrastructure.ingest.yt_dlp import YtDlpDownloader

logger = logging.getLogger("src.interfaces.ingest")

DEFAULT_OUT_DIR = "input/downloads"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.interfaces.ingest",
        description="Download remote videos into the MediaFlow workspace.",
    )
    parser.add_argument("--url", action="append", default=[],
                        help="Source URL (repeatable)")
    parser.add_argument("--urls-file", default=None,
                        help="Text file with one URL per line ('#' comments allowed)")
    parser.add_argument("--sources", default=None,
                        help="Watch file: one channel, playlist, or video URL per line. "
                             "Optional visual=1|0 and audio=1|0.")
    parser.add_argument("--limit", type=int, default=10,
                        help="Newest entries to inspect per source (default: 10)")
    parser.add_argument("--interval", type=int, default=0,
                        help="Poll again after this many seconds (0 = once)")
    parser.add_argument("--out", default=DEFAULT_OUT_DIR,
                        help=f"Download directory (default: {DEFAULT_OUT_DIR})")
    parser.add_argument("--quality", default="1080",
                        choices=["best", "1080", "720", "480"],
                        help="Height cap (default: 1080)")
    parser.add_argument("--trim", type=float, default=0.0,
                        help="Keep only the first N seconds (0 = whole video)")
    parser.add_argument("--subtitles", action="store_true",
                        help="Download available subtitles when the source has them")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-download sources already present in the manifest")
    parser.add_argument("--cookie-file", default=None,
                        help="Netscape cookie file for authenticated sources")
    parser.add_argument("--cookies-from-browser", default=None,
                        help="Read cookies from a browser profile (e.g. chrome)")
    parser.add_argument("--proxy", default=None, help="Proxy URL for the downloader")
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--backoff-base", type=float, default=5.0)
    parser.add_argument("--json", action="store_true",
                        help="Print the batch result as JSON instead of a summary")
    parser.add_argument("--then-process", action="store_true",
                        help="Run the MediaFlow pipeline over each downloaded video")
    parser.add_argument("--preset", default="yolo-4class",
                        help="Pipeline preset for --then-process")
    parser.add_argument("--subtitle-action", default="translate",
                        help="Subtitle action for --then-process")
    parser.add_argument("--extra-text-action", default="preserve",
                        choices=["blur", "preserve"],
                        help="Watermark action for --then-process when a source line omits visual=")
    parser.add_argument("--dub-audio", action="store_true",
                        help="Enable dubbing for --then-process")
    parser.add_argument("--device", default="auto")
    return parser


def read_urls(args: argparse.Namespace) -> List[str]:
    urls: List[str] = list(args.url or [])
    if args.urls_file:
        path = Path(args.urls_file)
        if not path.is_file():
            raise FileNotFoundError(args.urls_file)
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            urls.append(line)
    # Preserve order but drop duplicates: the same URL twice in one run is
    # always a mistake and would just hit the manifest skip path.
    seen: set[str] = set()
    unique: List[str] = []
    for url in urls:
        if url not in seen:
            seen.add(url)
            unique.append(url)
    return unique


def main(argv: Optional[Sequence[str]] = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    from src.interfaces.web.server import load_project_env

    load_project_env()
    args = build_parser().parse_args(argv)
    if args.limit < 1:
        logger.error("--limit must be at least 1.")
        return 2
    if args.interval < 0:
        logger.error("--interval must not be negative.")
        return 2

    downloader = YtDlpDownloader(
        cookie_file=args.cookie_file,
        cookies_from_browser=args.cookies_from_browser,
        proxy=args.proxy,
    )
    service = IngestService(
        downloader,
        max_attempts=args.max_attempts,
        backoff_base_seconds=args.backoff_base,
    )

    while True:
        code = _poll_once(args, downloader, service)
        if args.interval <= 0 or code == 2:
            return code
        logger.info("Ingest: next poll in %ss.", args.interval)
        time.sleep(args.interval)


def _poll_once(args: argparse.Namespace, downloader: YtDlpDownloader, service: IngestService) -> int:
    try:
        plans = _collect_plans(args, downloader, service)
    except FileNotFoundError as exc:
        logger.error("Source file not found: %s", exc)
        return 2
    except ValueError as exc:
        logger.error("%s", exc)
        return 2
    except RuntimeError as exc:
        logger.error("Ingest: %s", exc)
        return 1
    if plans is None:
        logger.error("No URLs given. Use --url, --urls-file, or --sources.")
        return 2

    batch = service.ingest_many([spec for spec, _watch in plans], args.out, overwrite=args.overwrite)
    _print_batch(batch, json_output=args.json)

    if args.then_process:
        actions = {spec.url: _actions(watch, args) for spec, watch in plans}
        return _process_downloads(batch, args, actions)
    return 1 if batch.failed_count else 0


def _collect_plans(args, downloader: YtDlpDownloader, service: IngestService):
    """Return ``(SourceSpec, WatchSource)`` pairs, or None when nothing was asked."""
    plans: List[tuple[SourceSpec, WatchSource]] = []
    direct = read_urls(args)
    for url in direct:
        plans.append((
            SourceSpec(
                url=url,
                quality=args.quality,
                fetch_subtitles=args.subtitles,
                trim_seconds=args.trim,
            ),
            WatchSource(url=url),
        ))

    if not args.sources:
        return plans or None

    watches = parse_watch_text(Path(args.sources).read_text(encoding="utf-8"))
    if not watches and not plans:
        return None
    # A process retry must still see videos the manifest already holds.
    # Download itself stays skipped; only a missing output file is rendered.
    known = set() if args.overwrite or args.then_process else service.known_ok_urls(args.out)
    list_errors = 0
    for watch in watches:
        try:
            entries = downloader.list_entries(watch.url, limit=args.limit)
        except Exception as exc:
            list_errors += 1
            logger.error("Ingest: could not list %s: %s", watch.url, exc)
            continue
        for spec in plan_downloads(
            entries,
            known,
            quality=args.quality,
            fetch_subtitles=args.subtitles,
            trim_seconds=args.trim,
            note=watch.url,
        ):
            plans.append((spec, watch))
    if not plans and list_errors:
        raise RuntimeError(f"{list_errors} source(s) could not be listed")
    return plans


def _rendered_output(source: Path, preset: str) -> Path:
    """Finished videos go to outputs/cli, not next to the downloaded source."""
    from src.interfaces.cli import DEFAULT_OUTPUT_DIR

    return DEFAULT_OUTPUT_DIR / f"{source.stem}-{preset}.mp4"


def _actions(watch: WatchSource, args: argparse.Namespace) -> tuple[str, str, bool]:
    reup = str(getattr(args, "preset", "")).removesuffix("-generic") == "reup"
    if watch.visual is True or (watch.visual is None and reup):
        subtitle, extra = "translate", "blur"
    elif watch.visual is False:
        subtitle, extra = "preserve", "preserve"
    else:
        subtitle, extra = args.subtitle_action, args.extra_text_action
    if watch.audio is None:
        audio = True if reup else bool(args.dub_audio)
    else:
        audio = watch.audio
    return subtitle, extra, bool(audio)


def _print_batch(batch, *, json_output: bool) -> None:
    if json_output:
        print(json.dumps(batch.to_dict(), ensure_ascii=False, indent=2))
        return
    for result in batch.results:
        if result.ok:
            print(f"OK      {result.video_path}  ({result.width}x{result.height}, "
                  f"{result.duration_seconds:.1f}s)")
        elif result.status == "SKIPPED":
            print(f"SKIPPED {result.video_path}")
        else:
            print(f"FAILED  {result.source.url}: {result.error}")
    print(f"\n{batch.ok_count}/{len(batch.results)} downloaded, {batch.failed_count} failed")


def _should_process(result, args: argparse.Namespace) -> bool:
    """Process a fresh download, or a skipped one whose output is not on disk yet."""
    if not result.video_path or result.status not in {"OK", "SKIPPED"}:
        return False
    if result.status != "SKIPPED":
        return True
    source = Path(result.video_path)
    output = _rendered_output(source, args.preset)
    return args.overwrite or not output.is_file()


def _process_downloads(batch, args: argparse.Namespace, actions: dict[str, tuple[str, str, bool]]) -> int:
    """Chain downloaded files into the standard pipeline."""
    from src.interfaces.jobs.runner import JobOptions, JobRunner, format_event

    videos = [r for r in batch.results if _should_process(r, args)]
    if not videos:
        logger.error("Nothing downloaded; skipping --then-process.")
        return 1

    runner = JobRunner()
    failures = 0
    for result in videos:
        source = Path(result.video_path)
        output = _rendered_output(source, args.preset)
        chosen = actions.get(result.source.url)
        if chosen is None:
            subtitle = args.subtitle_action
            extra = args.extra_text_action
            dub = bool(args.dub_audio)
        else:
            subtitle, extra, dub = chosen
        options = JobOptions(
            input_path=str(source),
            output_path=str(output),
            preset=args.preset,
            subtitle_action=subtitle,
            extra_text_action=extra,
            dub_audio=dub,
            device=args.device,
            overwrite=True,
        )
        try:
            runner.run(options, on_event=lambda event: print(format_event(event), flush=True))
        except Exception as exc:
            failures += 1
            logger.error("Pipeline failed for %s: %s", result.video_path, exc)
    return 1 if failures else 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
