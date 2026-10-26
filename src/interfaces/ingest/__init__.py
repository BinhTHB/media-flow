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
from pathlib import Path
from typing import List, Optional, Sequence

from src.core.models.ingest import SourceSpec
from src.features.ingest.service import IngestService
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
    args = build_parser().parse_args(argv)

    try:
        urls = read_urls(args)
    except FileNotFoundError as exc:
        logger.error("URLs file not found: %s", exc)
        return 2
    if not urls:
        logger.error("No URLs given. Use --url or --urls-file.")
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

    sources = [
        SourceSpec(
            url=url,
            quality=args.quality,
            fetch_subtitles=args.subtitles,
            trim_seconds=args.trim,
        )
        for url in urls
    ]
    batch = service.ingest_many(sources, args.out, overwrite=args.overwrite)

    if args.json:
        print(json.dumps(batch.to_dict(), ensure_ascii=False, indent=2))
    else:
        for result in batch.results:
            if result.ok:
                print(f"OK      {result.video_path}  ({result.width}x{result.height}, "
                      f"{result.duration_seconds:.1f}s)")
            elif result.status == "SKIPPED":
                print(f"SKIPPED {result.video_path}")
            else:
                print(f"FAILED  {result.source.url}: {result.error}")
        print(f"\n{batch.ok_count}/{len(batch.results)} downloaded, {batch.failed_count} failed")

    if args.then_process:
        return _process_downloads(batch, args)

    return 1 if batch.failed_count else 0


def _process_downloads(batch, args: argparse.Namespace) -> int:
    """Chain downloaded files into the standard pipeline."""
    from src.interfaces.jobs.runner import JobOptions, JobRunner, format_event

    videos = [r.video_path for r in batch.results if r.ok and r.video_path]
    if not videos:
        logger.error("Nothing downloaded; skipping --then-process.")
        return 1

    runner = JobRunner()
    failures = 0
    for video in videos:
        source = Path(video)
        output = source.with_name(f"{source.stem}-{args.preset}.mp4")
        options = JobOptions(
            input_path=str(source),
            output_path=str(output),
            preset=args.preset,
            subtitle_action=args.subtitle_action,
            dub_audio=bool(args.dub_audio),
            device=args.device,
            overwrite=True,
        )
        try:
            runner.run(options, on_event=lambda event: print(format_event(event), flush=True))
        except Exception as exc:
            failures += 1
            logger.error("Pipeline failed for %s: %s", video, exc)
    return 1 if failures else 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
