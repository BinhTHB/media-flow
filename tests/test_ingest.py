"""Ingest tests: fake downloader by default, no network.

The real yt-dlp adapter is exercised only through its pure helpers (format
selection, error mapping, output resolution) so CI never touches a platform.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.core.contracts import TransientIngestError
from src.core.models.ingest import (
    IngestBatch,
    IngestResult,
    SourceSpec,
    infer_platform,
)
from src.features.ingest.service import IngestService, MANIFEST_NAME


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://www.youtube.com/watch?v=abc", "youtube"),
        ("https://youtu.be/abc", "youtube"),
        ("https://www.tiktok.com/@user/video/1", "tiktok"),
        ("https://www.douyin.com/video/1", "douyin"),
        ("https://www.bilibili.com/video/BV1", "bilibili"),
        ("https://www.instagram.com/reel/x", "instagram"),
        ("https://fb.watch/abc", "facebook"),
        ("https://x.com/user/status/1", "twitter"),
        ("https://example.com/clip.mp4", "generic"),
    ],
)
def test_infer_platform(url: str, expected: str) -> None:
    assert infer_platform(url) == expected


def test_source_spec_rejects_bad_input() -> None:
    with pytest.raises(ValueError, match="url"):
        SourceSpec("")
    with pytest.raises(ValueError, match="quality"):
        SourceSpec("https://youtu.be/x", quality="4k")
    with pytest.raises(ValueError, match="trim_seconds"):
        SourceSpec("https://youtu.be/x", trim_seconds=-1)


def test_source_spec_max_height() -> None:
    assert SourceSpec("https://youtu.be/x", quality="best").max_height is None
    assert SourceSpec("https://youtu.be/x", quality="720").max_height == 720


def test_batch_counts() -> None:
    batch = IngestBatch()
    ok = IngestResult(source=SourceSpec("https://youtu.be/a"), status="OK", video_path="a.mp4")
    failed = IngestResult(source=SourceSpec("https://youtu.be/b"), status="FAILED", error="x")
    batch.add(ok)
    batch.add(failed)

    assert batch.ok_count == 1
    assert batch.failed_count == 1
    assert batch.to_dict()["total"] == 2


# --------------------------------------------------------------------------
# Fake downloader
# --------------------------------------------------------------------------

class _FakeDownloader:
    def __init__(
        self,
        *,
        fail_transient: int = 0,
        fail_permanent: bool = False,
        write_file: bool = True,
    ) -> None:
        self.calls: list[SourceSpec] = []
        self._transient = fail_transient
        self._permanent = fail_permanent
        self._write = write_file

    def download(self, source: SourceSpec, output_dir: str) -> IngestResult:
        self.calls.append(source)
        if self._permanent:
            raise RuntimeError("video is private")
        if len(self.calls) <= self._transient:
            raise TransientIngestError("http error 429")

        path = Path(output_dir) / "video.mp4"
        if self._write:
            path.write_bytes(b"downloaded")
        return IngestResult(
            source=source,
            status="OK",
            video_path=str(path),
            title="A clip",
            duration_seconds=12.5,
            width=1080,
            height=1920,
            content_id="abc123",
        )


def _service(downloader, **kwargs) -> IngestService:
    kwargs.setdefault("sleep", lambda _s: None)
    kwargs.setdefault("backoff_base_seconds", 0.0)
    return IngestService(downloader, **kwargs)


# --------------------------------------------------------------------------
# IngestService
# --------------------------------------------------------------------------

def test_ingest_one_downloads_and_writes_manifest(tmp_path) -> None:
    downloader = _FakeDownloader()
    result = _service(downloader).ingest_one(
        SourceSpec("https://youtu.be/abc"), str(tmp_path)
    )

    assert result.ok
    assert result.title == "A clip"
    manifest = json.loads((tmp_path / MANIFEST_NAME).read_text(encoding="utf-8"))
    assert "https://youtu.be/abc" in manifest["sources"]
    assert manifest["sources"]["https://youtu.be/abc"]["status"] == "OK"


def test_second_run_skips_already_downloaded_source(tmp_path) -> None:
    downloader = _FakeDownloader()
    service = _service(downloader)
    source = SourceSpec("https://youtu.be/abc")

    service.ingest_one(source, str(tmp_path))
    second = service.ingest_one(source, str(tmp_path))

    # Only the first run hit the downloader.
    assert len(downloader.calls) == 1
    assert second.status == "SKIPPED"


def test_overwrite_forces_a_fresh_download(tmp_path) -> None:
    downloader = _FakeDownloader()
    service = _service(downloader)
    source = SourceSpec("https://youtu.be/abc")

    service.ingest_one(source, str(tmp_path))
    service.ingest_one(source, str(tmp_path), overwrite=True)

    assert len(downloader.calls) == 2


def test_missing_file_on_disk_triggers_redownload(tmp_path) -> None:
    downloader = _FakeDownloader()
    service = _service(downloader)
    source = SourceSpec("https://youtu.be/abc")

    first = service.ingest_one(source, str(tmp_path))
    Path(first.video_path).unlink()
    second = service.ingest_one(source, str(tmp_path))

    # The manifest said OK but the file vanished: re-fetch rather than
    # reporting a phantom success.
    assert len(downloader.calls) == 2
    assert second.status == "OK"


def test_transient_failure_is_retried_then_succeeds(tmp_path) -> None:
    downloader = _FakeDownloader(fail_transient=2)
    result = _service(downloader, max_attempts=3).ingest_one(
        SourceSpec("https://youtu.be/abc"), str(tmp_path)
    )

    assert result.ok
    assert len(downloader.calls) == 3


def test_transient_failure_exhausts_attempts(tmp_path) -> None:
    downloader = _FakeDownloader(fail_transient=99)
    result = _service(downloader, max_attempts=2).ingest_one(
        SourceSpec("https://youtu.be/abc"), str(tmp_path)
    )

    assert result.status == "FAILED"
    assert "transient" in (result.error or "")
    assert len(downloader.calls) == 2


def test_permanent_failure_is_not_retried(tmp_path) -> None:
    downloader = _FakeDownloader(fail_permanent=True)
    result = _service(downloader, max_attempts=5).ingest_one(
        SourceSpec("https://youtu.be/abc"), str(tmp_path)
    )

    assert result.status == "FAILED"
    assert "permanent" in (result.error or "")
    assert len(downloader.calls) == 1


def test_failure_is_recorded_in_manifest(tmp_path) -> None:
    downloader = _FakeDownloader(fail_permanent=True)
    _service(downloader).ingest_one(SourceSpec("https://youtu.be/abc"), str(tmp_path))

    manifest = json.loads((tmp_path / MANIFEST_NAME).read_text(encoding="utf-8"))
    entry = manifest["sources"]["https://youtu.be/abc"]
    assert entry["status"] == "FAILED"
    assert entry["error"]


def test_batch_processes_every_source(tmp_path) -> None:
    downloader = _FakeDownloader()
    sources = [SourceSpec(f"https://youtu.be/{i}") for i in range(3)]

    batch = _service(downloader).ingest_many(sources, str(tmp_path))

    assert len(batch.results) == 3
    assert batch.ok_count == 3
    assert len(downloader.calls) == 3


def test_corrupt_manifest_does_not_block_ingest(tmp_path) -> None:
    (tmp_path / MANIFEST_NAME).write_text("{not json", encoding="utf-8")
    downloader = _FakeDownloader()

    result = _service(downloader).ingest_one(
        SourceSpec("https://youtu.be/abc"), str(tmp_path)
    )

    assert result.ok
    # The manifest is rewritten into a valid document.
    assert json.loads((tmp_path / MANIFEST_NAME).read_text(encoding="utf-8"))["sources"]


def test_manifest_has_no_temporary_leftovers(tmp_path) -> None:
    _service(_FakeDownloader()).ingest_one(SourceSpec("https://youtu.be/a"), str(tmp_path))
    assert not list(tmp_path.glob("*.tmp"))


# --------------------------------------------------------------------------
# yt-dlp adapter (pure helpers only)
# --------------------------------------------------------------------------

def test_format_selector_caps_height() -> None:
    from src.infrastructure.ingest.yt_dlp import YtDlpDownloader

    selector = YtDlpDownloader._format_selector(SourceSpec("https://youtu.be/x", quality="720"))
    assert "height<=720" in selector


def test_format_selector_best_has_no_cap() -> None:
    from src.infrastructure.ingest.yt_dlp import YtDlpDownloader

    assert YtDlpDownloader._format_selector(
        SourceSpec("https://youtu.be/x", quality="best")
    ) == "bv*+ba/b"


@pytest.mark.parametrize(
    ("message", "transient"),
    [
        ("HTTP Error 429: Too Many Requests", True),
        ("Connection timed out", True),
        ("HTTP Error 503: Service Unavailable", True),
        ("This video is private", False),
        ("Video unavailable", False),
        ("Unsupported URL", False),
        ("something nobody classified", False),
    ],
)
def test_download_error_mapping(message: str, transient: bool) -> None:
    from src.infrastructure.ingest.yt_dlp import _map_download_error

    error = _map_download_error("https://youtu.be/x", message)
    assert isinstance(error, TransientIngestError) is transient


def test_resolve_output_finds_merged_extension(tmp_path) -> None:
    from src.infrastructure.ingest.yt_dlp import _resolve_output

    guessed = tmp_path / "abc.webm"
    merged = tmp_path / "abc.mp4"
    merged.write_bytes(b"x")

    assert _resolve_output(guessed, tmp_path) == merged


def test_find_subtitle_picks_sidecar(tmp_path) -> None:
    from src.infrastructure.ingest.yt_dlp import _find_subtitle

    video = tmp_path / "abc.mp4"
    video.write_bytes(b"x")
    (tmp_path / "abc.vi.srt").write_text("1", encoding="utf-8")

    assert _find_subtitle(video).endswith(".srt")


def test_watch_file_keeps_visual_and_audio_flags() -> None:
    from src.features.ingest.watch import parse_watch_text

    sources = parse_watch_text(
        "\n".join([
            "# channel, picture only",
            "https://www.youtube.com/@name/videos visual=1 audio=0",
            "https://www.youtube.com/playlist?list=PL visual=0 audio=1",
            "https://youtu.be/abc",
        ])
    )

    assert [(item.visual, item.audio) for item in sources] == [(True, False), (False, True), (None, None)]


def test_watch_file_rejects_a_bad_flag() -> None:
    from src.features.ingest.watch import parse_watch_text

    with pytest.raises(ValueError, match="không hợp lệ"):
        parse_watch_text("https://youtu.be/abc visual=maybe")


def test_plan_downloads_skips_urls_already_on_disk() -> None:
    from src.features.ingest.watch import plan_downloads

    planned = plan_downloads(
        [
            "https://youtu.be/old",
            "https://youtu.be/new",
            "https://youtu.be/new",
        ],
        ["https://youtu.be/old"],
        quality="720",
        note="https://www.youtube.com/@name/videos",
    )

    assert [item.url for item in planned] == ["https://youtu.be/new"]
    assert planned[0].quality == "720"
    assert planned[0].note.endswith("/videos")


def test_known_ok_urls_ignore_a_missing_file(tmp_path) -> None:
    service = _service(_FakeDownloader())
    service.ingest_one(SourceSpec("https://youtu.be/abc"), str(tmp_path))
    assert service.known_ok_urls(str(tmp_path)) == {"https://youtu.be/abc"}
    (tmp_path / "video.mp4").unlink()
    assert service.known_ok_urls(str(tmp_path)) == set()


def test_flat_playlist_ids_become_watch_urls() -> None:
    from src.infrastructure.ingest.yt_dlp import entry_urls

    info = {
        "entries": [
            {"id": "aaa"},
            {"webpage_url": "https://www.youtube.com/watch?v=bbb"},
            {"id": "ccc"},
        ]
    }
    assert entry_urls(info, "https://www.youtube.com/@name/videos", limit=2) == [
        "https://www.youtube.com/watch?v=aaa",
        "https://www.youtube.com/watch?v=bbb",
    ]


def test_source_line_sets_reup_stages() -> None:
    from argparse import Namespace

    from src.features.ingest.watch import WatchSource
    from src.interfaces.ingest import _actions

    args = Namespace(subtitle_action="blur", extra_text_action="preserve", dub_audio=False)
    assert _actions(WatchSource("https://youtu.be/a", visual=True, audio=False), args) == (
        "translate", "blur", False,
    )
    assert _actions(WatchSource("https://youtu.be/b", visual=False, audio=True), args) == (
        "preserve", "preserve", True,
    )
    assert _actions(WatchSource("https://youtu.be/c"), args) == ("blur", "preserve", False)
    reup = Namespace(preset="reup", subtitle_action="blur", extra_text_action="preserve", dub_audio=False)
    assert _actions(WatchSource("https://youtu.be/d"), reup) == ("translate", "blur", True)
    assert _actions(WatchSource("https://youtu.be/e", visual=False, audio=False), reup) == (
        "preserve", "preserve", False,
    )


def test_then_process_still_plans_a_video_the_manifest_already_has(tmp_path) -> None:
    from argparse import Namespace

    from src.interfaces.ingest import _collect_plans

    source = tmp_path / "sources.txt"
    source.write_text("https://www.youtube.com/shorts/aaa visual=1 audio=0\n", encoding="utf-8")
    watch_url = "https://www.youtube.com/watch?v=aaa"

    class Catalog:
        def list_entries(self, url, *, limit):
            return [watch_url]

    class Ledger:
        def known_ok_urls(self, output_dir):
            return {watch_url}

    args = Namespace(
        url=[],
        urls_file=None,
        sources=str(source),
        limit=1,
        overwrite=False,
        then_process=True,
        quality="1080",
        subtitles=False,
        trim=0.0,
    )
    plans = _collect_plans(args, Catalog(), Ledger())
    assert [spec.url for spec, _watch in plans] == [watch_url]


def test_then_process_keeps_watch_flags_and_retries_a_skipped_file(tmp_path, monkeypatch) -> None:
    from argparse import Namespace

    from src.core.models.ingest import IngestBatch, IngestResult, SourceSpec
    from src.interfaces.ingest import _process_downloads
    from src.interfaces.jobs import runner as runner_module

    fresh = tmp_path / "fresh.mp4"
    skipped = tmp_path / "skipped.mp4"
    fresh.write_bytes(b"mp4")
    skipped.write_bytes(b"mp4")
    seen: list = []

    class FakeRunner:
        def run(self, options, on_event=None):
            seen.append((
                options.input_path,
                options.subtitle_action,
                options.extra_text_action,
                options.dub_audio,
                options.output_path.replace("\\", "/"),
            ))

    monkeypatch.setattr(runner_module, "JobRunner", FakeRunner)
    args = Namespace(preset="reup", subtitle_action="translate", dub_audio=False, device="cpu", overwrite=False)
    fresh_url = "https://www.youtube.com/watch?v=fresh"
    skipped_url = "https://www.youtube.com/watch?v=skipped"
    batch = IngestBatch(results=[
        IngestResult(source=SourceSpec(url=fresh_url), status="OK", video_path=str(fresh)),
        IngestResult(source=SourceSpec(url=skipped_url), status="SKIPPED", video_path=str(skipped)),
    ])
    code = _process_downloads(batch, args, {
        fresh_url: ("translate", "blur", False),
        skipped_url: ("translate", "blur", False),
    })

    assert code == 0
    assert [item[:4] for item in seen] == [
        (str(fresh), "translate", "blur", False),
        (str(skipped), "translate", "blur", False),
    ]
    assert seen[0][4].endswith("outputs/cli/fresh-reup.mp4")
    assert seen[1][4].endswith("outputs/cli/skipped-reup.mp4")
    assert str(tmp_path).replace("\\", "/") not in seen[0][4]
