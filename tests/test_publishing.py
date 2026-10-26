"""Publishing feature tests: mock providers by default; the YouTube provider
is exercised against a fake HTTP session so no network is touched."""

from __future__ import annotations

import json

import pytest

from src.core.contracts import TransientPublishError
from src.core.models.publication import (
    PublishItem,
    PublishRecord,
    VideoMetadata,
    item_id_for,
)
from src.features.publishing.service import PublishService
from src.infrastructure.publishing.file_target import FileTargetPublication
from src.infrastructure.publishing.publication_store import JsonPublicationStore


class _FakeTarget:
    def __init__(self, name: str = "fake", failures_before_success: int = 0, permanent: bool = False):
        self.platform_name = name
        self.calls = 0
        self.items: list[PublishItem] = []
        self._failures = failures_before_success
        self._permanent = permanent

    def publish(self, item: PublishItem) -> PublishRecord:
        self.calls += 1
        self.items.append(item)
        if self._permanent:
            raise RuntimeError("hard rejection")
        if self.calls <= self._failures:
            raise TransientPublishError("rate limited")
        return PublishRecord(
            item_id=item.item_id, video_path=item.video_path, platform=self.platform_name,
            status="PUBLISHED", platform_video_id=f"fake-{self.calls}", url="https://x/fake",
        )


class _FakeStore:
    def __init__(self):
        self.records: list[PublishRecord] = []
        self.published: set[tuple[str, str]] = set()

    def is_published(self, item_id, platform):
        return (item_id, platform) in self.published

    def add(self, record):
        self.records.append(record)
        if record.status == "PUBLISHED":
            self.published.add((record.item_id, record.platform))


def _video(tmp_path, name="clip.mp4"):
    path = tmp_path / name
    path.write_bytes(b"fake-video-bytes")
    return str(path)


def test_publish_file_records_one_published_record_per_target(tmp_path):
    a, b = _FakeTarget("fake-a"), _FakeTarget("fake-b")
    store = _FakeStore()
    service = PublishService(targets=[a, b], store=store)
    records = service.publish_file(_video(tmp_path))

    assert [r.status for r in records] == ["PUBLISHED", "PUBLISHED"]
    assert a.calls == 1 and b.calls == 1
    assert len(store.records) == 2
    assert store.is_published(item_id_for(records[0].video_path), "fake-a")


def test_store_prevents_duplicate_publish_on_second_run(tmp_path):
    target = _FakeTarget()
    service = PublishService(targets=[target], store=_FakeStore())
    service.publish_file(_video(tmp_path))
    second = service.publish_file(_video(tmp_path))

    assert target.calls == 1
    assert second[0].error == "duplicate-skipped"


def test_transient_failures_retry_with_exponential_backoff(tmp_path):
    target = _FakeTarget(failures_before_success=2)
    sleeps: list[float] = []
    service = PublishService(
        targets=[target], store=_FakeStore(),
        backoff_base_seconds=30.0, sleep=sleeps.append,
    )
    records = service.publish_file(_video(tmp_path))

    assert sleeps == [30.0, 60.0]
    assert records[0].status == "PUBLISHED"
    assert records[0].attempts == 3


def test_permanent_failure_marks_failed_without_retry(tmp_path):
    target = _FakeTarget(permanent=True)
    service = PublishService(targets=[target], store=_FakeStore())
    records = service.publish_file(_video(tmp_path))

    assert target.calls == 1
    assert records[0].status == "FAILED"
    assert "permanent" in records[0].error


def test_sidecar_overrides_generated_metadata(tmp_path):
    video = _video(tmp_path)
    from pathlib import Path

    Path(video).with_suffix(".publish.json").write_text(
        json.dumps({"title": "Tiêu đề từ sidecar", "privacy": "public", "tags": ["reup"]}),
        encoding="utf-8",
    )
    target = _FakeTarget()
    PublishService(targets=[target], store=_FakeStore()).publish_file(video)

    meta = target.items[0].metadata
    assert meta.title == "Tiêu đề từ sidecar"
    assert meta.privacy == "public"
    assert meta.tags == ["reup"]


def test_queue_dir_is_idempotent_across_runs(tmp_path):
    queue = tmp_path / "queue"
    queue.mkdir()
    (queue / "a.mp4").write_bytes(b"aaa")
    target = _FakeTarget()
    service = PublishService(targets=[target], store=_FakeStore())

    first = service.process_queue_dir(str(queue))
    second = service.process_queue_dir(str(queue))

    assert [r.status for r in first] == ["PUBLISHED"]
    assert second == []
    assert target.calls == 1


def test_file_target_copies_video_and_writes_sidecar(tmp_path):
    video = _video(tmp_path, "movie.mp4")
    dest_dir = tmp_path / "delivered"
    target = FileTargetPublication(str(dest_dir))

    item = PublishItem(
        item_id=item_id_for(video), video_path=video, platform="file",
        metadata=VideoMetadata(title="Phim dịch", tags=["vi"]),
    )
    record = target.publish(item)

    assert record.status == "PUBLISHED"
    assert (dest_dir / "movie.mp4").read_bytes() == b"fake-video-bytes"
    sidecar = json.loads((dest_dir / "movie.publish.json").read_text(encoding="utf-8"))
    assert sidecar["title"] == "Phim dịch"


def test_json_store_persists_across_instances(tmp_path):
    state = str(tmp_path / "state.json")
    store = JsonPublicationStore(state)
    store.add(PublishRecord(item_id="i1", video_path="v.mp4", platform="youtube", status="PUBLISHED"))

    reloaded = JsonPublicationStore(state)
    assert reloaded.is_published("i1", "youtube")
    assert not reloaded.is_published("i1", "file:other")
    assert not reloaded.is_published("i2", "youtube")


def test_json_store_survives_corrupt_state_file(tmp_path):
    state = tmp_path / "state.json"
    state.write_text("{not json", encoding="utf-8")

    store = JsonPublicationStore(str(state))
    assert not store.is_published("any", "youtube")


# ---------------------------------------------------------------- YouTube ---

class _Resp:
    def __init__(self, status_code, headers=None, text="", payload=None):
        self.status_code = status_code
        self.headers = headers or {}
        self.text = text
        self._payload = payload or {}

    def json(self):
        return self._payload


class _FakeSession:
    def __init__(self):
        self.posts = []
        self.puts = []

    def post(self, url, data=None, headers=None, params=None, timeout=None):
        self.posts.append({"url": url, "data": data, "headers": headers, "params": params})
        if "oauth2.googleapis.com" in url:
            return _Resp(200, payload={"access_token": "tok123", "expires_in": 3600})
        return _Resp(200, headers={"Location": "https://upload/session/1"})

    def put(self, url, headers=None, data=None, timeout=None):
        self.puts.append({"url": url, "headers": headers, "data": data})
        return _Resp(200, payload={"id": "vidXYZ"})


def _youtube_provider(session):
    from src.infrastructure.publishing.youtube import YouTubePublication

    return YouTubePublication("cid", "csecret", "rtoken", session=session)


def _yt_item(tmp_path, **meta):
    video = _video(tmp_path, "short.mp4")
    return PublishItem(
        item_id=item_id_for(video), video_path=video, platform="youtube",
        metadata=VideoMetadata(title="Tiêu đề test", privacy="public", **meta),
    )


def test_youtube_provider_resumable_flow(tmp_path):
    session = _FakeSession()
    record = _youtube_provider(session).publish(_yt_item(tmp_path))

    assert record.status == "PUBLISHED"
    assert record.platform_video_id == "vidXYZ"
    assert record.url == "https://youtu.be/vidXYZ"

    token_call, session_call = session.posts
    assert token_call["data"]["grant_type"] == "refresh_token"
    assert session_call["params"] == {"uploadType": "resumable", "part": "snippet,status"}
    body = json.loads(session_call["data"])
    assert body["snippet"]["title"] == "Tiêu đề test"
    assert body["status"]["privacyStatus"] == "public"
    assert body["status"]["selfDeclaredMadeForKids"] is False

    (put,) = session.puts
    assert put["url"] == "https://upload/session/1"
    assert put["headers"]["Content-Range"] == "bytes 0-15/16"
    assert put["data"] == b"fake-video-bytes"


def test_youtube_429_is_transient(tmp_path, monkeypatch):
    session = _FakeSession()
    monkeypatch.setattr(
        session, "put",
        lambda url, **kw: _Resp(429, text="too many requests"),
    )
    with pytest.raises(TransientPublishError):
        _youtube_provider(session).publish(_yt_item(tmp_path))


def test_youtube_quota_exceeded_is_transient(tmp_path, monkeypatch):
    session = _FakeSession()
    monkeypatch.setattr(
        session, "put",
        lambda url, **kw: _Resp(403, text='{"error": "quotaExceeded"}'),
    )
    with pytest.raises(TransientPublishError):
        _youtube_provider(session).publish(_yt_item(tmp_path))


# ------------------------------------------------------------- AI metadata --

class _FakeGeminiClient:
    def __init__(self, text):
        self._text = text

    def generate_content(self, request):
        class _R:
            pass

        response = _R()
        response.text = self._text
        return response


class _FakeRuntime:
    def __init__(self, text):
        self._text = text

    def get_client(self, model=None):
        return _FakeGeminiClient(self._text)


def test_metadata_generator_parses_fenced_json(tmp_path):
    from src.infrastructure.publishing.gemini_metadata import GeminiMetadataGenerator

    fenced = '```json\n{"title": "Bản dịch hay", "description": "Mô tả", "tags": ["one piece"]}\n```'
    generator = GeminiMetadataGenerator(runtime=_FakeRuntime(fenced))
    meta = generator.generate(_video(tmp_path, "one_piece_ep1.mp4"))

    assert meta.title == "Bản dịch hay"
    assert meta.tags == ["one piece"]


def test_metadata_generator_falls_back_on_model_error(tmp_path):
    from src.infrastructure.publishing.gemini_metadata import GeminiMetadataGenerator

    class _BrokenRuntime:
        def get_client(self, model=None):
            raise RuntimeError("no api key")

    meta = GeminiMetadataGenerator(runtime=_BrokenRuntime()).generate(
        _video(tmp_path, "clip_xy.mp4")
    )
    assert meta.title == "clip_xy"
