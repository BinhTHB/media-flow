"""Zernio account rows and the post body sent for one platform key."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.infrastructure.publishing.zernio import ZernioClient
from src.infrastructure.publishing.zernio_store import ZernioAccountStore
from src.interfaces.web.server import RequestError, start_source_poll


class _Response:
    def __init__(self, body: str) -> None:
        self._body = body.encode("utf-8")

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, *args: object) -> bool:
        return False


def test_account_store_masks_the_key(tmp_path: Path) -> None:
    store = ZernioAccountStore(tmp_path / "zernio_accounts.json")
    created = store.add(label="YouTube chính", platform="youtube", api_key="zernio-secret-key")
    assert created["api_key_masked"].endswith("key")
    assert "secret" not in created["api_key_masked"]
    assert store.get(created["id"])["api_key"] == "zernio-secret-key"
    assert store.remove(created["id"]) is True
    assert store.list_public() == []


def test_publish_uses_the_matching_connected_account(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"video-bytes")
    seen: list[tuple[str, str, bytes | None]] = []

    def opener(request, timeout=0):
        body = request.data if isinstance(request.data, bytes) else None
        seen.append((request.full_url, request.method, body))
        if request.method == "GET":
            payload = {"accounts": [{"_id": "acc-yt", "platform": "youtube", "name": "Kenh"}]}
        elif request.full_url.endswith("/media/presign"):
            payload = {"uploadUrl": "https://upload.example/put", "publicUrl": "https://media.example/clip.mp4"}
        elif request.method == "PUT":
            return _Response("")
        else:
            payload = {"post": {"_id": "post-1", "status": "published", "platforms": [{"platformPostUrl": "https://youtu.be/clip"}]}}
        return _Response(json.dumps(payload))

    result = ZernioClient("zernio-secret-key", opener=opener).publish_video(
        video_path=str(video),
        platform="youtube",
        title="Tieu de",
        content="Mo ta",
        tags=["bonsai", "vietnam"],
    )

    assert result["account_id"] == "acc-yt"
    assert result["url"] == "https://youtu.be/clip"
    assert "zernio-secret-key" not in json.dumps(result)
    post = json.loads(seen[-1][2].decode("utf-8"))
    assert post["platforms"] == [{
        "platform": "youtube",
        "accountId": "acc-yt",
        "platformSpecificData": {
            "title": "Tieu de",
            "description": "Mo ta",
            "tags": ["bonsai", "vietnam"],
        },
    }]
    assert post["content"].startswith("Mo ta")
    assert "#bonsai" in post["content"]
    assert post["mediaItems"] == [{"url": "https://media.example/clip.mp4", "type": "video"}]
    assert post["publishNow"] is True


def test_spoken_lines_for_video_reads_the_producing_job(tmp_path: Path) -> None:
    from src.interfaces.web.server import spoken_lines_for_video

    video = tmp_path / "out.mp4"
    video.write_bytes(b"video")
    job = tmp_path / "jobs" / "job_1"
    (job / "audio").mkdir(parents=True)
    (job / "result.json").write_text(
        json.dumps({"output_video_path": str(video)}), encoding="utf-8"
    )
    (job / "audio" / "segments.json").write_text(
        json.dumps([
            {"spoken_text": "Xin chào", "source_text": "ignored when spoken exists"},
            {"translated_text": "Cây cảnh"},
        ]),
        encoding="utf-8",
    )
    assert spoken_lines_for_video(video, jobs_root=tmp_path / "jobs") == ["Xin chào", "Cây cảnh"]


def test_source_poll_rejects_an_empty_list() -> None:
    with pytest.raises(RequestError, match="Chưa có nguồn"):
        start_source_poll("  ", 10, False)


def test_source_poll_rejects_a_bad_flag() -> None:
    with pytest.raises(RequestError, match="không hợp lệ"):
        start_source_poll("https://youtu.be/abc visual=maybe", 10, False)
