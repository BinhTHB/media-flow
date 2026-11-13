"""Zernio REST client: one API key talks to the accounts connected on that key.

MediaFlow keeps a separate key per platform row. This client never logs the
key. Callers pass it in and receive account ids, public media URLs, and post
ids back.
"""

from __future__ import annotations

import json
import mimetypes
from pathlib import Path
from typing import Any, Callable, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

API_ROOT = "https://zernio.com/api/v1"

#: Platforms the studio can target. The value is the one Zernio expects.
ZERNIO_PLATFORMS = (
    "youtube",
    "tiktok",
    "instagram",
    "facebook",
    "twitter",
    "linkedin",
    "pinterest",
    "threads",
    "bluesky",
    "reddit",
)


class ZernioError(RuntimeError):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = int(status)


class ZernioClient:
    def __init__(self, api_key: str, *, opener: Optional[Callable[..., Any]] = None, timeout: float = 60.0) -> None:
        key = (api_key or "").strip()
        if not key:
            raise ZernioError(401, "Thiếu API key Zernio.")
        self._key = key
        self._open = opener or urlopen
        self._timeout = float(timeout)

    def list_accounts(self) -> list[dict[str, str]]:
        payload = self._json("GET", "/accounts")
        rows = payload.get("accounts") if isinstance(payload, dict) else None
        if not isinstance(rows, list):
            rows = []
        found: list[dict[str, str]] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            account_id = str(row.get("_id") or row.get("id") or "").strip()
            platform = str(row.get("platform") or "").strip().lower()
            if not account_id or not platform:
                continue
            name = str(row.get("name") or row.get("username") or row.get("displayName") or "")
            found.append({"id": account_id, "platform": platform, "name": name})
        return found

    def account_for(self, platform: str) -> dict[str, str]:
        wanted = platform.strip().lower()
        matches = [row for row in self.list_accounts() if row["platform"] == wanted]
        if not matches:
            raise ZernioError(404, f"API key này chưa kết nối nền tảng {wanted}.")
        return matches[0]

    def upload_video(self, video_path: str) -> str:
        path = Path(video_path)
        if not path.is_file():
            raise ZernioError(404, f"Không thấy video {path.name}.")
        content_type = mimetypes.guess_type(path.name)[0] or "video/mp4"
        presign = self._json(
            "POST",
            "/media/presign",
            {"filename": path.name, "contentType": content_type, "size": path.stat().st_size},
        )
        upload_url = str(presign.get("uploadUrl") or "")
        public_url = str(presign.get("publicUrl") or "")
        if not upload_url.startswith("https://") or not public_url.startswith("https://"):
            raise ZernioError(502, "Zernio không trả URL tải lên.")
        self._put_file(upload_url, path, content_type)
        return public_url

    def publish_video(
        self,
        *,
        video_path: str,
        platform: str,
        title: str,
        content: str,
        tags: Optional[list[str]] = None,
    ) -> dict[str, str]:
        account = self.account_for(platform)
        public_url = self.upload_video(video_path)
        target: dict[str, Any] = {"platform": account["platform"], "accountId": account["id"]}
        clean_tags = [str(tag).lstrip("#").strip() for tag in (tags or []) if str(tag).strip()]
        if account["platform"] == "youtube":
            specific: dict[str, Any] = {}
            if title.strip():
                specific["title"] = title.strip()
            if content.strip():
                specific["description"] = content.strip()
            if clean_tags:
                specific["tags"] = clean_tags[:30]
            if specific:
                target["platformSpecificData"] = specific
        hashtags = " ".join(f"#{tag.replace(' ', '')}" for tag in clean_tags)
        caption = content.strip() or title.strip() or Path(video_path).stem
        if hashtags and hashtags not in caption:
            caption = f"{caption}\n\n{hashtags}"
        created = self._json(
            "POST",
            "/posts",
            {
                "content": caption,
                "mediaItems": [{"url": public_url, "type": "video"}],
                "platforms": [target],
                "publishNow": True,
            },
            timeout=max(self._timeout, 180.0),
        )
        post = created.get("post") if isinstance(created, dict) else None
        if not isinstance(post, dict):
            post = {}
        post_url = ""
        platforms = post.get("platforms")
        if isinstance(platforms, list) and platforms and isinstance(platforms[0], dict):
            post_url = str(platforms[0].get("platformPostUrl") or "")
        return {
            "post_id": str(post.get("_id") or post.get("id") or ""),
            "status": str(post.get("status") or ""),
            "platform": account["platform"],
            "account_id": account["id"],
            "url": post_url,
        }

    def _json(self, method: str, path: str, body: Optional[dict[str, Any]] = None, timeout: Optional[float] = None) -> dict[str, Any]:
        data = None if body is None else json.dumps(body).encode("utf-8")
        request = Request(
            API_ROOT + path,
            data=data,
            method=method,
            headers={
                "Authorization": f"Bearer {self._key}",
                "Accept": "application/json",
                "Content-Type": "application/json",
            },
        )
        payload = self._read(request, timeout or self._timeout)
        return payload if isinstance(payload, dict) else {}

    def _put_file(self, upload_url: str, path: Path, content_type: str) -> None:
        with path.open("rb") as handle:
            request = Request(
                upload_url,
                data=handle.read(),
                method="PUT",
                headers={"Content-Type": content_type},
            )
        self._read(request, max(self._timeout, 300.0))

    def _read(self, request: Request, timeout: float) -> Any:
        try:
            with self._open(request, timeout=timeout) as response:
                raw = response.read()
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:300]
            raise ZernioError(exc.code, detail or f"Zernio HTTP {exc.code}") from exc
        except URLError as exc:
            raise ZernioError(503, f"Không kết nối được Zernio: {exc.reason}") from exc
        if not raw:
            return {}
        try:
            return json.loads(raw.decode("utf-8"))
        except ValueError:
            return {}
