"""YouTube publication provider — official Data API v3, resumable upload.

Auth is a long-lived refresh token (the only setup step a human performs,
once). Service accounts do NOT work for YouTube uploads; refresh tokens are
the documented automation path.

Protocol (plain HTTP, no Google client library):
1. POST refresh token -> access token (cached until expiry).
2. POST /upload/youtube/v3/videos?uploadType=resumable with the metadata
   JSON -> session URL in the ``Location`` header.
3. PUT the bytes in 8 MiB chunks with ``Content-Range``; ``308`` means
   "continue", ``200/201`` means done.

Rate limits (HTTP 429), quota exhaustion and 5xx are TransientPublishError
so the use case's backoff owns retries; auth errors and malformed metadata
are permanent."""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Optional

import requests

from src.core.contracts import PublicationPort, TransientPublishError
from src.core.models.publication import PublishItem, PublishRecord

logger = logging.getLogger(__name__)

_TOKEN_URL = "https://oauth2.googleapis.com/token"
_UPLOAD_URL = "https://www.googleapis.com/upload/youtube/v3/videos"
_CHUNK_SIZE = 8 * 1024 * 1024


class YouTubePublication(PublicationPort):
    platform_name = "youtube"

    def __init__(
        self,
        client_id: str,
        client_secret: str,
        refresh_token: str,
        session: Optional[Any] = None,
    ) -> None:
        self._client_id = client_id
        self._client_secret = client_secret
        self._refresh_token = refresh_token
        self._session = session if session is not None else requests.Session()
        self._access_token: Optional[str] = None
        self._token_expires_at = 0.0

    # -- auth ---------------------------------------------------------------

    def _access_token_now(self) -> str:
        if self._access_token and time.monotonic() < self._token_expires_at:
            return self._access_token
        resp = self._session.post(
            _TOKEN_URL,
            data={
                "client_id": self._client_id,
                "client_secret": self._client_secret,
                "refresh_token": self._refresh_token,
                "grant_type": "refresh_token",
            },
            timeout=30,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"YouTube token refresh failed ({resp.status_code}): {resp.text[:300]}")
        payload = resp.json()
        self._access_token = payload["access_token"]
        # Refresh a minute early so an upload never starts with a dying token.
        self._token_expires_at = time.monotonic() + max(60.0, payload.get("expires_in", 3600) - 60)
        return self._access_token

    # -- upload -------------------------------------------------------------

    def publish(self, item: PublishItem) -> PublishRecord:
        video_id = self._upload(item)
        return PublishRecord(
            item_id=item.item_id,
            video_path=item.video_path,
            platform=self.platform_name,
            status="PUBLISHED",
            platform_video_id=video_id,
            url=f"https://youtu.be/{video_id}",
        )

    def _upload(self, item: PublishItem) -> str:
        token = self._access_token_now()
        meta = item.metadata
        metadata_body = {
            "snippet": {
                "title": meta.title[:100],
                "description": meta.description[:5000],
                "tags": [t[:100] for t in meta.tags][:30],
                "categoryId": meta.category_id or "24",
                "defaultLanguage": meta.language,
                "defaultAudioLanguage": meta.language,
            },
            "status": {
                "privacyStatus": meta.privacy,
                "selfDeclaredMadeForKids": False,
            },
        }

        file_size = os.path.getsize(item.video_path)
        session_resp = self._session.post(
            _UPLOAD_URL,
            params={"uploadType": "resumable", "part": "snippet,status"},
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json; charset=UTF-8",
                "X-Upload-Content-Length": str(file_size),
                "X-Upload-Content-Type": "video/mp4",
            },
            data=json.dumps(metadata_body),
            timeout=60,
        )
        if session_resp.status_code in (429, 500, 502, 503):
            raise TransientPublishError(
                f"upload session http {session_resp.status_code}: {session_resp.text[:200]}"
            )
        if session_resp.status_code != 200:
            raise RuntimeError(
                f"upload session rejected ({session_resp.status_code}): {session_resp.text[:300]}"
            )
        upload_url = session_resp.headers.get("Location")
        if not upload_url:
            raise RuntimeError("upload session returned no Location header")

        with open(item.video_path, "rb") as fh:
            offset = 0
            while True:
                chunk = fh.read(_CHUNK_SIZE)
                if not chunk and offset > 0:
                    break
                headers = {
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "video/mp4",
                    "Content-Length": str(len(chunk)),
                    "Content-Range": f"bytes {offset}-{offset + len(chunk) - 1}/{file_size}",
                }
                put = self._session.put(upload_url, headers=headers, data=chunk, timeout=600)
                if put.status_code == 308:
                    offset += len(chunk)
                    continue
                if put.status_code in (429, 500, 502, 503):
                    raise TransientPublishError(f"chunk http {put.status_code}: {put.text[:200]}")
                if put.status_code in (200, 201):
                    return put.json()["id"]
                if put.status_code == 403 and "quotaExceeded" in put.text:
                    raise TransientPublishError("YouTube quota exceeded for today")
                raise RuntimeError(f"chunk rejected ({put.status_code}): {put.text[:300]}")
        raise RuntimeError("upload ended without a completion response")
