"""TikTok session-based publisher.

TikTok's Content Posting API forbids "an app that copies arbitrary contents
from other platforms to TikTok", caps posts per creator, and restricts
unaudited clients to private-only posts. Driving the real upload page with a
logged-in session is how consumer re-upload tools operate instead.

Selectors are deliberately centralized in ``_SEL`` and use role/text/placeholder
queries where possible: TikTok's class names are generated hashes that change
on every deploy, so CSS-class selectors rot within days.

This module is import-safe without Playwright installed — the browser is only
created when ``publish`` runs.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from src.core.models.publication import PublishItem
from src.infrastructure.publishing.browser_base import BrowserPublisher

logger = logging.getLogger(__name__)

UPLOAD_URL = "https://www.tiktok.com/tiktokstudio/upload"

#: UI handles, kept in one place because TikTok changes markup often. Prefer
#: role/placeholder/text selectors over generated class names.
_SEL = {
    "file_input": "input[type=file]",
    "caption": "div[contenteditable=true]",
    "post_button": "button[data-e2e='post_video_button']",
    "login_wall": "input[type=password]",
    "upload_progress": "[data-e2e='upload-progress']",
    "success_marker": "[data-e2e='post-success']",
}


class TikTokBrowserPublisher(BrowserPublisher):
    platform_name = "tiktok"

    def _is_logged_in(self, page: Any) -> bool:
        try:
            return page.locator(_SEL["login_wall"]).count() == 0
        except Exception:
            # A transient locator error must not be read as "logged out".
            return True

    def _open_upload_page(self, page: Any) -> None:
        page.goto(UPLOAD_URL, wait_until="domcontentloaded")
        # The uploader is a client-rendered SPA: wait for the file input rather
        # than a fixed sleep.
        page.wait_for_selector(_SEL["file_input"], state="attached")

    def _submit(self, page: Any, item: PublishItem) -> Optional[str]:
        self._wait_for_file_chooser(page, item.video_path)

        # The caption editor appears only after the video is accepted.
        page.wait_for_selector(_SEL["caption"], state="visible")
        caption = _compose_caption(item)
        page.fill(_SEL["caption"], caption)

        page.click(_SEL["post_button"])

        # Success is asynchronous: the button re-enables and a confirmation
        # appears. Waiting on the marker is what makes the record truthful.
        page.wait_for_selector(_SEL["success_marker"], state="visible")
        logger.info("TikTok: published %s", item.video_path)
        return None


def _compose_caption(item: PublishItem) -> str:
    """TikTok caption = title + description + hashtags, within its limit."""
    parts = [item.metadata.title.strip()]
    if item.metadata.description.strip():
        parts.append(item.metadata.description.strip())
    tags = " ".join(
        tag if tag.startswith("#") else f"#{tag}"
        for tag in item.metadata.tags
        if tag.strip()
    )
    if tags:
        parts.append(tags)
    caption = "\n\n".join(p for p in parts if p)
    # TikTok truncates silently past ~2200 chars; trimming here keeps the log
    # honest about what was actually posted.
    return caption[:2200]
