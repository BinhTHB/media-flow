"""FileTargetPublication: copies the finished video into a target folder.

The offline/local destination used for dry-runs and for hand-off to other
tools (e.g. a separate uploader or a NAS drop). Zero network, fully
deterministic — also the provider used by unit tests of the service."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from src.core.contracts import PublicationPort
from src.core.models.publication import PublishItem, PublishRecord


class FileTargetPublication(PublicationPort):
    #: Per-instance name: two file targets pointing at different directories
    #: are distinct destinations, not duplicates of one platform.
    def __init__(self, target_dir: str) -> None:
        self.platform_name = f"file:{target_dir}"
        self._target_dir = Path(target_dir)

    def publish(self, item: PublishItem) -> PublishRecord:
        self._target_dir.mkdir(parents=True, exist_ok=True)
        src = Path(item.video_path)
        dest = self._target_dir / src.name
        if dest.exists() and dest.stat().st_size == src.stat().st_size:
            # Idempotent copy: same file already delivered.
            return self._record(item, dest)
        shutil.copy2(src, dest)
        sidecar = dest.with_suffix(".publish.json")
        sidecar.write_text(
            json.dumps(
                {
                    "title": item.metadata.title,
                    "description": item.metadata.description,
                    "tags": item.metadata.tags,
                    "privacy": item.metadata.privacy,
                    "source_item_id": item.item_id,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return self._record(item, dest)

    def _record(self, item: PublishItem, dest: Path) -> PublishRecord:
        return PublishRecord(
            item_id=item.item_id,
            video_path=item.video_path,
            platform=self.platform_name,
            status="PUBLISHED",
            url=str(dest),
        )
