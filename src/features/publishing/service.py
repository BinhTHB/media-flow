"""Publish use case: one finished video -> metadata -> N platforms -> records.

Retry policy lives here (not in providers): TransientPublishError backs off
exponentially; any other exception is permanent. The state store makes the
whole flow idempotent — a worker restart never re-publishes a video that
already has a PUBLISHED record on a platform.
"""

from __future__ import annotations

import json
import logging
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.core.contracts import (
    MetadataGenerationPort,
    PublicationPort,
    TransientPublishError,
    VideoVariationPort,
)
from src.core.models.publication import (
    PublishItem,
    PublishRecord,
    VideoMetadata,
    fallback_metadata,
    item_id_for,
)
from src.features.publishing.variation import VariationPolicy

logger = logging.getLogger(__name__)

#: Optional sidecar next to the queued video: {"title": ..., "description": ...,
#: "tags": [...], "privacy": ...}. Overrides generated metadata entirely.
SIDECAR_SUFFIX = ".publish.json"
VIDEO_GLOBS = ("*.mp4", "*.mov", "*.mkv", "*.webm")


class PublicationStoreProtocol:
    """Minimal store surface the service depends on (structural typing)."""

    def is_published(self, item_id: str, platform: str) -> bool: ...
    def add(self, record: PublishRecord) -> None: ...


class PublishService:
    def __init__(
        self,
        targets: List[PublicationPort],
        metadata_generator: Optional[MetadataGenerationPort] = None,
        store: Optional[PublicationStoreProtocol] = None,
        max_attempts: int = 5,
        backoff_base_seconds: float = 30.0,
        sleep: Any = time.sleep,
        variation_policy: Optional[VariationPolicy] = None,
        variation_provider: Optional[VideoVariationPort] = None,
        variation_dir: Optional[str] = None,
    ) -> None:
        if not targets:
            raise ValueError("PublishService needs at least one publication target.")
        if variation_policy is not None and variation_provider is None:
            raise ValueError(
                "variation_policy requires a variation_provider to render the profile."
            )
        self.targets = targets
        self.metadata_generator = metadata_generator
        self.store = store
        self.max_attempts = max(1, int(max_attempts))
        self.backoff_base_seconds = float(backoff_base_seconds)
        self._sleep = sleep
        self.variation_policy = variation_policy
        self.variation_provider = variation_provider
        self.variation_dir = variation_dir

    # -- variation ----------------------------------------------------------

    def _varied_copy(
        self, item_id: str, video_path: str, attempt: int
    ) -> tuple[Optional[str], Optional[dict]]:
        """Render the variant for this (item, attempt), or ``(None, None)``.

        Returns the path of the varied file plus its profile dict for the
        audit trail. A rendering failure is not fatal: the caller publishes the
        master file rather than dropping the item entirely.
        """
        if self.variation_policy is None or self.variation_provider is None:
            return None, None
        delivered = self._delivered_profile(video_path)
        if delivered is not None:
            return None, delivered
        profile = self.variation_policy.profile_for(item_id, attempt)
        if profile.is_identity:
            return None, profile.to_dict()

        source = Path(video_path)
        base_dir = Path(self.variation_dir) if self.variation_dir else source.parent / "_variants"
        base_dir.mkdir(parents=True, exist_ok=True)
        target = base_dir / f"{source.stem}__{item_id}_{attempt}.mp4"
        try:
            rendered = self.variation_provider.apply(str(source), profile, str(target))
            logger.info("Publish: rendered variant %s for %s", Path(rendered).name, source.name)
            return rendered, profile.to_dict()
        except Exception as exc:
            logger.warning(
                "Publish: variation failed for %s (%s); publishing the master file.",
                source.name, exc,
            )
            return None, profile.to_dict()

    @staticmethod
    def _delivered_profile(video_path: str) -> Optional[dict]:
        """Return the profile already baked into a processed file, if any.

        The processing job writes ``<stem>.variation.json`` beside the finished
        video. That file is the upload, so publishing it again would stack a
        second speed and crop change on top of the checked result.
        """
        sidecar = Path(video_path).with_name(f"{Path(video_path).stem}.variation.json")
        if not sidecar.is_file():
            return None
        try:
            payload = json.loads(sidecar.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return payload if isinstance(payload, dict) else None

    def _metadata_for(self, video_path: str, hints: Optional[Dict[str, Any]]) -> VideoMetadata:
        sidecar = self._read_sidecar(video_path)
        if sidecar is not None:
            return VideoMetadata(
                title=str(sidecar.get("title") or Path(video_path).stem),
                description=str(sidecar.get("description") or ""),
                tags=[str(t) for t in sidecar.get("tags") or []],
                privacy=str(sidecar.get("privacy") or "private"),
                language=str(sidecar.get("language") or "vi"),
            )
        if self.metadata_generator is not None:
            return self.metadata_generator.generate(video_path, hints or {})
        return fallback_metadata(video_path)

    @staticmethod
    def _read_sidecar(video_path: str) -> Optional[Dict[str, Any]]:
        sidecar = Path(video_path).with_suffix(SIDECAR_SUFFIX)
        if not sidecar.is_file():
            return None
        try:
            payload = json.loads(sidecar.read_text(encoding="utf-8"))
            return payload if isinstance(payload, dict) else None
        except (OSError, ValueError) as exc:
            logger.warning("Publish: unreadable sidecar %s (%s); ignoring.", sidecar.name, exc)
            return None

    def publish_file(
        self, video_path: str, hints: Optional[Dict[str, Any]] = None
    ) -> List[PublishRecord]:
        video = Path(video_path)
        if not video.is_file():
            raise FileNotFoundError(video_path)
        item_id = item_id_for(video_path)
        metadata = self._metadata_for(video_path, hints)
        metadata.validate()

        records: List[PublishRecord] = []
        for target in self.targets:
            record = self._publish_one(item_id, video_path, target, metadata)
            records.append(record)
            if self.store is not None:
                self.store.add(record)
        return records

    def _publish_one(
        self, item_id: str, video_path: str, target: PublicationPort, metadata: VideoMetadata
    ) -> PublishRecord:
        platform = getattr(target, "platform_name", target.__class__.__name__)
        if self.store is not None and self.store.is_published(item_id, platform):
            logger.info("Publish: %s already published on %s; skipping.", video_path, platform)
            return PublishRecord(
                item_id=item_id, video_path=video_path, platform=platform,
                status="PUBLISHED", attempts=0, error="duplicate-skipped",
            )

        item = PublishItem(item_id=item_id, video_path=video_path, platform=platform, metadata=metadata)
        last_error: Optional[str] = None
        for attempt in range(1, self.max_attempts + 1):
            # Render the variant for THIS attempt. A retry after a transient
            # failure reuses the same attempt-derived profile only if the attempt
            # number is unchanged; the profile is deterministic per (item,
            # attempt) so a true retry re-uploads the same bytes.
            varied_path, profile_dict = self._varied_copy(item_id, video_path, attempt)
            publish_item = item
            if varied_path:
                publish_item = PublishItem(
                    item_id=item.item_id,
                    video_path=varied_path,
                    platform=platform,
                    metadata=metadata,
                )
            try:
                record = target.publish(publish_item)
                logger.info(
                    "Publish: %s -> %s OK (%s) on attempt %d%s.",
                    Path(video_path).name, platform, record.url, attempt,
                    f" (variant {profile_dict.get('label')})" if varied_path else "",
                )
                record.attempts = attempt
                if profile_dict:
                    record.error = None
                return record
            except TransientPublishError as exc:
                last_error = f"transient: {exc}"
                if attempt < self.max_attempts:
                    delay = self.backoff_base_seconds * (2 ** (attempt - 1))
                    logger.warning(
                        "Publish: %s -> %s transient failure (attempt %d/%d), retrying in %.0fs: %s",
                        Path(video_path).name, platform, attempt, self.max_attempts, delay, exc,
                    )
                    self._sleep(delay)
            except Exception as exc:  # permanent: no retry
                last_error = f"permanent: {exc}"
                logger.error("Publish: %s -> %s failed permanently: %s", video_path, platform, exc)
                break
            finally:
                # Variants are derived artifacts: never accumulate them.
                if varied_path:
                    Path(varied_path).unlink(missing_ok=True)

        return PublishRecord(
            item_id=item_id, video_path=video_path, platform=platform,
            status="FAILED", error=last_error, attempts=attempt,
        )

    def process_queue_dir(self, queue_dir: str) -> List[PublishRecord]:
        """Publish every new video in a queue directory; idempotent across
        runs via the store. Returns this run's records (may be empty)."""
        queue = Path(queue_dir)
        if not queue.is_dir():
            raise FileNotFoundError(queue_dir)

        videos: List[Path] = []
        for pattern in VIDEO_GLOBS:
            videos.extend(p for p in queue.glob(pattern) if p.is_file())
        videos.sort(key=lambda p: p.stat().st_mtime)

        records: List[PublishRecord] = []
        for video in videos:
            item_id = item_id_for(str(video))
            already = all(
                (self.store is None)
                or self.store.is_published(item_id, getattr(t, "platform_name", t.__class__.__name__))
                for t in self.targets
            )
            if already and self.store is not None:
                logger.info("Publish: %s fully published already; skipping.", video.name)
                continue
            records.extend(self.publish_file(str(video)))
        return records
