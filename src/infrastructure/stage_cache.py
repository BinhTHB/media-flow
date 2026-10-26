"""Content-addressed provider caches. No credentials or media pixels are persisted."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import logging
import os
from pathlib import Path
import tempfile
from typing import Any

from src.core.contracts import DetectorPort, OCRPort, SceneBoundaryPort, TranslationPort
from src.core.models.detection import Detection

logger = logging.getLogger(__name__)


def file_digest(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class JsonStageCache:
    def __init__(self, root: str | Path, identity: dict[str, Any]):
        self.root = Path(root)
        self.identity = identity
        self.hits = 0
        self.misses = 0

    def _path(self, stage: str, inputs: Any) -> Path:
        encoded = json.dumps({"version": 1, "identity": self.identity, "inputs": inputs},
                             sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
        return self.root / stage / (hashlib.sha256(encoded).hexdigest() + ".json")

    def get(self, stage: str, inputs: Any) -> Any:
        path = self._path(stage, inputs)
        try:
            envelope = json.loads(path.read_text(encoding="utf-8"))
            payload = envelope["payload"]
            if envelope["checksum"] != self._checksum(payload):
                raise ValueError("Cache checksum mismatch")
            self.hits += 1
            return payload
        except (OSError, ValueError, KeyError, TypeError):
            self.misses += 1
            return None

    @staticmethod
    def _checksum(payload: Any) -> str:
        return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False,
                                         allow_nan=False).encode()).hexdigest()

    def put(self, stage: str, inputs: Any, payload: Any) -> None:
        path = self._path(stage, inputs)
        path.parent.mkdir(parents=True, exist_ok=True)
        envelope = {"checksum": self._checksum(payload), "payload": payload}
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                             suffix=".tmp", delete=False) as stream:
                temporary = Path(stream.name)
                json.dump(envelope, stream, ensure_ascii=False, allow_nan=False)
            os.replace(temporary, path)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()


class CachedDetector(DetectorPort):
    def __init__(self, provider: DetectorPort, cache: JsonStageCache):
        self.provider, self.cache = provider, cache

    def detect_video(self, video_path: str, stride: int = 1) -> list[Detection]:
        inputs = {"video_sha256": file_digest(video_path), "stride": stride}
        payload = self.cache.get("detection", inputs)
        if payload is not None:
            try:
                return [Detection.from_dict(row) for row in payload]
            except (TypeError, KeyError, ValueError):
                logger.warning("Invalid detection cache; recomputing")
        result = self.provider.detect_video(video_path, stride=stride)
        self.cache.put("detection", inputs, [asdict(item) for item in result])
        return result


def _valid_texts(value: Any, count: int, allow_empty: bool = False) -> bool:
    return (isinstance(value, list) and len(value) == count
            and all(isinstance(item, str) and (allow_empty or bool(item.strip())) for item in value))


class CachedSceneCuts(SceneBoundaryPort):
    def __init__(self, provider: SceneBoundaryPort, cache: JsonStageCache):
        self.provider, self.cache = provider, cache

    def detect_cuts(self, video_path: str) -> list[int]:
        inputs = {"video_sha256": file_digest(video_path)}
        payload = self.cache.get("scene", inputs)
        if payload is not None:
            return [int(cut) for cut in payload]
        cuts = self.provider.detect_cuts(video_path)
        self.cache.put("scene", inputs, [int(cut) for cut in cuts])
        return cuts


class CachedOCR(OCRPort):
    def __init__(self, provider: OCRPort, cache: JsonStageCache):
        self.provider, self.cache = provider, cache

    def read_crops(self, crop_images: list[Any]) -> list[str]:
        if not crop_images:
            return []
        inputs = [{"shape": list(crop.shape), "dtype": str(crop.dtype),
                   "sha256": hashlib.sha256(crop.tobytes()).hexdigest()} for crop in crop_images]
        payload = self.cache.get("ocr", inputs)
        # Empty strings are legitimate cached outcomes (unreadable crops);
        # rejecting them here forced re-reads whose recognition drift made
        # source texts unstable across runs.
        if _valid_texts(payload, len(crop_images), allow_empty=True):
            return payload
        result = self.provider.read_crops(crop_images)
        if not _valid_texts(result, len(crop_images), allow_empty=True):
            raise ValueError("OCR returned a mismatched or invalid batch")
        # Persist every batch, including reads that contain empty strings.
        # Within-run transient failures are already retried by the caller's
        # bounded rounds; skipping the cache here made re-reads across runs
        # return different recognitions for the same crop, which drifted the
        # source texts, invalidated the translation cache, and could regress
        # READY counts (original subtitles leaking back in).
        self.cache.put("ocr", inputs, result)
        return result


class CachedTranslation(TranslationPort):
    def __init__(self, provider: TranslationPort, cache: JsonStageCache):
        self.provider, self.cache = provider, cache

    def translate_texts(self, texts: list[str], target_language: str = "vi",
                        source_language: str = "auto") -> list[str]:
        if not texts:
            return []
        # Cache the entire ordered batch: neighboring sentences affect translation.
        inputs = {"texts": texts, "target": target_language, "source": source_language}
        payload = self.cache.get("translation", inputs)
        if _valid_texts(payload, len(texts)):
            return payload
        result = self.provider.translate_texts(texts=texts, target_language=target_language,
                                               source_language=source_language)
        if not _valid_texts(result, len(texts), allow_empty=True):
            raise ValueError("Translation returned a mismatched or invalid batch")
        if _valid_texts(result, len(texts)):
            self.cache.put("translation", inputs, result)
        return result
