"""Regression tests: OCR stage cache must be deterministic across runs.

CachedOCR used to skip caching (and re-read) any batch containing an empty
string. Re-reading across runs made Gemini return slightly different
recognitions for the same crop (e.g. full-width vs half-width punctuation),
which drifted source texts, missed the translation cache, and regressed READY
counts during provider outages (original subtitles leaking back in)."""

import numpy as np

from src.infrastructure.stage_cache import CachedOCR, JsonStageCache


class _CountingProvider:
    def __init__(self, results):
        self.results = results
        self.calls = 0

    def read_crops(self, crops):
        self.calls += 1
        return self.results


def test_cached_ocr_persists_batches_containing_empty_reads(tmp_path):
    provider = _CountingProvider(["你好", ""])
    crops = [np.zeros((8, 8, 3), dtype=np.uint8), np.ones((8, 8, 3), dtype=np.uint8)]
    adapter = CachedOCR(provider, JsonStageCache(tmp_path, {"stage": "test"}))

    assert adapter.read_crops(crops) == ["你好", ""]
    assert adapter.read_crops(crops) == ["你好", ""]
    # Second read is a cache hit: no provider call, identical results.
    assert provider.calls == 1


def test_cached_ocr_batch_without_empties_still_cached(tmp_path):
    provider = _CountingProvider(["a", "b"])
    crops = [np.zeros((8, 8, 3), dtype=np.uint8), np.ones((8, 8, 3), dtype=np.uint8)]
    adapter = CachedOCR(provider, JsonStageCache(tmp_path, {"stage": "test"}))

    assert adapter.read_crops(crops) == ["a", "b"]
    assert adapter.read_crops(crops) == ["a", "b"]
    assert provider.calls == 1
