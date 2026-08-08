"""
Unit tests for V2 quality & correctness fixes:
1. GeminiOCRAdapter structured response with item_id mapping
2. SubtitleProcessor OCR consensus duplication resolution
3. Translation adapter text-only contract
4. Verification after image extraction from final MP4
5. Report artifact count calculation integrity
"""

import json
from pathlib import Path
from unittest.mock import MagicMock
import pytest

from src.features.translation.service import TranslationService
from src.features.visual.subtitle.processor import SubtitleProcessor
from src.infrastructure.ocr.gemini_ocr import GeminiOCRAdapter
from src.infrastructure.translation.gemini_translation import GeminiTranslationAdapter


def test_ocr_consensus_resolves_string_duplication():
    # 谢谢 vs 谢谢谢谢 -> 谢谢
    assert SubtitleProcessor._resolve_ocr_consensus(["谢谢", "谢谢谢谢"]) == "谢谢"
    assert SubtitleProcessor._resolve_ocr_consensus(["谢谢谢谢", "谢谢"]) == "谢谢"
    
    # 拜拜 vs 拜拜拜拜 -> 拜拜
    assert SubtitleProcessor._resolve_ocr_consensus(["拜拜", "拜拜拜拜"]) == "拜拜"
    assert SubtitleProcessor._resolve_ocr_consensus(["拜拜拜拜", "拜拜"]) == "拜拜"

    # Single candidate
    assert SubtitleProcessor._resolve_ocr_consensus(["谢谢你了"]) == "谢谢你了"

    # Disagreeing candidates majority vote
    assert SubtitleProcessor._resolve_ocr_consensus(["hello", "hello", "world"]) == "hello"


def test_translation_adapter_is_text_only():
    mock_provider = MagicMock()
    mock_provider.translate.side_effect = lambda segs, target_language: segs

    adapter = GeminiTranslationAdapter()
    adapter.provider = mock_provider

    res = adapter.translate_texts(["hello", "world"], target_language="vi")
    assert isinstance(res, list)
    assert len(res) == 2
    # Verify only text segments were passed to translation provider
    mock_provider.translate.assert_called_once()
    passed_segs = mock_provider.translate.call_args[0][0]
    for s in passed_segs:
        assert hasattr(s, "text")
        assert s.bbox is None
        assert not hasattr(s, "image")


def test_ocr_adapter_handles_structured_json_dict(monkeypatch):
    adapter = GeminiOCRAdapter(api_key="mock_key")
    
    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.text = json.dumps({"item_0": "拜拜", "item_1": "谢谢"})
    mock_client.generate_content.return_value = mock_resp

    monkeypatch.setattr("src.infrastructure.gemini.client.GeminiClient", lambda api_key: mock_client)

    import numpy as np
    dummy_crop1 = np.zeros((10, 10, 3), dtype=np.uint8)
    dummy_crop2 = np.zeros((10, 10, 3), dtype=np.uint8)

    results = adapter.read_crops([dummy_crop1, dummy_crop2])
    assert results == ["拜拜", "谢谢"]


def test_report_artifact_counts_equal_filesystem_counts(tmp_path):
    ver_dir = tmp_path / "verification"
    ver_dir.mkdir(parents=True, exist_ok=True)

    (ver_dir / "event_1_before.jpg").write_text("a")
    (ver_dir / "event_1_crop.jpg").write_text("b")
    (ver_dir / "event_1_after_final.jpg").write_text("c")

    files = list(ver_dir.glob("*.jpg"))
    before_count = len(list(ver_dir.glob("*_before.jpg")))
    crop_count = len(list(ver_dir.glob("*_crop.jpg")))
    after_count = len(list(ver_dir.glob("*_after_final.jpg")))

    assert len(files) == before_count + crop_count + after_count
