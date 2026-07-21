import json
import pytest
from pathlib import Path
import importlib.util

SCRIPT = Path(__file__).resolve().parents[1] / "artifacts" / "round19" / "parse_mask_coverage.py"

def load_module():
    spec = importlib.util.spec_from_file_location("mask_coverage", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module

def test_parse_empty_ass_returns_zeros(tmp_path):
    module = load_module()
    ass_file = tmp_path / "empty.ass"
    ass_file.write_text("[Script Info]\nPlayResX: 1920\nPlayResY: 1080\n", encoding="utf-8")
    
    result = module.compute_coverage(ass_file)
    assert result["mask_count"] == 0
    assert result["union_area_pct_mean"] == 0.0

def test_parse_simple_mask_events(tmp_path):
    module = load_module()
    ass_file = tmp_path / "test.ass"
    # Create simple ASS content with Styles MarginL, MarginR, MarginV
    ass_content = """[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,20,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,-1,0,0,0,100,100,0,0,1,2,2,2,100,100,10,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: 0,0:00:01.00,0:00:03.00,Default,,100,100,10,,1
Dialogue: 0,0:00:02.00,0:00:04.00,Default,,150,150,10,,1
"""
    ass_file.write_text(ass_content, encoding="utf-8")
    result = module.compute_coverage(ass_file)
    
    # MarginL style defaults used in code (actually style parser in code maps MarginL=100, MarginR=100, MarginV=10)
    # But margin from dialogue format takes precedence in compute_coverage?
    # Let's inspect dialogues: Start 1.0 to 3.0, and 2.0 to 4.0.
    # Total duration covered: 1.0 to 4.0 (3 seconds). Frame count: 3+1 = 4 frames sampled.
    # frame 0: t=1.0. Active: dialogue 1. area = (1920 - 100 - 100)*108 / (1920*1080) = 1720 / 19200 = 8.958%
    # frame 1: t=2.0. Active: dialogue 1 & 2. union width = (1720 + 1620) = 3340 -> capped at 1920. area = 10% (capping)
    # frame 2: t=3.0. Active: dialogue 2. area = (1920 - 150 - 150)*108 / (1920*1080) = 1620 / 19200 = 8.4375%
    # frame 3: t=4.0. Active: none. area = 0.0%
    assert result["mask_count"] == 2
    assert result["total_frame_count"] == 4
    # Check max and mean are within correct ranges
    assert result["union_area_pct_max"] == 10.0
    assert result["union_area_pct_mean"] > 5.0
    assert result["boxes_per_frame_max"] == 2
