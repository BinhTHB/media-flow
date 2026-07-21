"""RED test: compute_coverage measures real geometry masks (\pos + \p1)."""
import json
import pytest
from pathlib import Path
import importlib.util

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "parse_mask_coverage.py"


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
    geo = {"width": 1920, "height": 1080, "fps": 30.0, "frame_count": 0}

    result = module.compute_mask_coverage(ass_file, geo)
    assert result["mask_events_layer_0"] == 0
    assert result["union_mask_area_pct"]["mean"] == 0.0


def test_parse_simple_mask_events(tmp_path):
    """Parser reads \pos + \p1 geometry, not margins. 3 frames sampled at fps=1."""
    module = load_module()
    ass_file = tmp_path / "test.ass"
    # 1920x1080, 3 frames at fps=1: t=0.0, 1.0, 2.0
    # Mask event 1: x=100,y=800, 1720x108 (≈9.0% of screen)
    # Mask event 2: x=500,y=800, 1400x108 (≈7.29% of screen)
    ass_content = """[Script Info]
PlayResX: 1920
PlayResY: 1080

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,20,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,-1,0,0,0,100,100,0,0,1,2,2,2,0,0,0,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: 0,0:00:00.00,0:00:02.00,Default,,0,0,0,,{\\p1}\\pos(100,800)m 0 0 l 1720 0 l 1720 108 l 0 108{\\p0}
Dialogue: 0,0:00:01.00,0:00:03.00,Default,,0,0,0,,{\\p1}\\pos(500,800)m 0 0 l 1400 0 l 1400 108 l 0 108{\\p0}
"""
    ass_file.write_text(ass_content, encoding="utf-8")

    # fps=1, frame_count=3 -> frames at t=0.0, 1.0, 2.0
    geo = {"width": 1920, "height": 1080, "fps": 1.0, "frame_count": 3}
    result = module.compute_mask_coverage(ass_file, geo, padding_seconds=0.0)

    # Frame 0 (t=0.0): only event 1 active, area = 1720*108 / (1920*1080) = 8.9583%
    # Frame 1 (t=1.0): both active, union via sweep-line
    #   [100,500): only ev1 -> 1720*108
    #   [500,1820): both -> union y=108px, x=1320 -> 142560px
    #   [1820,1900): only ev2 -> 1400*108
    #   total = 1720*108*(500-100)/1720 + 1320*108 + 1400*108*(1900-1820)/1400
    #   = 43200 + 142560 + 8640 = 194400 / 2073600 = 9.3750%
    # Frame 2 (t=2.0): only event 2 active, area = 1400*108 / (1920*1080) = 7.2917%
    assert result["mask_events_layer_0"] == 2
    assert result["total_frame_count"] == 3
    assert abs(result["union_mask_area_pct"]["max"] - 9.375) < 0.01
    assert result["active_mask_boxes_per_frame"]["max"] == 2

def test_parse_skips_dialogue_without_pos(tmp_path):
    """Dialogues without \pos are silently skipped."""
    module = load_module()
    ass_file = tmp_path / "no_pos.ass"
    ass_file.write_text(
        "[Script Info]\nPlayResX: 1920\nPlayResY: 1080\n\n"
        "[V4+ Styles]\nFormat: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,"
        "OutlineColour,BackColour,Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,"
        "Spacing,Angle,BorderStyle,Outline,Shadow,Alignment,MarginL,MarginR,"
        "MarginV,Encoding\n"
        "Style: Default,Arial,20,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,"
        "-1,0,0,0,100,100,0,0,1,2,2,2,0,0,0,1\n\n"
        "[Events]\nFormat: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text\n"
        "Dialogue: 0,0:00:00.00,0:00:02.00,Default,,0,0,0,,Subtitle text\n",
        encoding="utf-8",
    )
    geo = {"width": 1920, "height": 1080, "fps": 1.0, "frame_count": 2}
    result = module.compute_mask_coverage(ass_file, geo)
    # Text dialogue (no \pos) -> skipped; mask count = 0
    assert result["mask_events_layer_0"] == 0
    assert result["union_mask_area_pct"]["mean"] == 0.0


def test_frame_wise_union_uses_sweep_line(tmp_path):
    """Overlapping rectangles must compute real union area, not sum."""
    module = load_module()
    ass_file = tmp_path / "overlap.ass"
    # Two fully overlapping rectangles: same region
    ass_file.write_text(
        "[Script Info]\nPlayResX: 1920\nPlayResY: 1080\n\n"
        "[V4+ Styles]\nFormat: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,"
        "OutlineColour,BackColour,Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,"
        "Spacing,Angle,BorderStyle,Outline,Shadow,Alignment,MarginL,MarginR,"
        "MarginV,Encoding\n"
        "Style: Default,Arial,20,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,"
        "-1,0,0,0,100,100,0,0,1,2,2,2,0,0,0,1\n\n"
        "[Events]\nFormat: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text\n"
        "Dialogue: 0,0:00:00.00,0:00:02.00,Default,,0,0,0,,"
        "{\\p1}\\pos(0,0)m 0 0 l 960 0 l 960 540 l 0 540{\\p0}\n"
        "Dialogue: 0,0:00:00.00,0:00:02.00,Default,,0,0,0,,"
        "{\\p1}\\pos(0,0)m 0 0 l 960 0 l 960 540 l 0 540{\\p0}\n",
        encoding="utf-8",
    )
    geo = {"width": 1920, "height": 1080, "fps": 1.0, "frame_count": 2}
    result = module.compute_mask_coverage(ass_file, geo)
    # Full overlap: 960*540=518400 pixels. screen=2073600. pct=25%.
    # With 2 boxes but union is same 25%. boxes_per_frame_max=2 but area=max 25%
    assert result["union_mask_area_pct"]["mean"] == 25.0
    assert result["active_mask_boxes_per_frame"]["max"] == 2