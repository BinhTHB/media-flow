import json
from pathlib import Path
import pytest

from scripts.run_quality_benchmark import load_manifest, classify, aggregate, count_overlaps, extract_metrics

def test_manifest_parsing(tmp_path):
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text("""
samples:
  - id: test_sample
    path: dummy.mp4
    source_language: en
    target_language: vi
    frame_interval: 1.0
""", encoding="utf-8")
    
    samples = load_manifest(manifest)
    assert len(samples) == 1
    assert samples[0]["id"] == "test_sample"

def test_count_overlaps():
    segments = [
        {"start": 0.0, "end": 2.0},
        {"start": 1.5, "end": 3.0},
        {"start": 4.0, "end": 5.0},
    ]
    overlaps, simultaneous = count_overlaps(segments)
    assert overlaps == 1
    assert simultaneous == 2


def test_extract_metrics_from_artifacts(tmp_path):
    job = tmp_path / "job"
    (job / "debug" / "ocr_frames").mkdir(parents=True)
    (job / "debug" / "subtitle_region_frames").mkdir()
    (job / "transcripts").mkdir()
    (job / "subtitles").mkdir()
    (job / "videos").mkdir()
    (job / "videos" / "output.mp4").write_bytes(b"video")
    (job / "subtitles" / "subtitle.srt").write_text("caption", encoding="utf-8")
    (job / "transcripts" / "segmented.json").write_text(
        json.dumps([{"start": 0.0, "end": 1.0}]), encoding="utf-8"
    )
    (job / "debug" / "debug_metadata.json").write_text(
        json.dumps({"quality_metrics": {"video_duration": 2.0, "source_text_leakage_count": 0, "quality_gate_status": "PASS"}}),
        encoding="utf-8",
    )
    (job / "debug" / "content_quality_report.json").write_text(
        json.dumps({"captions": [{"score": 0.8, "findings": []}]}),
        encoding="utf-8",
    )
    (job / "debug" / "ocr_frames" / "frame.json").write_text("{}", encoding="utf-8")
    metrics = extract_metrics(job)
    assert metrics["caption_count"] == 1
    assert metrics["content_quality_scores"] == [0.8]
    assert metrics["ocr_checkpoints"] == 1

def test_classify_and_aggregate():
    sample = {
        "id": "sample_1",
        "expected_min_caption_count": 2,
        "expected_max_source_leakage": 0,
        "expected_max_overlap": 0,
    }
    
    # 1. Failing case (missing output.mp4)
    metrics_fail = {
        "output_exists": False,
        "subtitle_exists": True,
        "caption_count": 3,
        "content_quality_scores": [0.8, 0.9],
        "placeholder_count": 0,
        "source_leakage_count": 0,
        "overlap_count": 0,
        "max_simultaneous": 1,
        "captions_beyond_duration": 0,
        "quality_gate_status": "PASS",
    }
    assert classify(sample, metrics_fail) == "fail"

    # 2. Conditional case (overlap fails expectations)
    metrics_cond = {
        "output_exists": True,
        "subtitle_exists": True,
        "caption_count": 3,
        "content_quality_scores": [0.8, 0.9],
        "placeholder_count": 0,
        "source_leakage_count": 0,
        "overlap_count": 1, # expected max is 0
        "max_simultaneous": 1,
        "captions_beyond_duration": 0,
        "quality_gate_status": "PASS",
    }
    assert classify(sample, metrics_cond) == "conditional"

    # 3. Passing case
    metrics_pass = {
        "output_exists": True,
        "subtitle_exists": True,
        "caption_count": 3,
        "content_quality_scores": [0.8, 0.9],
        "placeholder_count": 0,
        "source_leakage_count": 0,
        "overlap_count": 0,
        "max_simultaneous": 1,
        "captions_beyond_duration": 0,
        "quality_gate_status": "PASS",
    }
    assert classify(sample, metrics_pass) == "pass"

    # Aggregate
    results = [
        {"sample_id": "s1", "metrics": metrics_pass, "status": "pass"},
        {"sample_id": "s2", "metrics": metrics_cond, "status": "conditional"},
    ]
    report = aggregate(results)
    assert report["samples_passed"] == 1
    assert report["samples_failed"] == 0
    assert report["samples_conditional"] == 1
    assert report["average_caption_count"] == 3.0
    assert report["average_content_quality_score"] == pytest.approx(0.85)