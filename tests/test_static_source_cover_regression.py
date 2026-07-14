import json
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "tests" / "fixtures" / "static_source_cover_regression_manifest.json"
OUTPUT_DIR = ROOT / "output" / "static_source_cover_mvp_v2"
QA_DIR = ROOT / "output" / "static_source_cover_mvp_v2_qa"

def test_regression_manifest_exists():
    assert MANIFEST_PATH.exists()

def test_samples_match_regression_manifest():
    if not OUTPUT_DIR.exists():
        pytest.skip("Output directory for v2 does not exist. Run regenerate first.")
    
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)
        
    for sample, expected in manifest.items():
        sample_dir = OUTPUT_DIR / sample
        assert sample_dir.exists(), f"Sample directory for {sample} is missing."
        
        report_path = sample_dir / "debug" / "static_source_cover_report.json"
        assert report_path.exists(), f"Report for {sample} is missing."
        with open(report_path, "r", encoding="utf-8") as f_rep:
            report = json.load(f_rep)
            
        assert report["stable_region_found"] == expected["stable_region_found"], f"{sample} stable_region_found mismatch"
        assert report["rendered_count"] >= expected["min_rendered_count"], f"{sample} rendered_count mismatch"
        assert report["skipped_count"] <= expected["max_skipped_count"], f"{sample} skipped_count mismatch"
        
        for rel_file in expected["expected_output_files"]:
            file_path = sample_dir / rel_file
            assert file_path.exists(), f"Output file {rel_file} is missing for {sample}"
            
        cover_report_path = sample_dir / "_static_cover" / "debug" / "cover_source_subtitle_report.json"
        if cover_report_path.exists():
            with open(cover_report_path, "r", encoding="utf-8") as f_cov:
                cover_report = json.load(f_cov)
            for cap in cover_report["captions"]:
                assert cap.get("is_fixed_bottom_fallback") is False, "fixed_bottom_fallback must be False"
                
        assert (sample_dir / "subtitles" / "output.ass").exists()
        assert (sample_dir / "subtitles" / "output.srt").exists()
        
        assert (sample_dir / "videos" / "static_subtitle_burn.mp4").exists()
        
        prov_path = sample_dir / "debug" / "artifact_provenance_report.json"
        assert prov_path.exists()
        with open(prov_path, "r", encoding="utf-8") as f_prov:
            prov = json.load(f_prov)
            
        assert "input_video_path" in prov
        assert "input_video_sha256" in prov
        assert "resolution" in prov
        assert "duration" in prov
        assert "ocr_device_used" in prov
        
        assert prov.get("dynamic_subtitle_replacement") == "OUT_OF_SCOPE"
        assert prov.get("production_ready") == "NO"

def test_visual_qa_report_matches_regression():
    qa_report_path = QA_DIR / "static_source_cover_visual_qa_report.json"
    if not qa_report_path.exists():
        pytest.skip("QA report file is missing.")
        
    with open(qa_report_path, "r", encoding="utf-8") as f:
        qa_report = json.load(f)
        
    assert qa_report.get("overall_visual_verdict") == "PASS"
    assert qa_report.get("readiness") == "READY"
    assert qa_report.get("dynamic_subtitle_replacement") == "OUT_OF_SCOPE"
    assert qa_report.get("production_ready") == "NO"