import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = ROOT / "output" / "static_source_cover_mvp_v2"
MANIFEST_PATH = (
    ROOT / "tests" / "fixtures" / "static_source_cover_regression_manifest.json"
)


@pytest.fixture(scope="module")
def manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


@pytest.mark.parametrize("sample", ("sample2", "sample4", "sample5"))
def test_static_source_cover_baseline_artifacts(
    sample: str, manifest: dict
) -> None:
    if not ARTIFACT_ROOT.is_dir():
        pytest.skip("Static Source Cover MVP v2 artifacts are unavailable.")

    expected = manifest[sample]
    root = ARTIFACT_ROOT / sample
    report = json.loads(
        (root / "debug" / "static_source_cover_report.json").read_text(
            encoding="utf-8"
        )
    )
    provenance = json.loads(
        (root / "debug" / "artifact_provenance_report.json").read_text(
            encoding="utf-8"
        )
    )

    assert report["stable_region_found"] is expected["stable_region_found"]
    assert report["rendered_count"] >= expected["min_rendered_count"]
    assert report["skipped_count"] <= expected["max_skipped_count"]
    assert report["cover_overflow_risk"] == "low"
    render_report = json.loads(
        (
            root
            / "_static_cover"
            / "debug"
            / "cover_source_subtitle_report.json"
        ).read_text(encoding="utf-8")
    )
    assert all(
        item.get("is_fixed_bottom_fallback") is False
        for item in render_report["captions"]
    )

    for relative_path in expected["expected_output_files"]:
        artifact = root / relative_path
        assert artifact.is_file() and artifact.stat().st_size > 0

    for field in (
        "input_video_sha256",
        "resolution",
        "duration",
        "ocr_device_requested",
        "ocr_device_used",
    ):
        assert provenance[field]
    assert provenance["ocr_device_used"] == "gpu"
    assert provenance["subtitle_output_mode"] == "static_source_cover"
    assert provenance["dynamic_subtitle_replacement"] == "OUT_OF_SCOPE"
    assert provenance["fallback_generated"] is True
