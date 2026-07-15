import json
import pytest
from src.interfaces.adapters.static_source_cover_artifact_adapter import (
    StaticSourceCoverArtifactAdapter,
)

def test_adapter_parses_sample_metadata(tmp_path) -> None:
    sample_dir = tmp_path / "sample_test"
    sample_dir.mkdir()
    (sample_dir / "debug").mkdir()
    (sample_dir / "_static_cover" / "debug").mkdir(parents=True)
    
    # Write dummy reports
    cover_data = {
        "captions": [
            {
                "interval_id": "interval_0",
                "time_range": {"start": 0.0, "end": 1.0},
                "source_subtitle_bbox": {"left": 10.0, "top": 20.0, "right": 100.0, "bottom": 60.0},
                "mask_bbox": {"left": 5.0, "top": 15.0, "right": 105.0, "bottom": 65.0},
                "keep_reject_reason": None,
                "static_region_person_overlap_override": True,
                "override_reason": "static_region_person_overlap_override",
            },
            {
                "interval_id": "interval_1",
                "time_range": {"start": 1.0, "end": 2.0},
                "source_subtitle_bbox": {"left": 10.0, "top": 20.0, "right": 100.0, "bottom": 60.0},
                "mask_bbox": {"left": 5.0, "top": 15.0, "right": 105.0, "bottom": 65.0},
                "keep_reject_reason": "true_person_overlap_risk",
                "static_region_person_overlap_override": False,
            }
        ]
    }
    interval_data = {
        "intervals": [
            {
                "interval_id": "interval_0",
                "selected_candidate": {
                    "normalized_text": "hello"
                }
            },
            {
                "interval_id": "interval_1",
                "selected_candidate": {
                    "normalized_text": "world"
                }
            }
        ]
    }
    prov_data = {
        "resolution": {"width": 720, "height": 1280}
    }
    
    with open(sample_dir / "_static_cover/debug/cover_source_subtitle_report.json", "w", encoding="utf-8") as f:
        json.dump(cover_data, f)
    with open(sample_dir / "debug/interval_decision_report.json", "w", encoding="utf-8") as f:
        json.dump(interval_data, f)
    with open(sample_dir / "debug/artifact_provenance_report.json", "w", encoding="utf-8") as f:
        json.dump(prov_data, f)
        
    adapter = StaticSourceCoverArtifactAdapter(str(sample_dir))
    intervals = adapter.load_intervals()
    
    assert len(intervals) == 2
    assert intervals[0].interval_id == "interval_0"
    assert intervals[0].source_text == "hello"
    assert intervals[0].source_bbox == {"left": 10.0, "top": 20.0, "right": 100.0, "bottom": 60.0}
    assert intervals[0].expanded_cover_bbox == {"left": 5.0, "top": 15.0, "right": 105.0, "bottom": 65.0}
    assert intervals[0].safety_status == "safe"
    assert intervals[0].metadata["static_region_person_overlap_override"] is True
    assert intervals[0].metadata["override_reason"] == "static_region_person_overlap_override"
    
    assert intervals[1].interval_id == "interval_1"
    assert intervals[1].source_text == "world"
    assert intervals[1].safety_status == "skipped"
    assert intervals[1].safety_reason == "true_person_overlap_risk"
    
    w, h = adapter.get_frame_dimensions()
    assert w == 720
    assert h == 1280