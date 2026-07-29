from src.domain.entities.subtitle_region import SubtitleRegion

def test_subtitle_region_cache_preserves_provenance(tmp_path):
    # Create a region with full provenance
    original = SubtitleRegion(
        start_time=1.0,
        end_time=2.0,
        x=0.1,
        y=0.2,
        width=0.3,
        height=0.4,
        confidence=0.9,
        track_id='T001',
    )
    
    serialized = [original.to_dict(include_provenance=True)]
    restored = SubtitleRegion(**serialized[0])

    assert restored.track_id == 'T001'
    assert restored.track_id == original.track_id
