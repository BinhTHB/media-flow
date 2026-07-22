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
        classification_id='class_123',
        is_subtitle=True,
        classifier_model='gemini-3.5-flash-lite',
        track_id='T001',
        proposal_ids=['prop_1'],
        category='dialogue_subtitle',
        classifier_confidence=0.95
    )
    
    serialized = [original.to_dict(include_provenance=True)]
    restored = SubtitleRegion(**serialized[0])

    assert restored.classification_id == 'class_123'
    assert restored.is_subtitle is True
    assert restored.classifier_model == 'gemini-3.5-flash-lite'
    assert restored.track_id == 'T001'
    assert restored.proposal_ids == ['prop_1']
    assert restored.category == 'dialogue_subtitle'
    assert restored.classifier_confidence == 0.95
