import copy
from pathlib import Path
import pytest
from unittest.mock import patch, MagicMock

from scripts.run_g514_sample2 import (
    G514ValidationError,
    validate_completed_stages,
    reconstruct_v5,
    geometry_frame_times,
    build_v4_batch_checkpoint,
    validate_v4_batch_checkpoint,
    translation_reuse,
    select_eligible_render_events,
    generate_ass_file,
    build_render_checkpoint,
    validate_render_checkpoint,
    technical_qa,
    validate_visual_qa_result,
    validate_visual_qa_batch,
    event_timing_hash,
)

def _events(count=15):
    return [{
        'event_id': f'merged_0:e_{i}',
        'candidate_id': 'merged_0',
        'start_seconds_in_source': float(i),
        'end_seconds_in_source': float(i) + 1.0,
        'representative_box_2d': [700, 100, 800, 900],
    } for i in range(count)]

def _replay(events):
    return {
        'video_sha256': 'video_hash',
        'v3_event_timing_sha256': event_timing_hash(events),
        'provenance': [{
            'stage': 'v5_transcription',
            'model': 'gemini-3.1-flash-lite',
            'provider': 'gemini_video_direct',
            'schema_version': 'v1',
            'validator_version': 'v1',
        }],
        'accepted_events': [{
            'event_id': e['event_id'],
            'text_verbatim': f'Text verbatim {i}',
            'start_seconds_in_source': e['start_seconds_in_source'],
            'end_seconds_in_source': e['end_seconds_in_source'],
        } for i, e in enumerate(events)]
    }

def _manifest(events):
    return {
        'video_sha256': 'video_hash',
        'identity': {'video_sha256': 'video_hash', 'model': 'gemini-3.1-flash-lite'},
        'artifacts': {
            'stage_status': {
                'V1': {'status': 'COMPLETED'},
                'V2': {'status': 'COMPLETED'},
                'V3': {'status': 'COMPLETED'},
            },
            'checkpoints': {'temporal_events': {'events': events}}
        }
    }

def test_validate_completed_stages_success():
    events = _events(15)
    manifest = _manifest(events)
    assert validate_completed_stages(manifest, 'video_hash') == events

def test_validate_completed_stages_fails():
    events = _events(15)
    
    # Video SHA mismatch
    manifest = _manifest(events)
    with pytest.raises(G514ValidationError, match='BASELINE_IDENTITY_MISMATCH'):
        validate_completed_stages(manifest, 'wrong_hash')
        
    # Wrong model
    manifest = _manifest(events)
    manifest['identity']['model'] = 'wrong_model'
    with pytest.raises(G514ValidationError, match='BASELINE_MODEL_MISMATCH'):
        validate_completed_stages(manifest, 'video_hash')
        
    # Missing stage
    manifest = _manifest(events)
    manifest['artifacts']['stage_status']['V2']['status'] = 'FAILED'
    with pytest.raises(G514ValidationError, match='BASELINE_V2_NOT_COMPLETED'):
        validate_completed_stages(manifest, 'video_hash')
        
    # Wrong event count
    bad_events = _events(14)
    manifest = _manifest(bad_events)
    with pytest.raises(G514ValidationError, match='BASELINE_REQUIRES_EXACTLY_15_EVENTS'):
        validate_completed_stages(manifest, 'video_hash')

def test_reconstruct_v5_success():
    events = _events(15)
    replay = _replay(events)
    texts = reconstruct_v5(events, replay, 'video_hash')
    assert len(texts) == 15
    assert texts['merged_0:e_0'] == 'Text verbatim 0'

def test_reconstruct_v5_fails():
    events = _events(15)
    replay = _replay(events)
    
    # Tampered timing
    replay['accepted_events'][0]['start_seconds_in_source'] += 0.5
    with pytest.raises(G514ValidationError, match='V5_PROVENANCE_RECONSTRUCTION_FAILED'):
        reconstruct_v5(events, replay, 'video_hash')
        
    # Mismatch video SHA
    replay = _replay(events)
    with pytest.raises(G514ValidationError, match='V5_PROVENANCE_RECONSTRUCTION_FAILED'):
        reconstruct_v5(events, replay, 'wrong_hash')

def test_geometry_frame_times():
    # short event (<0.8s)
    e1 = {'start_seconds_in_source': 1.0, 'end_seconds_in_source': 1.5}
    assert len(geometry_frame_times(e1)) == 1
    
    # medium event (<=1.5s)
    e2 = {'start_seconds_in_source': 1.0, 'end_seconds_in_source': 2.2}
    assert len(geometry_frame_times(e2)) == 2
    
    # long event (>1.5s)
    e3 = {'start_seconds_in_source': 1.0, 'end_seconds_in_source': 3.0}
    assert len(geometry_frame_times(e3)) == 3

def test_v4_batch_checkpoint():
    batch = _events(2)
    geometry = [{'event_id': e['event_id'], 'needs_review': False} for e in batch]
    checkpoint = build_v4_batch_checkpoint(batch, geometry, 'video_hash')
    assert validate_v4_batch_checkpoint(checkpoint, batch, 'video_hash')
    
    # wrong video hash
    assert not validate_v4_batch_checkpoint(checkpoint, batch, 'wrong_hash')
    
    # wrong events in batch
    other_batch = _events(3)[1:]
    assert not validate_v4_batch_checkpoint(checkpoint, other_batch, 'video_hash')

def test_translation_reuse_complete():
    events = _events(15)
    checkpoint = {
        'input_v5_sha256': 'v5_hash',
        'provenance': [{
            'stage': 'translation',
            'model': 'gemini-3.1-flash-lite',
            'provider': 'gemini_video_direct',
            'schema_version': 'v1',
            'validator_version': 'v1',
        }],
        'events': [{
            'event_id': e['event_id'],
            'start_seconds_in_source': e['start_seconds_in_source'],
            'end_seconds_in_source': e['end_seconds_in_source'],
            'translated_text': f'Dịch {i}'
        } for i, e in enumerate(events)]
    }
    reused, rerun_ids, status_mark = translation_reuse(events, checkpoint, 'v5_hash')
    assert len(reused) == 15
    assert not rerun_ids
    assert status_mark is None

def test_translation_reuse_fails_partial():
    events = _events(15)
    checkpoint = {
        'input_v5_sha256': 'v5_hash',
        'provenance': [{
            'stage': 'translation',
            'model': 'gemini-3.1-flash-lite',
            'provider': 'gemini_video_direct',
            'schema_version': 'v1',
            'validator_version': 'v1',
        }],
        'events': [{
            'event_id': e['event_id'],
            'start_seconds_in_source': e['start_seconds_in_source'],
            'end_seconds_in_source': e['end_seconds_in_source'],
            'translated_text': f'Dịch {i}'
        } for i, e in enumerate(events)]
    }
    
    # Tamper one event translation (make it empty)
    checkpoint['events'][0]['translated_text'] = ''
    reused, rerun_ids, status_mark = translation_reuse(events, checkpoint, 'v5_hash')
    assert len(reused) == 14
    assert rerun_ids == ['merged_0:e_0']
    assert status_mark == 'TRANSLATION_PROVENANCE_RECONSTRUCTION_FAILED'

def test_select_eligible_render_events():
    events = _events(2)
    geometry = [
        {'event_id': 'merged_0:e_0', 'needs_review': False, 'representative_box_2d': [1,2,3,4]},
        {'event_id': 'merged_0:e_1', 'needs_review': True, 'representative_box_2d': [5,6,7,8]},
    ]
    translations = {'merged_0:e_0': 'text 0', 'merged_0:e_1': 'text 1'}
    selected = select_eligible_render_events(events, geometry, translations)
    assert len(selected) == 1
    assert selected[0]['event_id'] == 'merged_0:e_0'

def test_render_checkpoint_logic(tmp_path):
    input_video = tmp_path / 'in.mp4'
    output_video = tmp_path / 'out.mp4'
    ass_path = tmp_path / 'sub.ass'
    input_video.write_bytes(b'in')
    output_video.write_bytes(b'out')
    ass_path.write_bytes(b'ass')
    
    geometry = [{'event_id': 'merged_0:e_0'}]
    translations = {'merged_0:e_0': 'text'}
    
    checkpoint = build_render_checkpoint(input_video, output_video, geometry, translations, ass_path)
    assert validate_render_checkpoint(checkpoint, input_video, output_video, geometry, translations, ass_path)
    
    # mismatched output video
    output_video.write_bytes(b'changed')
    assert not validate_render_checkpoint(checkpoint, input_video, output_video, geometry, translations, ass_path)

@patch('scripts.run_g514_sample2._probe')
def test_technical_qa_success(mock_probe, tmp_path):
    mock_probe.side_effect = lambda path: {
        'format': {'duration': '10.0'},
        'streams': [
            {'codec_type': 'video', 'codec_name': 'h264', 'width': 720, 'height': 1280},
            {'codec_type': 'audio', 'codec_name': 'aac'}
        ]
    }
    input_video = tmp_path / 'in.mp4'
    output_video = tmp_path / 'out.mp4'
    input_video.write_bytes(b'in')
    output_video.write_bytes(b'out')
    
    rendered = [{'event_id': 'e1', 'representative_box_2d': [100, 100, 200, 200], 'translated_text': 'vi'}]
    result = technical_qa(input_video, output_video, rendered)
    assert result['verdict'] == 'PASS'
    assert result['video_stream'] == 'h264'
    assert result['audio_stream'] == 'aac'
    assert result['rendered_events'] == 1

@patch('scripts.run_g514_sample2._probe')
def test_technical_qa_resolution_mismatch(mock_probe, tmp_path):
    mock_probe.side_effect = lambda path: {
        'format': {'duration': '10.0'},
        'streams': [
            {'codec_type': 'video', 'codec_name': 'h264', 'width': 1280, 'height': 720},
            {'codec_type': 'audio', 'codec_name': 'aac'}
        ]
    }
    input_video = tmp_path / 'in.mp4'
    output_video = tmp_path / 'out.mp4'
    input_video.write_bytes(b'in')
    output_video.write_bytes(b'out')
    
    rendered = [{'event_id': 'e1', 'representative_box_2d': [100, 100, 200, 200], 'translated_text': 'vi'}]
    result = technical_qa(input_video, output_video, rendered)
    assert result['verdict'] == 'FAIL'
    assert 'RESOLUTION_NOT_720X1280' in result['reasons']

@pytest.mark.parametrize('result, expected_status', [
    ({'status': 'PASS', 'source_text_visible': False, 'banner_visible': False, 'has_duplicate': False, 'has_overflow': False, 'has_damage': False}, 'PASS'),
    ({'status': 'PASS', 'source_text_visible': True, 'banner_visible': False, 'has_duplicate': False, 'has_overflow': False, 'has_damage': False}, 'FAIL'),
    ({'status': 'PASS', 'source_text_visible': False, 'banner_visible': True, 'has_duplicate': False, 'has_overflow': False, 'has_damage': False}, 'FAIL'),
    ({'status': 'PASS', 'source_text_visible': False, 'banner_visible': False, 'has_duplicate': True, 'has_overflow': False, 'has_damage': False}, 'FAIL'),
    ({'status': 'PASS', 'source_text_visible': False, 'banner_visible': False, 'has_duplicate': False, 'has_overflow': True, 'has_damage': False}, 'FAIL'),
    ({'status': 'PASS', 'source_text_visible': False, 'banner_visible': False, 'has_duplicate': False, 'has_overflow': False, 'has_damage': True}, 'FAIL'),
    ({'status': 'REVIEW_REQUIRED', 'source_text_visible': False, 'banner_visible': False, 'has_duplicate': False, 'has_overflow': False, 'has_damage': False}, 'REVIEW_REQUIRED'),
    ({'status': 'INVALID'}, 'REVIEW_REQUIRED'),
])
def test_validate_visual_qa_result(result, expected_status):
    assert validate_visual_qa_result(result)['status'] == expected_status

def test_validate_visual_qa_batch():
    results = [
        {'status': 'PASS', 'source_text_visible': False, 'banner_visible': False, 'has_duplicate': False, 'has_overflow': False, 'has_damage': False},
        {'status': 'REVIEW_REQUIRED', 'source_text_visible': False, 'banner_visible': False, 'has_duplicate': False, 'has_overflow': False, 'has_damage': False},
    ]
    assert validate_visual_qa_batch(results) == 'REVIEW_REQUIRED'
    
    results[1]['status'] = 'FAIL'
    assert validate_visual_qa_batch(results) == 'FAIL'
    
    results[1]['status'] = 'PASS'
    assert validate_visual_qa_batch(results) == 'PASS'
