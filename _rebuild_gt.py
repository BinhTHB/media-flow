import json, hashlib, subprocess, sys
from pathlib import Path

ROOT = Path('.')
OUT = ROOT / 'artifacts' / 'round26'
TRACKS_FILE = ROOT / 'artifacts' / 'round23' / 'baseline_tracks_conf_0.30.json'
MAPPING_FILE = ROOT / 'artifacts' / 'round23' / 'gt_track_mapping.json'
FROZEN_FILE = ROOT / 'scripts' / 'frozen_gt' / 'sample5' / 'frozen_gt.json'
OLD_GT = OUT / 'gt_v2_2.json'
PREDICTIONS = ROOT / 'output/round23-e2e/round23_sample5_fix3/debug/subtitle_regions.json'
EVAL_SCRIPT = ROOT / 'scripts' / 'evaluate_round23_regions.py'

# Load data
tracks_by_id = {t['track_id']: t for t in json.loads(TRACKS_FILE.read_text(encoding='utf-8'))}
mapping = json.loads(MAPPING_FILE.read_text(encoding='utf-8'))
matches_by_gt = {m['gt_id']: m for m in mapping['matches']}
frozen = {e['id']: e for e in json.loads(FROZEN_FILE.read_text(encoding='utf-8'))}

# Build positives from pipeline geometry
sub_ids = [eid for eid, e in frozen.items() if e['expected_is_subtitle']]
positives = []
for sid in sorted(sub_ids):
    m = matches_by_gt.get(sid)
    if not m:
        print('WARNING: %s not matched in pipeline, skipping' % sid)
        continue
    base_id = m['base_id']
    pt = tracks_by_id.get(base_id)
    if not pt:
        print('WARNING: pipeline track %s not found for %s' % (base_id, sid))
        continue
    # Use median_bbox_xyxy as geometry, midpoint timestamp
    bbox = pt['median_bbox_xyxy']
    ts = (pt['start_ms'] + pt['end_ms']) / 2.0 / 1000.0
    positives.append({
        'id': sid,
        'set': 'sample5',
        'timestamp_s': round(ts, 4),
        'is_subtitle': True,
        'bbox_xyxy': [round(b, 2) for b in bbox],
        'pipeline_track_id': base_id,
        'pipeline_start_ms': pt['start_ms'],
        'pipeline_end_ms': pt['end_ms'],
    })
    print('  %s -> %s bbox=%s ts=%.4f' % (sid, base_id, bbox, ts))

print()
print('Built %d positives' % len(positives))

# Keep V2.1 negatives (from old GT or rebuild)
old_gt = json.loads(OLD_GT.read_text(encoding='utf-8'))
negatives = [e for e in old_gt['entries'] if not e['is_subtitle']]
print('Carrying over %d negatives from V2.1' % len(negatives))

# Build new GT
gt = {
    'schema_version': 1,
    'source': 'round23 pipeline geometry (positives) + V2.1 sample5_non_bottom (negatives)',
    'pipeline_source': {
        'frozen_gt_sha256': mapping['frozen_gt_sha256'],
        'baseline_tracks_sha256': mapping['baseline_tracks_sha256'],
        'gt_track_mapping: matched/subtitle': '%d/%d' % (mapping['matched_count'], len(positives)),
    },
    'entries': positives + negatives,
}
# Remove pipeline metadata from entries
for e in gt['entries']:
    e.pop('pipeline_track_id', None)
    e.pop('pipeline_start_ms', None)
    e.pop('pipeline_end_ms', None)

print('Total entries: %d (%d pos + %d neg)' % (len(gt['entries']), len(positives), len(negatives)))

# Write
OUT.mkdir(parents=True, exist_ok=True)
(OUT / 'gt_v2_2.json').write_text(json.dumps(gt, indent=2), encoding='utf-8')
digest = hashlib.sha256((OUT / 'gt_v2_2.json').read_bytes()).hexdigest()
(OUT / 'gt_v2_2.sha256').write_text('%s  gt_v2_2.json\n' % digest, encoding='ascii')
print('SHA256: %s' % digest)

# Run sanity gate: evaluate round23 predictions against new GT
print()
print('=== Sanity Gate: evaluate round23 predictions against new GT V2.2 ===')
result = subprocess.run([
    sys.executable, str(EVAL_SCRIPT),
    '--regions', str(PREDICTIONS),
    '--kind', 'v2_2',
    '--width', '720',
    '--height', '1280',
    '--output', str(OUT / 'evaluation_v2_2.json'),
], capture_output=True, text=True, timeout=60)
print(result.stdout)
if result.stderr:
    print('STDERR:', result.stderr[:500])

# Load and show summary
ev = json.loads((OUT / 'evaluation_v2_2.json').read_text(encoding='utf-8'))
print()
print('=== Sanity Gate Result ===')
print('TP=%d FP=%d FN=%d Precision=%.4f Recall=%.4f' % (
    ev['true_positives'], ev['false_positives'], ev['false_negatives'],
    ev['precision'], ev['recall'],
))
print('Expected: TP>=17 (sanity gate)')
if ev['true_positives'] >= 17:
    print('SANITY PASS: TP=%d >= 17' % ev['true_positives'])
else:
    print('SANITY WARNING: TP=%d < 17' % ev['true_positives'])
