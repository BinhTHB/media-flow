import json, hashlib, subprocess, sys
from pathlib import Path

ROOT = Path('.')
OUT = ROOT / 'artifacts' / 'round26'
MASK_FILE = ROOT / 'artifacts' / 'round23' / 'gt_mask_geometry.json'
FROZEN_FILE = ROOT / 'scripts' / 'frozen_gt' / 'sample5' / 'frozen_gt.json'
OLD_GT = OUT / 'gt_v2_2.json'
PREDICTIONS = ROOT / 'output/round23-e2e/round23_sample5_fix3/debug/subtitle_regions.json'
EVAL_SCRIPT = ROOT / 'scripts' / 'evaluate_round23_regions.py'

# Load data
mask = json.loads(MASK_FILE.read_text(encoding='utf-8'))
mask_by_id = {t['track_id']: t for t in mask['tracks']}
frozen = {e['id']: e for e in json.loads(FROZEN_FILE.read_text(encoding='utf-8'))}

# Build positives from mask geometry (parse_ass_dialogues output)
sub_ids = [eid for eid, e in frozen.items() if e['expected_is_subtitle']]
positives = []
for sid in sorted(sub_ids):
    mt = mask_by_id.get(sid)
    if not mt:
        print('WARNING: %s not in mask geometry, skipping' % sid)
        continue
    # Remove 8px padding (same as _frozen_entries())
    bbox = [mt['bbox_xyxy'][0] + 8, mt['bbox_xyxy'][1] + 8,
            mt['bbox_xyxy'][2] - 8, mt['bbox_xyxy'][3] - 8]
    ts = (mt['start_time_s'] + mt['end_time_s']) / 2.0
    positives.append({
        'id': sid,
        'set': 'sample5',
        'timestamp_s': round(ts, 4),
        'is_subtitle': True,
        'bbox_xyxy': bbox,
    })
    print('  %s: bbox=%s ts=%.4f dur=%.3f' % (sid, bbox, ts, mt['duration_s']))

print()
print('Built %d positives' % len(positives))

# Keep V2.1 negatives
old_gt = json.loads(OLD_GT.read_text(encoding='utf-8'))
negatives = [e for e in old_gt['entries'] if not e['is_subtitle']]
print('Carrying over %d negatives from V2.1' % len(negatives))

# Build new GT
gt = {
    'schema_version': 1,
    'source': 'mask geometry from parse_ass_dialogues (positives) + V2.1 sample5_non_bottom (negatives)',
    'pipeline_source': {
        'mask_geometry_sha256': hashlib.sha256(MASK_FILE.read_bytes()).hexdigest(),
        'frozen_gt_sha256': mask['frozen_gt_sha256'],
    },
    'entries': positives + negatives,
}

print('Total entries: %d (%d pos + %d neg)' % (len(gt['entries']), len(positives), len(negatives)))

# Write
OUT.mkdir(parents=True, exist_ok=True)
(OUT / 'gt_v2_2.json').write_text(json.dumps(gt, indent=2), encoding='utf-8')
digest = hashlib.sha256((OUT / 'gt_v2_2.json').read_bytes()).hexdigest()
(OUT / 'gt_v2_2.sha256').write_text('%s  gt_v2_2.json\n' % digest, encoding='ascii')
print('SHA256: %s' % digest)

# Run evaluation
print()
print('=== Sanity Gate: evaluate round23 predictions against GT V2.2 ===')
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

# Show summary
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
