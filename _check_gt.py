import json

frozen = json.loads(open('scripts/frozen_gt/sample5/frozen_gt.json', encoding='utf-8').read())
sub_ids = [e['id'] for e in frozen if e['expected_is_subtitle']]
print('Frozen GT: %d total, %d subtitle' % (len(frozen), len(sub_ids)))
print('Subtitle ids: %s' % sorted(sub_ids))

map_data = json.loads(open('artifacts/round23/gt_track_mapping.json', encoding='utf-8').read())
matches_by_gt = {m['gt_id']: m for m in map_data['matches']}

unmatched = []
for sid in sorted(sub_ids):
    m = matches_by_gt.get(sid)
    if m:
        print('  %s -> %s (iou=%.4f)' % (sid, m['base_id'], m['max_iou']))
    else:
        print('  %s -> UNMATCHED' % sid)
        unmatched.append(sid)
print('Unmatched subtitles: %s' % unmatched)
