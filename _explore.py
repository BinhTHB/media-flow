import json

tracks = json.loads(open('artifacts/round23/baseline_tracks_conf_0.30.json').read())
print(f'Total tracks: {len(tracks)}')
t = tracks[0]
print(f'Track keys: {list(t.keys())}')

# Find subtitle tracks
map_data = json.loads(open('artifacts/round23/gt_track_mapping.json').read())
sub_matches = [m for m in map_data['matches'] if m['is_subtitle']]
print(f'\nSubtitle matches: {len(sub_matches)}')

# Show first subtitle track
sub_gt_id = sub_matches[0]['gt_id']
sub_base_id = sub_matches[0]['base_id']
sub_track = [t for t in tracks if t['track_id'] == sub_base_id]
if sub_track:
    t = sub_track[0]
    print(f'\nPipeline track for {sub_gt_id}->{sub_base_id}:')
    print(f'  start_ms={t["start_ms"]}, end_ms={t["end_ms"]}')
    print(f'  group_count={len(t["groups"])}')
    g = t['groups'][0]
    print(f'  Group keys: {list(g.keys())}')
    print(f'  Sample group start: {json.dumps(g, indent=2)[:400]}')
