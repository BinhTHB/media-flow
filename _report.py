import json, subprocess

# GT V2.2 stats
gt = json.loads(open('artifacts/round26/gt_v2_2.json', encoding='utf-8').read())
pos = [e for e in gt['entries'] if e['is_subtitle']]
neg = [e for e in gt['entries'] if not e['is_subtitle']]
print('=== GT V2.2 ===')
print(f'  Entries: {len(gt["entries"])} ({len(pos)} positives, {len(neg)} negatives)')
print(f'  Kind: {gt.get("kind", "unknown")}')
print(f'  Source: {gt.get("source", "unknown")}')
print()

# Evaluation V2.2
ev22 = json.loads(open('artifacts/round26/evaluation_v2_2.json', encoding='utf-8').read())
print('=== Evaluation V2.2 (round23 baseline) ===')
s = ev22.get('summary', ev22)
print(json.dumps(s, indent=2))
print()

# Evaluation V2.1 baseline
ev21 = json.loads(open('artifacts/round26/evaluation_v2_1_baseline.json', encoding='utf-8').read())
print('=== Evaluation V2.1 (round25 baseline) ===')
s = ev21.get('summary', ev21)
print(json.dumps(s, indent=2))
print()

# Blocked status
bs = json.loads(open('artifacts/round26/blocked_status.json', encoding='utf-8').read())
print('=== Blocked Status ===')
print(json.dumps(bs, indent=2))
print()

# Package report
pr = json.loads(open('artifacts/round26/package_report.json', encoding='utf-8').read())
print('=== Package Report ===')
print(json.dumps(pr, indent=2))
print()

# File inventory
fi = json.loads(open('artifacts/round26/file_inventory.json', encoding='utf-8').read())
print('=== File Inventory ===')
for k, v in sorted(fi.items()):
    print(f'  {k}: {v}')
print()

# Rule sweep
rs = json.loads(open('artifacts/round26/rule_sweep.json', encoding='utf-8').read())
print('=== Rule Sweep ===')
print(json.dumps(rs, indent=2))
print()

# FN diagnosis
fn = json.loads(open('artifacts/round26/fn_diagnosis.json', encoding='utf-8').read())
print('=== FN Diagnosis ===')
print(json.dumps(fn, indent=2)[:500])
print()

# Test suite
result = subprocess.run(['.venv/Scripts/python.exe', '-m', 'pytest', 'tests/', '-q', '--tb=no'], capture_output=True, text=True, timeout=120)
print('=== Test Suite ===')
lines = result.stdout.strip().split('\n')
# show last 3 lines
for l in lines[-3:]:
    print(l)
