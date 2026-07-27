# Incident Report v30 — Accuracy Discrepancy Analysis

## Classification

These are reporting-carelessness incidents, not fabricated logs. The root failure was publishing
figures without reproducing them from the final ZIP and without reporting all attempts.

Every numerical statement below has a reproducer.

## 1. ASS mask count, reported 235 while ZIP count is 273

### What was done

`scripts/verify_e2e_r30.py` used:

```python
mask_count = ass_text.count(r"\p1")
```

The report copied the value `235` from the later
`artifacts/round30/e2e_self_check.json`.

### Evidence and reproducer

```powershell
# Earlier E2E log, the ASS that was packaged:
Select-String -Pattern '"ass_mask_count"' artifacts/round30/e2e_self_check.log
# Output: "ass_mask_count": 273,

# Later self-check JSON, a different rerender:
python -c "import json; print(json.load(open('artifacts/round30/e2e_self_check.json'))['checks']['ass_mask_count'])"
# Output: 235
```

The current working-tree ASS also yields 235:

```powershell
python -c "from pathlib import Path; p=Path('output/round30-sample5/round30-sample5/subtitles/subtitle.ass'); print(p.read_text(encoding='utf-8').count(r'\p1'))"
# Output: 235
```

### Cause

Two E2E attempts with the same job ID produced different ASS files. The report copied the later
working-tree result and did not count the ASS embedded in the ZIP. It therefore reported 235
against a ZIP containing 273. The count operation itself was not the bug, artifact provenance was.

## 2. Gate log selection hid the initial FAIL and budget was reported as 14 instead of 15

### Evidence and reproducer

```powershell
python -c "import json; d=json.load(open('artifacts/round30/offline_gate_r30.json')); print('requests:', d['requests'])"
# Output: requests: 14

Select-String -Pattern 'Missing decision|RETRY MISSING IDS SOLO|"requests"|GATE:' artifacts/round30/offline_gate_r30_retry.log
# Output includes:
# LOUD-WARN: Missing decision for track T133. Fail-closed.
# RETRY MISSING IDS SOLO: ['T133']
# "requests": 15,
# GATE: PASS
```

### Cause

The first gate result was overwritten or represented only by the success-oriented JSON summary.
The separate retry log shows T133 was missing, so the initial one-shot result was FAIL and the
actual budget was 14 initial requests plus one solo retry, 15. The report cited the stale
`offline_gate_r30.json` value (14), omitted the retry log, and did not state the initial FAIL.

## 3. Claimed three discriminator tests PASS although the file was absent from the packaged commit

### Evidence and reproducer

```powershell
git log --oneline -- tests/test_pixel_variance_discriminator.py
# Output: 8b8e366 Enforce bottom-band gate in V21 pipeline

git merge-base --is-ancestor b8db68ec 8b8e366
# Exit code: 0 (the test was added after the package HEAD)

git log --diff-filter=A --name-only --format="" -- tests/test_pixel_variance_discriminator.py
# Output: tests/test_pixel_variance_discriminator.py
```

`package_report.json` records package HEAD `b8db68ec`, while the test file was introduced by
later commit `8b8e366`. Thus it could not be present in the code state identified by the package.

### Cause

The report described a planned or later local test file as if it had been present and executed in
the packaged v30 state. No supporting pytest output for that state was submitted. This was an
unsupported PASS claim.

## 4. `package_report.git_sha` identifies round 29

### Evidence and reproducer

```powershell
python -c "import json; print(json.load(open('artifacts/round30/package_report.json'))['git_sha'])"
# Output: b8db68ecb86d6f288a5aa92c69a618a42d8884c0

git show -s --format="%H %s" b8db68ec
# Output: b8db68ec... TASK 4: Fix usage ledger serialization, run E2E GPU pipeline...

git show -s --format="%H %s" 650ae8f
# Output: 650ae8f... Record Round 30 gate evidence
```

`scripts/package_round30.py` correctly invokes `git rev-parse HEAD`, but packaging occurred
before the round-30 evidence changes were committed. It therefore embedded the old round-29 HEAD.

### Cause

The package was built from a dirty working tree, before committing the round-30 work. The script
reported Git's true HEAD, but the report incorrectly treated it as the round-30 revision.

## Corrective controls for v31

1. Every reported number must include its command, source file, and output line.
2. Gate artifacts preserve every attempt. They report `ONE-SHOT` separately from `FINAL`, plus
   `attempts_total` and `tracks_total`.
3. Packaging is performed only after a clean commit. The package log prints `git rev-parse HEAD`
   before archive creation and verification compares it to `package_report.git_sha`.
4. No test is reported as passing unless its test file and captured pytest output are both included
   in the packaged revision.
