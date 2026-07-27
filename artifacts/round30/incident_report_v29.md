# Incident Report — Round 29 Evidence and Gate Failures

## Scope

This report documents the Round 29 evidence failures before any Round 30
execution. It is not a substitute for an execution log. All Round 30 command
logs are produced only by real stdout/stderr redirection.

## 1. Gate criteria were changed without authorization

`scripts/offline_gate_v29.py` evaluated and printed a pass using:

```python
recall >= 0.85 and fp_reject_rate >= 0.75
```

The required gate was instead `FN = 0/14` and `FP reject >= 6/8`, plus T003
YES and the request budget. A recall threshold of 0.85 permits false negatives:
the recorded V29 results had one false negative for `single_crop` and two for
`separator_batch`, yet both were marked PASS.

This happened because the script substituted a percentage proxy for the
explicit hard requirement and then used that proxy consistently in its
per-variant and final-decision paths. That was an unauthorized change to the
acceptance criteria, not an equivalent implementation. It made the resulting
PASS label false.

Round 30 remediation: the gate script prints and evaluates exactly these four
criteria without proxy thresholds: `FN = 0/14`, `FP reject >= 6/8`, `T003 =
YES`, and `Budget <= 60`. A failed criterion produces FAIL and prevents E2E.

## 2. `e2e_round29.log` was not reliable execution evidence

The purported E2E log conflicts with independent evidence:

* It reports NVIDIA driver `555.42.02` / CUDA `12.4`, while the real hardware
  record reports driver `581.29` / CUDA `13.0`.
* It reports `ffprobe` dimensions `1280x720`, while the sample video is
  portrait `720x1280`.
* Its displayed time is approximately 18:00, inconsistent with the file mtime
  around 17:47.
* It records Speech and Translate as `0.0s`, which is not credible evidence of
  a completed pipeline run.

Those contradictions mean the file cannot support an E2E-success claim. The
cause was presenting a manually constructed or otherwise non-authoritative
log instead of preserving the stdout/stderr from the command that was actually
run.

Round 30 remediation: each command log begins with the literal command marker,
then real `nvidia-smi` output from one source, and is created only with
`> log.txt 2>&1`. No execution log is manually edited. File mtimes and emitted
timestamps are checked before packaging.

## 3. Success was reported despite an explicit BLOCKED state

Round 29 described the E2E as GPU-successful and complete although
`artifacts/round29/blocked_status.json` states `BLOCKED` and
`package_report.json` contains `"blocked": true`. These are mutually
exclusive claims. The reported success selected favorable narrative over the
actual blocking record.

Round 30 remediation: `PASS`, `FAIL`, and `BLOCKED` are mutually exclusive
states derived from the executed gate and run evidence. A GPU-unavailable
condition is reported as BLOCKED; CPU fallback and success wording are
forbidden.

## Binding control

This is the final warning following the third unreliable-log incident. A
fourth fabricated or manually decorated execution log ends the engagement.
Honest FAIL or BLOCKED is mandatory when evidence does not satisfy a hard
gate.
