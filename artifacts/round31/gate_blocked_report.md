# Round 31 Gate Status — BLOCKED

## Result

`ONE-SHOT: FAIL` and `FINAL: FAIL`.

The run sent live, non-cached Gemini requests, then encountered the free-tier request quota
(`15 requests`) before all 42 GT V2.3 tracks could be resolved. The script recorded every attempt
and did not convert missing decisions into semantic `NO` decisions.

## Reproducer

```powershell
.\.venv\Scripts\python.exe scripts\offline_gate_r31.py *>&1 |
  Tee-Object -FilePath artifacts\round31\offline_gate_r31.log
```

Source: `artifacts/round31/offline_gate_r31.log`

```text
GeminiRateLimitError: ... generate_content_free_tier_requests, limit: 15
ONE-SHOT: FAIL
FINAL:    FAIL
attempts_total=60  tracks_total=42  retries=24
```

## What Was Measured

The first 18 resolved Gemini decisions yielded:

```text
TP=11, FP=2, FN=1, TN=4, unknown=24
precision=0.8462, recall=0.9167
```

Reproducer:

```powershell
python -c "import json; d=json.load(open('artifacts/round31/offline_gate_r31.json')); print(d['layer_b']['confusion_matrix'], d['layer_b']['precision'], d['layer_b']['recall'])"
```

This is not a valid 42-track classifier result because 24 tracks are unresolved. It cannot be
used to claim either that the `precision >= 0.85` target is met or that it is impossible to meet.

## Required Named Diagnostics Obtained Before Quota Exhaustion

| Track | GT | Gemini decision | Meaning |
|---|---|---|---|
| T013 (`negative:T013`) | negative | NO | canary was correctly rejected |
| T036 | negative | NO | Gemini correctly rejected the band/discriminator survivor |
| T053 | negative | YES | real classifier FP |
| T126 | negative | YES | real classifier FP |
| T041 | legacy negative | NO | classifier result captured |
| T105 | legacy negative | NO | classifier result captured |
| T055 | legacy negative | NO | classifier result captured |

Reproducer:

```powershell
Select-String -Pattern '"track_id": "T036"|"track_id": "T053"|"track_id": "T126"|"track_id": "negative:T013"' artifacts\round31\gate_attempt_events_r31.jsonl
```

## Why Round 31 Is Not Closed

1. Gemini was not able to resolve all 42 tracks under the available quota.
2. The one-shot outcome is FAIL due to missing decisions.
3. The partial result contains one classifier false negative (T003) and two classifier false
   positives (T053, T126), so it is not acceptable to proceed to E2E render or packaging.
4. No E2E round31 output was rendered. No product-quality precision/recall or mask coverage was
   invented from round30 output.

## Next Required Action

Obtain a Gemini quota that permits at least 42 primary requests (with up to 18 explicit retries
inside the existing 60-request budget), then run the same command once. Do not retry this
quota-exhausted run as if it were a clean one-shot.
