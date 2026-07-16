# Phase G, Live End-to-End Gemini Activation

**Status:** IMPLEMENTED  
**Production Ready:** NO

## Preflight

- `GEMINI_API_KEY` was available without being printed.
- FFmpeg, FFprobe and Pillow were available.
- `gemini-3.1-flash-lite` and `gemini-3.5-flash` were available and support
  `generateContent`.
- Media used, without staging it: `test_videos/real_mini_benchmark/sample2.mp4`,
  `sample4.mp4`, and `sample5.mp4`.
- Runtime limits for the runs: RPM `4`, max retries `5`, max concurrency `1`,
  maximum escalation requests `2`.

## Live execution

The production composition root (`ApplicationBuilder`), real `JobOrchestrator`,
FFmpeg extraction/rendering, Gemini OCR, Gemini ASR and Gemini translation ran
on each sample.

| Sample | Gemini-only Run A | Normal Run B | OCR requests | ASR requests | Fallback |
|---|---:|---:|---:|---:|---|
| sample2 | PASS | PASS | 6 | 1 | not used |
| sample4 | PASS | PASS | 6 | 1 | not used |
| sample5 | PASS | PASS | 6 | 1 | not used |

Each completed job produced WAV audio, extracted frames, transcript,
translation, normalized/segmented subtitle artifacts and an MP4 render under
the untracked `output/live_gemini_phase_g/` directory.

## Runtime repair

Gemini-only mode exposed a provenance crash after an otherwise successful run:
raw Gemini providers expose request provenance as a list, while the fallback
wrapper exposes it as a mapping. `JobOrchestrator._provider_provenance()` now
only copies mappings and summarizes the raw request list separately. The
offline regression test `test_raw_gemini_provider_provenance_is_serializable`
covers this path.

## Provenance

Normal-mode jobs recorded:

```json
{
  "configured_provider": "gemini_vision",
  "effective_provider": "gemini_vision",
  "fallback_used": false,
  "exact_model_id": "gemini-3.1-flash-lite",
  "request_count": 6
}
```

ASR recorded `gemini_audio` as both configured and effective provider, one
request, no fallback, zero retries and inline/chunk metadata. No API key,
Authorization header or base64 media was written to artifacts.

## Controlled fallback

Offline controlled tests prove that quota failures route only to the configured
local fallback, corrupt WAV media does not fallback, and fallback provenance is
recorded. Gemini and local OCR/ASR outputs are never merged.

## Known limitations

- The live samples contain Vietnamese speech and mixed on-screen text. Asking
  for target Vietnamese can preserve or prefix already-Vietnamese text, which
  requires language-aware translation quality work rather than an ASR/OCR
  provider fallback.
- Phase G used the normal subtitle render flow. Dynamic source-cover safety
  remains governed by the existing static safety policy, including sample5
  `interval_6` `true_person_overlap_risk`.
- Live quality qualification, corpus review and SLA validation are outstanding;
  this does not revert the Gemini defaults.

## Verdict

```text
Gemini OCR live activation: ENABLED
Gemini ASR live activation: ENABLED
Gemini fallback policy: AVAILABLE
Live production readiness: NO
```
