# Phase G, Live End-to-End Gemini Evidence Audit

**Status:** CORRECTED
**Phase F commit:** `63b5ff9`
**Phase G commit:** `0f4463c`
**Production Ready:** NO

## Scope and evidence limits

This report is based on retained untracked artifacts under
`output/live_gemini_phase_g/`. Run commands and process exit codes were not
persisted with those artifacts. Filesystem timestamps below are derived from
the output directories, not process logs. No visual or reference review was
performed.

Typed provider defaults are defined in `AppConfig` and `ConfigLoader`. The
untracked, user-owned `config/providers.yaml` supplies local runtime options
when present, but is not the source of the Phase G typed defaults and was not
changed by the Phase G commits.

## Verified live API runs

`request_count > 0`, zero cache hits, Gemini effective providers, and no
fallback establish these as non-cache Gemini API executions. Each job has
transcript, translated transcript, subtitle, debug metadata and rendered-video
artifacts, so these are main-pipeline renders, not qualification/shadow output.

| Sample | Input SHA256 | Run type | OCR provider | ASR provider | Fallback | Cache | Output video | Media validation | Safety result | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| sample2 | `122B2B6056205A425FFE2990219BCEB199D19DA2244CC1DA1F03712B6984A559` | LIVE API RUN, Gemini-only repair | `GeminiVisionOCRProvider`, 6 requests | `GeminiAudioSpeechProvider`, 1 request | false / false | 0 hits / 7 misses | `output/live_gemini_phase_g/run_a_gemini_only_sample2/job_repair/sample2_gemini_only_repair/videos/output.mp4` | GENERATED_AND_PARSEABLE | not evaluated | PASS |
| sample4 | `0A127863C6DB792B736A46AF0F78C899753BAD64A49D70867136610D81DAA95C` | LIVE API RUN, Gemini-only | `GeminiVisionOCRProvider`, 6 requests | `GeminiAudioSpeechProvider`, 1 request | false / false | 0 hits / 7 misses | `output/live_gemini_phase_g/run_a_gemini_only_sample4_5/job_sample4/sample4_gemini_only/videos/output.mp4` | GENERATED_AND_PARSEABLE | not evaluated | PASS |
| sample5 | `A3FEFB099E0C7B416F74AA9C76AA0C39F7561236317CE1185CF970BA6409F2D7` | LIVE API RUN, Gemini-only | `GeminiVisionOCRProvider`, 6 requests | `GeminiAudioSpeechProvider`, 1 request | false / false | 0 hits / 7 misses | `output/live_gemini_phase_g/run_a_gemini_only_sample4_5/job_sample5/sample5_gemini_only/videos/output.mp4` | GENERATED_AND_PARSEABLE | not independently verified | PASS |

The normal-mode sample2, sample4 and sample5 artifacts also record Gemini as
the effective OCR/ASR provider, no fallback and no cache hits. The initial
Gemini-only sample2 artifact has contradictory ASR provenance and zero ASR
requests, so it is classified as **unreliable/cache-like evidence**, not a live
PASS. The repaired sample2 run above is the canonical sample2 live evidence.

## Media validation

FFprobe parsed each listed output as an MP4-compatible container with H.264
video and AAC audio. Input and output resolution is `720x1280`; duration is
preserved exactly to FFprobe precision.

| Sample | Input duration | Output duration | Video / audio streams | Result |
|---|---:|---:|---|---|
| sample2 | `21.989297s` | `21.989297s` | H.264 / AAC | parseable, resolution and audio preserved |
| sample4 | `24.450612s` | `24.450612s` | H.264 / AAC | parseable, resolution and audio preserved |
| sample5 | `24.311293s` | `24.311293s` | H.264 / AAC | parseable, resolution and audio preserved |

## Main-pipeline evidence

Each verified job's `debug_metadata.json` records ASR and OCR stage timings,
non-zero Gemini request counts, translation inputs and outputs, normalization,
segmentation, placement and render timings. Its `transcripts/translated.json`
contains speech segments with populated `translated_text`. This demonstrates:

```text
Gemini OCR / Gemini ASR
  -> merge and translation
  -> subtitle generation and placement
  -> FFmpeg render
```

The audit did not establish visual correctness or reference equivalence.

## Provenance repair and controlled fallback

The initial Gemini-only sample2 run exposed a crash because raw Gemini
provenance is a list while fallback provenance is a mapping. The parser now
preserves list entries under `request_provenance`, copies mappings, accepts
missing and empty provenance, and rejects malformed provenance rather than
turning it into success. Offline tests cover rate-limit fallback, invalid-media
fail-closed behavior, and all supported provenance shapes.

The fallback evidence is **OFFLINE FAKE TRANSPORT**, not a live local-fallback
run. It verifies quota fallback to Paddle/Faster-Whisper without merging output,
and confirms invalid media does not trigger fallback.

## sample5 safety evidence

No Phase G live output artifact contains `interval_6`,
`true_person_overlap_risk`, or `SKIPPED`. The retained sample5 placement report
has `source_visibility_not_evaluated_count: 9` and
`source_overlap_count: 0`. Therefore the historical
`sample5 interval_6 SKIPPED true_person_overlap_risk` decision is retained only
as earlier-strategy evidence, not claimed as verified by this Phase G run.

## Verdict

```text
OFFLINE END-TO-END: PASS
LIVE GEMINI API: PASS
LIVE END-TO-END: PASS
Production Ready: NO
```
