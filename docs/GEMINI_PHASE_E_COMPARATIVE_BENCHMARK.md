# Gemini Phase E Comparative Benchmark

## Status

- Phase E implementation: **PASS**
- Live benchmark: **BLOCKED/INCOMPLETE**
- Provider decision: **DEFERRED**
- Production Ready: **NO**

## Scope

The offline runner compares the planned OCR modes, `paddle_default`,
`gemini_full_frame_only`, and `gemini_full_frame_crop_retry`, and ASR modes,
`faster_whisper_small`, `gemini_full_audio`, and optional
`gemini_chunk_fallback`. It validates a hash-pinned JSON corpus manifest and
creates redacted artifacts only.

No live provider result exists. Corpus references are not approved for live
benchmarking, so live execution is deliberately blocked and requires reviewed
OCR and ASR references.

## Measurements

- OCR: detection precision, recall, F1, exact and normalized text rates, CER,
  bbox IoU, bbox center error, false positives, and misses.
- ASR: CER, WER, segment count, coverage, timestamp MAE, and exact counts for
  `因果` and `英国`.
- Safety: `sample5` `interval_6` remains `SKIPPED` with
  `true_person_overlap_risk` when present in corpus metadata.

## Limitations

- The runner does not call providers, even with `--live`.
- No comparative quality, latency, quota, or production decision can be made
  until reviewed corpus references and approved media handling are available.
- Cold and warm cache results are recorded as separate policies and must not
  be merged into one comparison.

## Recommendation

Keep the provider decision **DEFERRED**. Do not change defaults, safety gates,
or production routing based on this implementation-only result.
