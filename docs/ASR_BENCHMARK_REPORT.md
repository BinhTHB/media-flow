# ASR Benchmark Report

## Methodology

The benchmark script ran Faster-Whisper `tiny`, `base`, and `small` against
`sample.mp4` (264.5 seconds, Chinese dialogue) on CPU. It extracts a 16 kHz
mono WAV once, then writes one transcript, metrics file, and runtime file per
model under `benchmarks/20260711-170551/`.

The benchmark records runtime, segment timing, character and word counts,
empty and duplicate segments, average confidence, optional CER/WER, and known
suspicious substitutions. No reference transcript exists for `sample.mp4`, so
CER/WER were correctly skipped.

The targeted substitution is `因果` (karma/cause-and-effect) being recognized
as `英国` (England). This is a repeatable quality signal for this clip, not a
semantic correction.

## Results

| Model | STT runtime | Segments | Avg duration | `英国` occurrences | CER / WER |
| --- | ---: | ---: | ---: | ---: | --- |
| `tiny` | 18.528s | 160 | 1.576s | 9 | n/a |
| `base` | 30.698s | 171 | 1.175s | 9 | n/a |
| `small` | 147.911s | 68 | 3.769s | 0 | n/a |

Audio extraction took 0.329s and was shared between model runs.

## Strengths

- `tiny` provides fast smoke-test feedback at roughly 8x the input duration.
- `base` remains practical for lightweight local development.
- `small` removed all nine instances of the measured `因果 → 英国`
  substitution and produced fewer, longer segments suitable for subtitle
  normalization.
- No model emitted empty segments. `small` emitted no duplicate segments.

## Weaknesses

- Faster-Whisper does not populate usable confidence values in the current
  provider output, so the benchmark reports confidence as `n/a`.
- No human reference transcript is available, so CER/WER cannot quantify
  absolute transcript accuracy.
- `small` takes 147.911s on CPU for a 264.5s video. It is substantially
  slower than `tiny` or `base`.
- The benchmark reports known substitutions only; it intentionally does not
  correct or infer transcript meaning.

## Recommended Defaults

| Purpose | Recommendation | Reason |
| --- | --- | --- |
| Smoke tests / CI | `tiny` | Lowest runtime; verifies integration only. |
| Development | `base` | Faster iteration while preserving a representative model. |
| Production Chinese short videos | `small` | Best observed quality; removes the critical homophone error. |
| Production English short videos | `base` | A reasonable speed-quality baseline; benchmark English separately before finalizing. |

These are benchmark recommendations only. Production configuration remains
unchanged.

## Known Limitations

- Results are from one Chinese video and do not establish general accuracy.
- The current model has no reference transcript at
  `reference/sample.zh.txt`.
- Translation impact is inferred from the real-output review: incorrect ASR
  input such as `英国` was translated literally, producing incorrect
  Vietnamese subtitles. This benchmark does not alter `TranslationProvider`.

## Next Milestone

Add a reviewed Chinese reference transcript for `sample.mp4`, rerun the same
benchmark to collect CER, and benchmark `small` on representative hardware
(GPU if production will use it). Confirm the quality gain before changing any
production default.
