# Static Source Cover MVP Visual QA

## Scope
Visual-only QA for existing artifacts. No detection, OCR, render, fallback, watermark, or TTS logic was changed.

## Evidence
Each QA sheet shows input, static source cover, static subtitle burn, and the nearest region-detection overlay for four review timestamps.

| Sample | Verdict | Stable bbox | Rendered / skipped | Visual notes | Duplicate risk | False-cover risk | Next action |
|---|---|---|---:|---|---|---|---|
| sample2 | CONDITIONAL | `123.75, 966.0, 568.0, 1030.5` | 20 / 2 | Correct bottom lane. The source white bubble is wider than the black cover in reviewed frames, leaving visible bubble/text edges. | Medium | Low | Tune cover padding after explicit approval, then rerun QA. |
| sample4 | PASS | `268.0, 891.5, 455.0, 974.0` | 25 / 0 | Correct localized cover over green subtitle. Central watermark remains untouched. | Low | Low | Retain as MVP reference sample. |
| sample5 | CONDITIONAL | `277.0, 942.0, 442.0, 1040.0` | 20 / 5 | Correct rendered localization. Skipped intervals retain original source text by design, so cover is incomplete. | Medium | Low | Require manual acceptance of safe skips or add a reviewed policy later. |

## Final verdict

- **Static Source Cover MVP Visual QA: CONDITIONAL**
- **Static Source Cover MVP readiness: BLOCKED**
- **Dynamic subtitle replacement: OUT OF SCOPE**
- **Production Ready: NO**

The fallback burn output remains readable and ASS/SRT sidecars exist for every sample. The MVP is not ready to claim clean source replacement across the reviewed sample set because sample2 has visible source-bubble residuals and sample5 has deliberate uncovered intervals.