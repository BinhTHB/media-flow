# Phase F, Gemini Default Cutover

**Status:** IMPLEMENTED
**Production Ready:** NO

## Defaults

| Capability | Before | After | Fallback |
|---|---|---|---|
| OCR | Mock/local configuration | `gemini_vision` | `paddle` |
| Speech | Mock/local configuration | `gemini_audio` | `faster-whisper` |
| Translation | Independent request | `gemini` when selected | None |

`GEMINI_API_KEY` is the canonical key. A selected Gemini OCR or ASR provider
fails fast with `ProviderConfigurationError` when it is unavailable. Explicit
mock providers do not require a key.

## Runtime and pipeline

`ProviderFactory` creates one shared `GeminiRuntime`; Gemini OCR, ASR and
translation receive that same scheduler, cache and client owner. OCR and ASR
return their Gemini segments directly to the standard use cases. They are not
shadow artifacts, are not merged with local provider output, and translation
remains a separate request.

## Fallback policy

The infrastructure fallback wrapper only uses local providers for Gemini quota,
network/transient-server, model-availability and structured-response failures.
Missing/corrupt local media and local validation errors do not fall back.

Job debug metadata records configured/effective provider, fallback state and
trigger, primary error type, retry/request/cache/chunk/escalation counters, and
model identity. It excludes API keys, authorization headers and media payloads.

## Overrides and verification

CLI overrides remain available:

```text
--ocr-provider gemini_vision|paddle|mock
--speech-provider gemini_audio|faster-whisper|mock
```

Offline tests cover defaults, shared runtime ownership, missing-key failure,
explicit local overrides, controlled fallback, no fallback for corrupt media,
and non-merged fallback output. Live smoke is not run by pytest and was not run
for this report.

Known limitations: full live qualification, reviewed corpus quality gates and
production SLA validation remain outstanding. `Production Ready: NO` does not
change the default cutover.

## Verdict

```text
Gemini OCR default:
ENABLED

Gemini ASR default:
ENABLED

PaddleOCR fallback:
AVAILABLE

Faster-Whisper fallback:
AVAILABLE
```
