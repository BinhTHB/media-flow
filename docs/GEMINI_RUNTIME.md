# Shared Gemini Runtime

Gemini may be used by more than one MediaFlow provider. The Gemini infrastructure is therefore shared, while OCR, ASR and Translation remain separate capabilities.

## 1. Core rule

Do not build one `GeminiVideoUnderstandingProvider` that performs OCR + ASR + translation + rendering decisions.

Keep independent adapters:

```text
Gemini Vision OCR adapter
Gemini Audio ASR adapter        # optional/configurable
Gemini Translation adapter
            │
            └── all share GeminiRuntime
```

The runtime handles API mechanics. Providers handle capability-specific prompts/contracts. Features handle business orchestration.

## 2. Reusable Gemini infrastructure

Preserve/reuse the existing shared concepts where the source still supports them:

```text
GeminiRuntime
├── client
├── request scheduler
├── retry/backoff
├── response cache
├── structured-output parser
├── model validation
├── file lifecycle management when uploads are used
└── usage/provenance recorder
```

Do not duplicate these concerns inside OCR, ASR and Translation adapters.

## 3. Request scheduler

Gemini requests can share a project/model quota. Scheduler state must therefore be shared within an application process rather than creating an unrelated limiter for every provider.

Track at least what is actually available from the API/runtime, such as:

- request count;
- request rate/window;
- token usage when returned;
- retry count;
- rate-limit responses;
- cache hits/misses;
- model ID;
- operation/provider role.

Client-side caps are safety controls, not claims about Google's current official quotas.

Do not hardcode documentation claims about current Free Tier limits. Operational limits must be checked in the active Google project/provider console at deployment time.

## 4. Retry policy

Retry only transient technical failures, for example:

- rate limiting;
- eligible 5xx responses;
- timeouts/network interruptions.

Use bounded exponential backoff and respect `Retry-After` when present.

Structured response/schema failure may receive a small bounded retry with stricter parsing/prompt behavior when explicitly designed. Do not retry deterministic bad input indefinitely.

## 5. Cache contract

Cache identity must include every input that can change the semantic result.

Recommended key components:

```text
input content hash / normalized input text
exact model ID
provider/operation role
prompt version
schema version
generation configuration
media resolution when relevant
API surface
```

Different models or prompt/schema versions must not share stale results.

Use atomic writes and a job/cache policy that prevents corruption.

## 6. Credentials

- Resolve `GEMINI_API_KEY` lazily from environment/secret storage.
- Never write secrets into `ProcessingRequest`, job JSON, logs, cache keys, or provenance artifacts.
- Never log authorization headers.

## 7. Model selection

Model IDs are configuration, not architecture.

Rules:

- do not duplicate default model IDs across multiple provider classes/config files;
- record the exact model used in provenance;
- changing model ID invalidates relevant cache keys;
- validate configured model availability/capability before expensive jobs when practical;
- do not silently switch models without recording the decision.

This V2 documentation deliberately avoids freezing model lifecycle dates or Free Tier quotas because they are operational data that can change.

## 8. Gemini OCR in V2

Old experiments asked Gemini to find/classify subtitle regions in full video frames. That is no longer the canonical V2 responsibility because YOLO supplies the class and bbox.

Canonical flow:

```text
YOLO subtitle event
→ select/crop representative source pixels
→ Gemini OCR
→ source_text
```

Gemini OCR may use multiple crops/temporal consensus to resolve text changes or difficult frames, but it must not become a second source of truth for render geometry.

## 9. Gemini Translation in V2

Canonical flow:

```text
validated source text
→ Gemini Translation adapter
→ translated text
```

Translation never receives raw video/audio for the purpose of deciding bbox/class/rendering.

A translation quality check may validate obvious output failures (for example empty result or source-script leakage) without taking ownership of visual policy.

## 10. Gemini ASR in V2

If Gemini Audio is selected as the ASR provider:

```text
AudioPipeline
→ extract audio
→ Gemini Audio ASR adapter
→ SpeechSegment[]
```

It shares the Gemini runtime but remains interchangeable with non-Gemini ASR implementations.

## 11. Provenance

Record enough metadata to reproduce/debug provider behavior without leaking secrets:

```text
provider
operation
model ID
prompt/schema version
cache hit
retry count
usage metadata when available
timestamp/job ID
```

Do not store raw credentials.
