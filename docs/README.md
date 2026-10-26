# MediaFlow Documentation

**Documentation version:** V2 — Feature-first Modular Clean Architecture  
**Status:** CURRENT AUTHORITATIVE SPECIFICATION

This directory is the authoritative documentation set for `media-flow`. It intentionally replaces the previous collection of phase reports, architecture audits, benchmark reports, experimental subtitle policies, and duplicated `docs/core/` files.

## Authoritative documents

Read these in this order:

1. [`ARCHITECTURE.md`](ARCHITECTURE.md) — canonical system architecture and dependency rules.
2. [`PROJECT_OPERATING_SPECIFICATION.md`](PROJECT_OPERATING_SPECIFICATION.md) — engineering rules that all production code must follow.
3. [`FEATURE_CONTRACTS.md`](FEATURE_CONTRACTS.md) — contracts and responsibilities of Audio, Visual, Subtitle and shared actions.
4. [`PROVIDERS.md`](PROVIDERS.md) — replaceable external-provider boundaries.
5. [`GEMINI_RUNTIME.md`](GEMINI_RUNTIME.md) — shared Gemini infrastructure rules.
6. [`YOLO.md`](YOLO.md) — production YOLO detector contract and 4-class mapping.
7. [`PUBLISHING.md`](PUBLISHING.md) — reup/publish automation, platform OAuth setup, idempotent state.
8. [`E2E_VALIDATION.md`](E2E_VALIDATION.md) — acceptance and regression validation.
9. [`QUALITY_AND_OPERATIONS.md`](QUALITY_AND_OPERATIONS.md) — offline quality gate, CI, job lifecycle, and benchmark policy.
10. [`MIGRATION_PLAN.md`](MIGRATION_PLAN.md) — historical migration guidance only; current behavior is defined by the documents above.

## Canonical product shape

```text
Video Input
├── AudioPipeline
│   └── ASR → Translation → TTS → AudioComposer
└── VisualPipeline
    └── YOLO → class-aware tracking
        ├── subtitle → SubtitleProcessor
        │   └── event split → OCR → Translation → blur old text → draw translated text at source position
        ├── logo → configurable action, default blur
        ├── watermark → configurable action, default blur
        └── other_burn_in_text → configurable action, default blur

Shared capabilities
├── Translation
├── rendering actions
├── Gemini runtime
└── video I/O / muxing
```

Audio and Visual are independent features. Neither may require the other to run.

## Documentation rules

- These files describe the **current production architecture and contracts**, not historical experiments.
- Historical reports are intentionally not copied into this active docs set. Git history is the archive.
- Do not infer current production behavior from an old benchmark, phase name, output folder, or deleted report.
- When implementation changes intentionally, update the corresponding authoritative document in the same change.
- Do not add another architecture document unless it replaces `ARCHITECTURE.md` through an explicit decision.
- Temporary implementation notes belong in issues/commits, not as permanent `PHASE_X_REPORT.md` files.

## Current implementation

The runtime follows the V2 feature boundaries: `ProcessVideoUseCase` composes independent Audio and Visual pipelines, provider construction is lazy at the interface composition boundary, and local CLI/web jobs share `JobRunner` plus durable `JobStore` state. `MIGRATION_PLAN.md` is retained only as historical guidance and is not the source of truth for current behavior.
