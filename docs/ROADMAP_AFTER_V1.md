# Roadmap After Framework v1.0

## Scope

This roadmap covers only the product-validation work required after the MediaFlow Framework v1.0 sign-off. It does not add framework features or change the v1.0 architecture.

## Milestone 1: Production ASR Model Qualification

**Goal:** Select and approve an ASR model configuration for each supported source-language and hardware profile.

**Required evidence:**

- Representative Chinese and English media sets.
- Reviewed reference transcripts.
- Character/word error measurements for the selected model and hardware.
- Measured CPU/GPU runtime, memory use, and model-loading behavior.
- Explicit acceptance criteria for subtitle linguistic accuracy.

**Reason:** Current evidence shows `tiny` and `base` can make Chinese homophone errors. The same benchmark showed `small` removed the measured `因果` → `英国` substitution but is slower on CPU.

## Milestone 2: Real Provider Deployment Qualification

**Goal:** Validate the selected provider set in the intended deployment runtime.

**Required evidence:**

- FFmpeg availability with required codecs, ASS support, and production fonts.
- PaddleOCR installation, model initialization, and CPU/GPU resource measurements.
- Faster-Whisper device and compute-type validation.
- Translation-provider credentials, quota behavior, rate-limit behavior, and network-failure behavior.
- A real end-to-end run using representative media and the production provider configuration.

**Reason:** Framework tests use mocks or controlled provider behavior for most external dependencies. Real-provider performance and operating limits are deployment-specific.

## Milestone 3: Output Quality Acceptance

**Goal:** Establish product release acceptance for generated subtitles.

**Required evidence:**

- Human review of source transcription and Vietnamese translation quality.
- Detection or review workflow for untranslated source-language text in translated subtitles.
- Timing, readability, and subtitle-placement review on representative videos.
- Confirmation that the selected model/provider configuration meets the approved quality threshold.

**Reason:** The framework validates processing and rendering behavior, not universal linguistic correctness.

## Milestone 4: Capacity and Operational Validation

**Goal:** Establish safe production operating limits for the approved deployment.

**Required evidence:**

- CPU/GPU capacity measurements for concurrent workloads.
- FFmpeg render throughput under the target video resolutions and durations.
- Disk-space requirements for extracted audio, frames, subtitle artifacts, and output video.
- Recovery procedures for provider/network/process failures.
- Monitoring and logging requirements for operational jobs.

**Reason:** PaddleOCR, ASR, and FFmpeg have hardware- and media-dependent runtime costs.

## Completion Condition

Product release can be declared ready only when all four milestones have evidence for the intended languages, models, providers, media profile, and deployment hardware.
