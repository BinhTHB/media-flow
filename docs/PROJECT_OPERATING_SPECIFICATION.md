# Project Operating Specification V2

## 1. Purpose

MediaFlow processes short-form video using independent Audio and Visual capabilities while keeping external AI/media implementations replaceable.

Primary product capabilities:

```text
Audio:  ASR → Translation → TTS → composed dubbed audio
Visual: YOLO → class-specific visual processing
```

The system is designed as a modular monolith. It must remain simple enough to debug locally while preserving clean dependency boundaries for future API/frontend use.

## 2. Engineering principles

### 2.1 Feature-first

Business capabilities are organized by feature. Adding, removing, disabling, or upgrading Audio, Subtitle, Logo, Watermark, or OtherText processing should have minimal impact on unrelated features.

### 2.2 Clean dependencies

Features depend on contracts, not external SDKs. Gemini, Ultralytics, Faster-Whisper, TTS engines, OpenCV and FFmpeg belong at infrastructure boundaries.

### 2.3 Independent Audio and Visual execution

AudioPipeline and VisualPipeline must be runnable independently.

Valid product modes include:

- Visual only with original audio preserved.
- Audio only with original video pixels preserved.
- Audio + Visual together.
- Individual Visual subfeatures enabled/disabled independently.

### 2.4 Shared capability, single implementation

A capability used by multiple features must not be duplicated.

Examples:

- Audio and Subtitle use the same Translation contract/service.
- Logo, Watermark, OtherText and Subtitle background removal can share `BlurAction`.
- Gemini-based providers share one Gemini runtime/scheduler/cache within an application context.

### 2.5 No unnecessary framework layers

Do not introduce factories, registries, strategies, commands or manager classes without a concrete need.

Abstraction is justified primarily when:

- there are multiple implementations;
- the implementation is external/optional;
- tests need a stable seam; or
- the boundary is expected to change independently.

### 2.6 Fail explicitly

Do not silently switch providers, invent OCR/ASR text, fabricate translations, or hide failed rendering.

### 2.7 Source-of-truth ownership

Each piece of information has one owner:

- YOLO: class + visual bbox/confidence per detection.
- Tracker: temporal association and bbox trajectory.
- SubtitleEventSegmenter: text-event boundaries inside subtitle tracks.
- OCR: source text from supplied subtitle crops.
- Translation: translated text only.
- TTS: synthesized speech audio from supplied translated text.
- Actions: pixel manipulation/overlay.
- Renderer/muxer: output encoding and stream assembly.

## 3. Feature specification

### 3.1 Audio feature

Input: source video/audio.  
Output: optional dubbed audio stream plus speech/translation metadata.

```text
extract audio
→ ASR
→ SpeechSegment[]
→ Translation
→ translated SpeechSegment[]
→ TTS per segment/unit
→ AudioComposer
→ dubbed audio
```

AudioComposer is responsible for timeline assembly, silence, segment start/end, and duration reconciliation. TTS itself does not own video timing.

### 3.2 Visual feature

Input: source video.  
Output: rendered visual stream plus detection/track metadata.

```text
YOLO
→ class-aware tracking
→ route each class to its processor
→ execute shared visual actions
→ encode visual output
```

### 3.3 Subtitle feature

Default mode: `translate_replace`.

```text
YOLO subtitle geometry
→ track
→ split into subtitle text events
→ crop representative evidence
→ OCR
→ Translation
→ blur original subtitle region
→ draw translated text anchored to original region
```

The translated text may use adaptive font size, line wrapping, and bounded horizontal expansion. The source spatial anchor and temporal event must be preserved.

### 3.4 Logo feature

Default action: `blur`.

Allowed actions are limited to implemented actions, such as `blur`, `preserve`, `replace_image`, or `solid_cover`.

### 3.5 Watermark feature

Default action: `blur`.

It does not invoke OCR/Translation unless an explicit future feature requires it.

### 3.6 Other burn-in text feature

Default action: `blur`.

It remains independent from Subtitle. Do not automatically treat `other_burn_in_text` as subtitle text.

## 4. Data contracts

Stable contracts should be JSON serializable and use explicit units/names.

At minimum the system needs contracts equivalent to:

- `BBox`
- `Detection`
- `Track`
- `SubtitleEvent`
- `SpeechSegment`
- `TranslationUnit` or translated segment fields
- `ProcessingRequest`
- `ProcessingResult`

Do not leak Ultralytics result objects, PIL/OpenCV objects, Gemini raw responses, or TTS SDK objects into feature contracts.

## 5. Configuration

### 5.1 Precedence

```text
runtime / CLI / API request
→ project configuration
→ code defaults
```

### 5.2 Product configuration

Configuration should describe product features, not internal implementation graphs.

Recommended shape:

```yaml
audio:
  enabled: false
  asr:
    provider: faster-whisper
  translation:
    provider: gemini
    target_language: vi
  tts:
    provider: null

visual:
  enabled: true
  subtitle:
    enabled: true
    mode: translate_replace
    background: blur
    target_language: vi
  logo:
    enabled: true
    action: blur
  watermark:
    enabled: true
    action: blur
  other_burn_in_text:
    enabled: true
    action: blur
```

Provider-specific options may live under provider configuration, but business policy must have one source of truth.

## 6. Provider rules

- Provider implementations are replaceable adapters.
- Providers receive stable feature/core contracts or primitive data.
- Providers do not decide unrelated business policy.
- OCR does not translate.
- Translation does not inspect images/audio or create bbox values.
- ASR does not read visual subtitles.
- TTS does not decide timeline placement.
- Detector does not decide blur/replacement actions.
- Provider initialization should be lazy when the feature is optional.

See `PROVIDERS.md`.

## 7. Error handling

Use a small explicit error taxonomy.

Recommended categories:

- validation/configuration error;
- provider unavailable/authentication error;
- provider transient/rate-limit/timeout error;
- provider response/schema error;
- media decode/encode error;
- rendering/action error.

Rules:

- Retry only errors that are genuinely transient.
- Respect provider retry hints when available.
- Do not retry deterministic validation errors indefinitely.
- Preserve the original exception as the cause where practical.
- `ProcessingResult` must report failure/block reason without pretending an output operation succeeded.

## 8. Job and artifact management

Each job uses an isolated workspace under `outputs/jobs/<job_id>/`.

Only create artifacts required by the enabled features. Do not generate large debug/review packs during ordinary production execution unless explicitly requested.

Generated video/audio/log/cache artifacts must not be committed.

## 9. Testing requirements

### Unit tests

Test each feature and action independently using fakes/mocks for external providers.

Required independence assertions include:

- audio-only execution does not initialize YOLO;
- visual-only execution does not initialize ASR/TTS;
- blur-only visual execution does not require Gemini;
- disabling Subtitle prevents OCR/Translation for Subtitle;
- class routing never mixes subtitle/logo/watermark/other;
- shared actions are reusable across processors;
- translation is shared without business duplication.

### Contract tests

Each real provider adapter must satisfy the same port contract as its test fake.

### E2E tests

Use representative local videos and validate media invariants plus manual visual review. See `E2E_VALIDATION.md`.

## 10. Coding rules

Forbidden:

- production imports from benchmark/evaluation modules;
- generic conversion of all YOLO classes to `text`;
- hard bottom-band filtering as a universal subtitle rule;
- OCR/Translation provider deciding visual position;
- class processors duplicating pixel-operation implementations;
- Audio and Visual importing each other's concrete implementations;
- API/UI code containing media business logic;
- committing generated E2E videos/job artifacts.

Required:

- type hints for public contracts;
- deterministic class/action routing;
- bbox bounds validation;
- explicit timestamps/units;
- tests for new feature behavior;
- documentation update when public feature contracts change.

## 11. Extension rules

### Add a new provider

Implement the existing port and wire it at the composition root/config boundary. Do not modify feature business flow unless the provider introduces a genuinely new capability.

### Add a new visual action

Implement one shared action and expose it only to processors that need it. Do not add a generic workflow engine.

### Add a new visual class/feature

Add a processor and routing rule. Reuse detector/tracker/actions where possible.

### Add a new major feature

Create a feature module with its own pipeline and contracts. It may share common services through ports, but should not create direct dependencies on unrelated features.

## 12. When to introduce more complex architecture

Only consider a DAG/workflow engine, plugin framework, distributed worker system, or microservices when requirements actually include arbitrary user-defined workflows, third-party plugins, distributed execution, or independent scaling/deployment.

Until then, MediaFlow remains a modular monolith.
