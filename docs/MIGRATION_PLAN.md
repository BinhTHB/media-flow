# V2 Architecture Migration Plan

**Status:** TEMPORARY — use during the feature-first refactor, then remove or archive after completion.

## 1. Objective

Migrate the current policy-driven/generic processing code to the V2 Feature-first Modular Clean Architecture without throwing away tested working components.

Target product shape:

```text
AudioPipeline independent
VisualPipeline independent

VisualPipeline:
YOLO
├── SubtitleProcessor
├── LogoProcessor
├── WatermarkProcessor
└── OtherTextProcessor

Shared actions:
Blur / Preserve / ReplaceImage / TextOverlay / optional SolidCover

Shared Translation service
Shared Gemini runtime
```

## 2. Migration principle

Do not rewrite from zero.

For every current module classify it as:

```text
KEEP
MOVE
SPLIT
RENAME
REPLACE
DELETE/LEGACY
UNKNOWN
```

Never delete `UNKNOWN` until dependency/source usage is proven.

## 3. Reuse first

The following concepts are expected to be reusable when source confirms them:

- canonical 4-class YOLO detection;
- class-aware tracking;
- bbox validation/clamping;
- current working Gaussian blur/image/text rendering code;
- FFmpeg encode/mux utilities;
- job-scoped workspace/artifacts;
- ProcessingRequest/ProcessingResult concepts;
- lazy provider loading;
- Gemini client/scheduler/cache/structured-output/usage infrastructure;
- existing ASR/OCR/Translation provider adapters that still satisfy V2 contracts.

## 4. Replace/simplify candidates

Audit and simplify generic orchestration such as:

- large `ActionPolicyEngine` used as the center of all features;
- generic `LazyEnrichmentService` hiding feature ownership;
- large renderer containing every action implementation in one conditional chain;
- generic media processing service containing class-specific business rules;
- duplicate detection/business config sources;
- old OCR/ASR merge flow when used as a mandatory product path;
- old subtitle above/below/global-placement policy as the main replacement behavior;
- legacy Gemini track verifier / bottom-band gates / subtitle-only routing;
- benchmark/evaluation namespaces imported by production.

An intermediate render/action plan may remain as an artifact if useful, but it should not dictate the architecture.

## 5. Phase 0 — establish baseline

Before moving/removing production code:

- run the current test suite;
- record exact test count/result;
- run known E2E sample(s);
- record output duration/resolution/FPS/audio;
- preserve manual-review outputs outside Git.

Do not clean generated artifacts until baseline evidence is recorded.

## 6. Phase 1 — define stable core contracts

Introduce/reuse small contracts for:

```text
Detection
Track
SubtitleEvent
SpeechSegment
ProcessingRequest
ProcessingResult
```

Avoid external SDK objects in these contracts.

Keep serialization where existing frontend/job work already depends on it.

## 7. Phase 2 — extract shared providers/infrastructure

Preserve provider boundaries for:

```text
Detector
ASR
OCR
Translation
TTS
Video
```

Preserve one shared Gemini runtime for all Gemini adapters.

Move implementation files only after tests/imports prove the new path.

## 8. Phase 3 — build VisualPipeline

Create a clear VisualPipeline that owns:

```text
YOLO
→ tracking
→ class routing
→ processors
→ actions
→ visual rendering
```

Class routing is fixed to the YOLO contract:

```text
0 subtitle
1 watermark
2 logo
3 other_burn_in_text
```

Default product behavior:

```text
subtitle            → translate_replace
logo                → blur
watermark           → blur
other_burn_in_text  → blur
```

## 9. Phase 4 — split shared actions from renderer

Move pixel operations out of one large renderer when they are currently embedded there.

Target:

```text
BlurAction
PreserveAction
ReplaceImageAction
TextOverlayAction
SolidCoverAction (if retained)
```

Renderer owns frame iteration/encoding/order, not the algorithms for every action.

A simple action mapping is sufficient. Do not create a generic plugin framework.

## 10. Phase 5 — rebuild SubtitleProcessor around YOLO geometry

Canonical subtitle flow:

```text
YOLO subtitle track
→ SubtitleEventSegmenter
→ SubtitleEvent
→ crop representative evidence
→ OCR
→ Translation
→ blur old subtitle
→ draw translated text at source position
```

Important migration changes:

- do not ask Gemini to rediscover the primary bbox;
- do not force bottom-band placement;
- do not merge different text events just because they share geometry;
- preserve original source anchor/bbox trajectory;
- allow adaptive text fit/width/wrapping for Vietnamese.

Add the same-position/different-text regression fixture before declaring this phase complete.

## 11. Phase 6 — build AudioPipeline

Audit existing ASR/TTS code first.

Target:

```text
ASR
→ shared Translation
→ TTS
→ AudioComposer
```

If real production TTS does not exist yet:

- implement/retain the port;
- make Audio feature report incomplete/blocked for TTS mode;
- do not fake an E2E success.

AudioPipeline must run without importing/initializing YOLO.

## 12. Phase 7 — thin application composition

`process_video(request)` should only:

```text
validate request
prepare workspace
run VisualPipeline if enabled
run AudioPipeline if enabled
mux selected streams
validate output
return result
```

Remove class-specific/prompt/pixel logic from the application entrypoint.

## 13. Phase 8 — remove superseded production paths

Only after V2 tests and E2E pass:

- delete or isolate old generic production orchestration no longer referenced;
- remove duplicate config/business-policy sources;
- remove legacy production adapters that only re-export obsolete pipelines when no compatibility consumer remains;
- ensure production code does not import benchmark/evaluation modules;
- update README/CLI/config examples to V2.

Do not use `git clean -fd` or `git reset --hard` as a cleanup strategy.

## 14. Tests required before cleanup commit

At minimum:

- AudioPipeline independent from VisualPipeline;
- VisualPipeline independent from AudioPipeline;
- audio-only does not initialize YOLO;
- visual-only does not initialize ASR/TTS;
- blur-only works without Gemini;
- subtitle routes only to SubtitleProcessor;
- each non-subtitle class routes to its correct processor;
- shared BlurAction used by multiple processors;
- SubtitleEvent segmentation handles text changes at same bbox;
- OCR/Translation preserve source event timing/geometry identity;
- subtitle renderer anchors translated text to source position;
- disabled features do not initialize providers;
- production imports no benchmark/evaluation code;
- full regression suite green.

## 15. E2E after migration

Run the scenarios in `E2E_VALIDATION.md`.

At minimum produce manual-review outputs for:

```text
visual blur-only
visual subtitle translate/replace
audio-only if real TTS is production-capable
full Audio + Visual after both independent paths are green
```

## 16. Completion criteria

Migration is complete when:

- source tree clearly exposes Audio and Visual features;
- Audio and Visual are independently executable;
- YOLO remains the source of visual class/bbox geometry;
- SubtitleEvent segmentation prevents same-position text merging;
- shared Translation and shared actions are not duplicated;
- external providers remain replaceable;
- obsolete generic/legacy production path is removed;
- active docs match implemented code;
- E2E and automated tests pass.

After this point, delete/archive this migration guide and update `README.md` status from target/migration to implemented architecture.
