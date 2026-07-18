# Architecture Review Report

**Project**: AI Video Translator
**Date**: 2026-07-10
**Scope**: Full architecture review after STT, OCR, Translation, Subtitle Quality, Segmentation, Styling, Rendering milestones.

---

## 1. Current Strengths

### Clean Architecture Enforcement
- **Domain layer** (`src/domain/`) contains pure entities and validation with zero external dependencies.
- **Application layer** (`src/application/`) defines all ports (interfaces) and use cases. No infrastructure imports.
- **Infrastructure layer** (`src/infrastructure/`) implements ports with concrete providers (Mock, FFmpeg, Faster-Whisper, OpenAI, PaddleOCR).
- **Interfaces layer** (`src/interfaces/`) is the only composition root. CLI demo wires providers via `ProviderFactory`.

### Provider Pattern & Dependency Inversion
- All external capabilities (Speech, OCR, Translation, Video, Subtitle, Storage, Frame Extraction) implement Application ports.
- `ProviderRegistry` maps category→name→class; `ProviderFactory` merges YAML config + CLI overrides.
- Adding a new provider requires: implement port, register in registry, add config entry. No Application/Domain changes.

### Stable Data Contracts
- Domain entities (`VideoJob`, `Segment`, `Frame`, `SubtitleItem`, `SubtitleStyle`, `SubtitleQualityConfig`, `FeatureConfig`, `JobStatus`) use `@dataclass` with `__post_init__` validation.
- All entities expose `to_dict()` for JSON serialization.
- Backward compatibility maintained (no field removal, only additions).

### Feature Composition via JobOrchestrator
- `FeatureConfig` controls 5 independent flags: `speech_to_text`, `ocr`, `translate`, `subtitle`, `render_video`.
- Orchestrator conditionally executes stages, writes artifacts, transitions `JobStateMachine`.
- Skip/Fail policies are explicit and tested.

### Observability via JobStateMachine
- `JobStatus` enum covers: `CREATED`, `RUNNING`, `EXTRACTING_AUDIO`, `TRANSCRIBING`, `DETECTING_TEXT`, `TRANSLATING`, `GENERATING_SUBTITLE`, `RENDERING_VIDEO`, `COMPLETED`, `FAILED`, `SKIPPED`.
- State transitions update timestamps (`started_at`, `completed_at`).

### Artifact Management
- `ArtifactProvider` port defines all output paths; `ArtifactManager` implements filesystem I/O with overwrite guard.
- Layout: `output/<job_id>/audio/`, `transcripts/`, `subtitles/`, `videos/`, `frames/`, `logs/`.

### Testing Discipline
- 66 unit tests passing.
- Mock providers used by default; real providers opt-in via CLI.
- Architecture boundary tests enforce no Infrastructure imports in Domain/Application.

---

## 2. Technical Debt

### 2.1 Duplicated Logic
| Location | Issue |
|---|---|
| `NormalizeSegmentsUseCase.execute()` lines 50-80 | Inline duplicate-finding loops for "prefer speech over OCR" and "deduplicate OCR/speech text" repeat `TextSimilarityService.get_similarity()` calls with slightly different thresholds. Merge logic also inline. |
| `NormalizeSegmentsUseCase` vs `SubtitleSegmentationUseCase` | Both do duration clamping and line-wrapping. Segmentation re-clamps duration already handled in normalization. |
| `JobOrchestrator.execute()` artifact writing | Five identical `artifact_provider.write_json([s.to_dict() for s in segments], path)` patterns. Could use a small helper. |
| `JobOrchestrator` subtitle block | Calls `generate_subtitle_uc.execute()` twice (SRT then ASS) with nearly identical code. |

### 2.2 Duplicated Abstractions / Overlapping Concerns
| Area | Detail |
|---|---|
| `SubtitleQualityConfig` (Application) vs `SubtitleStyle` (Domain) | Both define `alignment`, `margin_v`, `margin_l`, `margin_r`, `max_chars_per_line`, `max_lines_per_subtitle`, `min_duration`, `max_duration`. Quality config used in Normalize+Segmentation; Style used in ASS generation. Some fields duplicated, others similar but named differently. |
| `providers.yaml` → `subtitle_quality` + `subtitle_styles.yaml` | Two separate YAML files configure subtitle formatting. Quality config drives Normalize/Segmentation; Style config drives ASS rendering. Overlap in `max_chars_per_line`, `max_lines_per_subtitle`, `min_duration`, `max_duration`. |
| `JobStateMachine` vs `VideoJob.status` | StateMachine only transitions; does not enforce valid transition graph (e.g., can go `FAILED` → `RUNNING`). Business logic enforces ordering implicitly in Orchestrator. |

### 2.3 Inconsistent Naming
| Inconsistency | Example |
|---|---|
| Artifact paths | `video_path` AND `video_output_path` both map to `output/<job_id>/videos/output.mp4` in `ArtifactProvider` and `ArtifactManager`. |
| ProviderFactory methods | `create_video_provider`, `create_speech_provider`, ... each has hardcoded `if provider_name == "ffmpeg":` branches. Naming pattern is consistent but implementation is not open/closed. |
| `SubtitleStylePreset.get()` parameter | Called `overrides` but merge is shallow and does not validate keys against `SubtitleStyle` fields. |

### 2.4 Unnecessary Complexity
| Location | Why |
|---|---|
| `JobOrchestrator.__init__` | 16 parameters (14 required, 2 optional). High arity makes instantiation error-prone. Could group into a single `OrchestratorConfig` dataclass. |
| `JobOrchestrator.execute()` | ~200 lines single method with nested conditionals. Readability suffers; testing individual stages requires full pipeline run. |
| `ProviderFactory` per-category methods | Adding a new video provider requires editing `create_video_provider()` (new `if` branch). Violates Open/Closed. Could use a `provider_initializers` registry mapping name → factory lambda. |
| `subtitle_styles.yaml` duplicates quality config fields | `max_chars_per_line`, `max_lines_per_subtitle`, `min_duration`, `max_duration` appear in both `subtitle_quality` (providers.yaml) and `subtitle_styles.yaml`. Source of truth unclear. |
| `Frame` entity | No `__post_init__` validation (negative timestamp, empty path). |

### 2.5 Missing Validation
- `SubtitleStylePreset.get(overrides={...})` accepts any dict keys; invalid keys silently ignored → typo-prone.
- `Frame` entity allows `timestamp < 0` and empty `image_path`.
- `JobStateMachine` allows arbitrary transitions (no guard for `FAILED` → `RUNNING`).

### 2.6 Overly Coupled Classes
- `JobOrchestrator` directly imports 10 use cases, `ArtifactProvider`, `JobStateMachine`, `FeatureLogger`, `ValidationError`, `SubtitleStyle`. High coupling to concrete use-case classes rather than a pipeline abstraction.

---

## 3. Improvement Opportunities

### High Priority (Architectural Impact)
1. **Group JobOrchestrator dependencies into a config object**
   - Create `OrchestratorConfig` dataclass holding all 16 parameters.
   - Single `JobOrchestrator(config: OrchestratorConfig, ...)` constructor.
   - Reduces constructor arity, enables defaults in one place.

2. **Extract stage methods from `JobOrchestrator.execute()`**
   - Private methods: `_run_audio_extraction()`, `_run_stt()`, `_run_ocr()`, `_run_merge()`, `_run_translate()`, `_run_normalize()`, `_run_segmentation()`, `_run_subtitle()`, `_run_render()`.
   - Keeps behavior identical; improves readability and enables unit-testing individual stages.

3. **Make ProviderFactory open/closed**
   - Replace per-category `if provider_name == "ffmpeg":` with a registry of initializer callables: `initializers: dict[str, Callable[[dict], Provider]]`.
   - New providers register their initializer at module import time.

4. **Consolidate subtitle quality/style configuration**
   - Single source of truth for `max_chars_per_line`, `max_lines_per_subtitle`, `min_duration`, `max_duration`.
   - `SubtitleQualityConfig` could embed a `SubtitleStyle` (or reference a preset name) so both Normalize/Segmentation and ASS generation read the same values.

### Medium Priority (Maintainability)
5. **Add `Frame.__post_init__` validation** (non-negative timestamp, non-empty image_path).
6. **Validate `SubtitleStylePreset.get(overrides)` keys** against `SubtitleStyle.__dataclass_fields__`.
7. **Remove duplicate artifact path** `video_output_path` from `ArtifactProvider`/`ArtifactManager`; keep only `video_path`.
8. **Extract artifact writing helper** in `JobOrchestrator` to eliminate repeated `write_json([s.to_dict()...])` boilerplate.
9. **Add transition guards to `JobStateMachine`** (optional) or document that Orchestrator enforces ordering.
10. **Fix CLI demo composition** - extract `build_orchestrator(args, config)` function for testability.

### Low Priority (Polish)
11. **Standardize `video_path` vs `video_output_path`** naming in `ArtifactProvider` (keep `video_path` only).
12. **Add docstrings** to all public UseCase `execute()` methods describing parameters and return values.
13. **Consider frozen dataclasses** for Domain entities where immutability is desired (e.g., `VideoJob.job_id`, `input_video`).

---

## 4. End-to-End Pipeline Verification

```text
Video
  ↓
Extract Audio (VideoProvider.extract_audio) → output/<job_id>/audio/audio.wav
  ↓
STT (SpeechProvider.transcribe) → List[Segment source="speech"]
  ↓
OCR (optional) (FrameExtractionProvider + OCRProvider) → List[Segment source="ocr"]
  ↓
Merge (MergeSegmentsUseCase) → timestamp-sorted combined list
  ↓
Translate (optional) (TranslationProvider) → translated_text populated
  ↓
Normalize (NormalizeSegmentsUseCase) → trim, dedup, prefer speech, clamp duration, wrap lines
  ↓
Segment (SubtitleSegmentationUseCase) → split long segments, proportional timing
  ↓
Subtitle (GenerateSubtitleUseCase + SubtitleProvider)
  ├─ SRT → subtitle.srt
  └─ ASS (if format=ass) → subtitle.ass
  ↓
Render (RenderVideoUseCase + VideoProvider.render_video)
  ├─ prefers .ass if exists (auto-detect)
  └─ falls back to .srt
  ↓
output/<job_id>/videos/output.mp4
```

**No duplicated processing steps. No missing stages.**

---

## 5. Summary

The architecture is solid: Clean Architecture boundaries are respected, Provider Pattern enables plug-and-play infrastructure, and contracts are stable. The identified technical debt is mostly **internal code organization** (large methods, high-arity constructors, duplicated helper logic, config duplication) rather than structural violations. All improvements can be made incrementally without changing public behavior.

Recommended next milestone: **Real Video Smoke Test & Validation**

Before adding REST API, TTS, dubbing, queue, or deployment, the system must be validated end-to-end with a real short video file (<5 minutes). All current tests use mock/monkeypatched providers. The validation must test:

1. FFmpeg real audio extraction.
2. FasterWhisper real transcription.
3. Translation provider behavior with real text.
4. Subtitle generation with Vietnamese Unicode.
5. ASS/SRT artifact creation.
5. FFmpeg real subtitle burn-in.
6. Windows path escaping.
7. Output video playback sanity.
8. Logs and error messages.
9. Whether the produced subtitle timing is acceptable.

Do not treat mock test pass as production readiness.


---

## 6. Resolved Technical Debt: Composition Root

- **High-Arity Constructor**: Simplified `JobOrchestrator.__init__` signature from 16 parameters down to a single `OrchestratorDependencies` object.
- **Duplicated Wiring Logic**: Removed manual instantiations of use cases, providers, factories, and registries inside CLI `demo.py`. CLI now only delegates to `ApplicationBuilder.from_cli_args(args).build()`.
- **Extensible Entry Points**: `MediaFlowApplication` acts as the unified public entry point, making it trivial to add REST APIs, Web/Desktop UIs, and Workers without duplicating composition logic.


---

## 7. Resolved Technical Debt: Provider Registry & Configuration Consolidation

- **ProviderFactory Open/Closed Violation**: Eliminated all hard-coded creation conditionals (`if name == "ffmpeg": ...`) from `ProviderFactory`. Creation is now fully dynamic via `registry.create(category, name, options)`.
- **Decoupled Configuration Parsing**: `ApplicationBuilder` no longer loads raw YAML. Configuration loading and merging of overrides is delegated to `ConfigLoader` returning a typed `AppConfig` instance.
- **Provider Registration**: Providers register factory callbacks on `ProviderRegistry` at boot time, enabling easy extensibility without modifying core framework code.
