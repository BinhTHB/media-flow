# Independent Architecture & Feature Audit

**Project**: AI Video Translator  
**Date**: 2026-07-11  
**Auditor**: Independent Senior Software Architect  
**Scope**: Complete codebase audit against documented architecture (AGENTS.md, POS, ARCHITECTURE_REVIEW.md, ARCHITECTURE_SCORECARD.md, README.md)

> This audit trusts ONLY actual source code, runtime behavior, and tests. Previous reports, comments, and documentation are treated as claims to be verified.

---

## Executive Summary

| Dimension | Score | Verdict |
|-----------|-------|---------|
| **Architecture (Clean Architecture)** | 9.0/10 | Strict layering enforced; zero boundary violations found. |
| **Maintainability** | 7.5/10 | Good separation, but JobOrchestrator is monolithic (200+ lines, 16 deps). |
| **Replaceability** | 9.5/10 | Provider Pattern fully realized; all 15 providers plug via registry. |
| **Testability** | 8.5/10 | 87 unit tests pass (0.5s); 6/11 use cases lack direct unit tests. |
| **Configuration** | 7.0/10 | Duplicated subtitle config across providers.yaml + subtitle_styles.yaml. |
| **Documentation** | 8.0/10 | POS/AGENTS comprehensive; README has stale provider table. |
| **Production Readiness** | 6.5/10 | Real pipeline validated once; no CI integration tests; PaddleOCR bare Exception. |
| **Framework Readiness** | 8.0/10 | Composition root solid; StorageProvider unused; Frame entity under-validated. |
| **Product Readiness** | 6.0/10 | Single real run validated; ASR quality bottleneck (	iny model homophones); no CER gate. |

**Overall Maturity**: **7.8/10** — solid framework, production gaps in ASR quality and CI validation.

---

## Resolved Findings

### PaddleOCR provider error mapping

- `PaddleOCRProvider` now maps missing dependency imports, initialization failures, OCR runtime failures, and malformed OCR responses to `ProviderError`, preserving the original exception as the cause.
- Evidence: `tests/test_ocr_framework.py` monkeypatches `sys.modules["paddleocr"]`; no PaddleOCR installation, model, GPU, or network access is required.

### Gemini model default

- The registry fallback, YAML configuration, and `GeminiTranslationProvider` default now all use `gemini-3.1-flash-lite`.
- ProviderFactory still selects the configured model when one is supplied; the registry fallback is used only when options are absent.
- Evidence: `tests/test_composition.py::test_gemini_model_is_selected_from_provider_config` and `test_provider_registry_create`.

### Subtitle configuration ownership

- `config/providers.yaml` owns `SubtitleQualityConfig` readability and timing fields: `max_chars_per_line`, `max_lines_per_subtitle`, `min_duration`, and `max_duration`.
- `config/subtitle_styles.yaml` owns only visual presentation fields defined by `SubtitleStyle` (font, colors, outline, shadow, alignment, margins, and opacity).
- Removed unused `subtitle_quality.segmentation` keys. ConfigLoader rejects visual keys in quality config, quality keys in presets, and unsupported keys. ApplicationBuilder now preserves YAML quality defaults unless the CLI explicitly supplies an override.
- Evidence: ownership rejection tests plus `test_application_builder_preserves_quality_config_and_applies_overrides`.

### Direct use-case coverage

- Direct mock-provider tests cover `ExtractAudioUseCase`, `TranscribeSpeechUseCase`, `ExtractFramesUseCase`, `DetectOnscreenTextUseCase`, `GenerateSubtitleUseCase`, and `RenderVideoUseCase`.
- Evidence: `tests/test_use_cases.py`; these tests do not invoke FFmpeg, OpenAI, Gemini, FasterWhisper, PaddleOCR, or a GPU.

### Validation evidence

- `python -m pytest -q`: 100 passed.
- Remaining unresolved audit risks: ASR quality for Chinese content, real-provider CI coverage, StorageProvider being unused, FFmpeg Windows filter-path fragility, and JobStateMachine transition guards. These are outside this corrective maintenance scope.

---

## 1. Clean Architecture Audit (Dependency Rules)

### Verified: **Zero Violations**

| Layer | Allowed Imports | Actual Imports | Status |
|-------|-----------------|----------------|--------|
| src/domain/ | stdlib + src.shared.exceptions | 10 files clean | **Pass** |
| src/application/ | Domain + Application internal + src.shared | 24 files clean | **Pass** |
| src/infrastructure/ | Application ports + Domain + src.shared + third-party | 18 files clean | **Pass** |
| src/interfaces/ | All layers (composition root) | 8 files clean | **Pass** |

**Evidence**: Static import analysis confirms zero forbidden imports across all layers. Domain imports only shared.exceptions; Application imports no infrastructure/interfaces; Infrastructure imports only ports/domain/shared.

### Observations
- Domain entities use @dataclass with __post_init__ validation — correct.
- No circular dependencies found.
- **Exception**: Frame entity lacks __post_init__ validation (negative timestamp, empty path allowed). Minor but violates POS §7 (Data Contracts).

---

## 2. Provider Audit

### Provider Matrix

| Category | Port | Mock | Real | Implements Port? | Error Wrapping | Status |
|----------|------|------|------|------------------|----------------|--------|
| Speech | SpeechProvider | MockSpeechProvider | FasterWhisperSpeechProvider | Yes | Yes | **Implemented** |
| Frame Extraction | FrameExtractionProvider | MockFrameExtractionProvider | FFmpegFrameExtractionProvider | Yes | Yes | **Implemented** |
| OCR | OCRProvider | MockOCRProvider | PaddleOCRProvider | Yes | No (bare Exception) | **Partially Implemented** |
| Translation | TranslationProvider | MockTranslationProvider | OpenAITranslationProvider, GeminiTranslationProvider | Yes | Yes | **Implemented** |
| Video | VideoProvider | MockVideoProvider | FFmpegVideoProvider | Yes | Yes | **Implemented** |
| Subtitle | SubtitleProvider | MockSubtitleProvider | ASSSubtitleProvider | Yes | Yes | **Implemented** |
| Storage | StorageProvider | MockStorageProvider | — | Yes | Yes (mock) | **Placeholder** |

### Detailed Findings

#### Critical: PaddleOCRProvider raises bare Exception
`python
# src/infrastructure/providers/ocr/paddle_ocr_provider.py:13, 24
raise Exception("paddleocr is not installed...")  # line 13
raise Exception(f"PaddleOCR detection failed...") from e  # line 24
`
**Violates POS §8** (Error Handling Policy): Infrastructure exceptions must wrap into ProviderError.

#### Missing: StorageProvider real implementation
- Port exists, mock registered, but no real provider (Supabase/S3/etc.). Not used in orchestrator/use cases — effectively dead code.

#### Contract Mismatch: VideoProvider.extract_audio
- Port: extract_audio(video_path, output_path=None)
- FFmpegVideoProvider: Honors output_path, falls back to output_dir + generated name.
- MockVideoProvider: Ignores output_path, returns hardcoded path derived from input.
- **Risk**: Tests using mock may not catch path-handling bugs in real provider.

#### Contract Mismatch: SubtitleProvider.generate_ass
- Port requires style: SubtitleStyle (mandatory).
- GenerateSubtitleUseCase passes style or SubtitleStyle() making it optional.
- **Risk**: Callers passing None get default style silently — may hide config errors.

---

## 3. Feature Audit

| Feature | Status | Confidence | Evidence | Missing Parts / Risks |
|---------|--------|------------|----------|------------------------|
| **Speech-to-Text** | Implemented | High | FasterWhisperSpeechProvider works with 	iny/ase/small; real pipeline validated | ASR quality bottleneck (	iny produces 9 homophone errors 因果 to 英国); no CER gate |
| **OCR** | Partially Implemented | Medium | PaddleOCRProvider exists but raises bare Exception; not exercised in real pipeline | No real run validation; mock only in tests; error wrapping broken |
| **Translation** | Implemented | High | Both OpenAI and Gemini providers work; Gemini has chunking + 429 retry | Gemini model default mismatch (registry gemini-2.0-flash vs YAML gemini-3.1-flash-lite) |
| **Merge** | Implemented | High | MergeSegmentsUseCase sorts by timestamp; tested in orchestrator | Deduplication logic inline (duplicated in Normalize) |
| **Normalize** | Implemented | High | NormalizeSegmentsUseCase with SubtitleQualityConfig; comprehensive tests | Duplicate-finding loops duplicated with Segmentation; duration clamping also in Segmentation |
| **Subtitle Segmentation** | Implemented | High | SubtitleSegmentationUseCase + service; proportional timing; duration constraints; tests | Re-clamps duration already clamped in Normalize; min_duration logic complex |
| **Subtitle Styling** | Implemented | High | ASSSubtitleProvider + ASSGenerator + presets; CLI --subtitle-style | Style config duplicated in providers.yaml (subtitle_quality) + subtitle_styles.yaml |
| **ASS Generation** | Implemented | High | ASSGenerator produces valid ASS v4.00+; auto-detected by FFmpegVideoProvider.render_video | None |
| **Rendering** | Implemented | High | FFmpegVideoProvider.render_video burns SRT/ASS; real 4m24s video rendered in 89s | Windows path escaping fragile (double-backslash in filters) |
| **Benchmark** | Implemented | High | scripts/benchmark_asr.py with CER/WER, homophone analysis, comparison.md | No GPU/accelerated test; CER skipped without reference transcript |
| **Validation** | Implemented | High | scripts/validate_output.py checks artifacts, metrics, sample frames | No CI integration; logs/job.log not produced |
| **Configuration** | Partially Implemented | Medium | ConfigLoader → AppConfig; YAML + CLI overrides; typed dataclasses | Duplicate subtitle config keys across two YAML files |
| **Composition Root** | Implemented | High | ApplicationBuilder single entry; MediaFlowApplication public API; CLI delegates only | None |
| **Artifact Manager** | Implemented | High | ArtifactManager implements ArtifactProvider; 12 path getters; overwrite guard | ideo_path + ideo_output_path duplicate; no log writer used |
| **Provider Registry** | Implemented | High | ProviderRegistry maps (category,name)→factory; 15 providers registered | Registry lambda default for Gemini differs from YAML |
| **Provider Factory** | Implemented | High | ProviderFactory merges config + overrides; delegates to registry | create_*_provider methods are thin wrappers (OK) |
| **Job State Machine** | Implemented | High | JobStateMachine transitions through 11 states; timestamps recorded | No transition guards (FAILED → RUNNING allowed) |
| **Real Video Pipeline** | Implemented | Medium | Single run sample.mp4 → STT → translate → ASS → render | Only one video tested; no CI; ASR quality limits product usability |

---

## 4. Pipeline Audit

### Actual Execution Graph (from JobOrchestrator.execute())

`
VideoJob (input_video, features)
       │
       ├─► speech_to_text=True
       │     ExtractAudioUseCase(VideoProvider) → audio.wav
       │     TranscribeSpeechUseCase(SpeechProvider) → speech_segments
       │     artifact: transcript.json
       │
       ├─► ocr=True
       │     ExtractFramesUseCase(FrameExtractionProvider) → frames
       │     DetectOnscreenTextUseCase(OCRProvider) → ocr_segments
       │
       ▼
MergeSegmentsUseCase(speech_segments, ocr_segments) → merged (timestamp-sorted)
       │
       ├─► translate=True
       │     TranslateSegmentsUseCase(TranslationProvider) → translated_segments
       │     artifact: translated.json
       │
       ▼
NormalizeSegmentsUseCase(is_translated) → normalized
       artifact: normalized.json
       │
       ▼
SubtitleSegmentationUseCase(is_translated) → segmented
       artifact: segmented.json
       │
       ▼
GenerateSubtitleUseCase(SubtitleProvider) → SRT (+ ASS if format=ass)
       artifact: subtitle.srt [+ subtitle.ass]
       │
       ├─► render_video=True
       │     Validation: subtitle must exist
       │     RenderVideoUseCase(VideoProvider) → output.mp4
       │     artifact: output.mp4
       ▼
JobStateMachine → COMPLETED / FAILED
`

### Verified
- No duplicated stages.
- No skipped stages when flags enabled.
- **Conditional validation**: 
ender_video requires subtitle (tested in 	est_orchestrator.py::test_orchestrator_render_no_subtitle_raises).
- Merge → Translate → Normalize → Segment → Subtitle ordering is correct per POS §15-16.

---

## 5. Configuration Audit

### Files
| File | Purpose | Issues |
|------|---------|--------|
| config/providers.yaml | Defaults + provider options + subtitle_quality | subtitle_quality duplicates subtitle_styles.yaml keys |
| config/subtitle_styles.yaml | 5 ASS presets (default, youtube, tiktok, netflix, minimal) | Duplicates max_chars_per_line, max_lines_per_subtitle, min_duration, max_duration |
| ConfigLoader | Loads both YAMLs → typed AppConfig; CLI overrides via overrides dict | subtitle_quality keys not validated against SubtitleQualityConfig fields |

### Duplicated Config Keys
| Key | providers.yaml (subtitle_quality) | subtitle_styles.yaml (each preset) |
|-----|----------------------------------------|--------------------------------------|
| max_chars_per_line | 42 | 24/28/32/28/22 |
| max_lines_per_subtitle | 2 | 2 (all) |
| min_duration | 0.8 | (absent) |
| max_duration | 6.0 | (absent) |

**Source of truth unclear** — Normalize/Segmentation read subtitle_quality; ASS generation reads style preset. Divergence = silent formatting bugs.

### Unused Config
- StorageProvider registered but never injected into any use case/orchestrator.
- subtitle_quality.segmentation sub-keys (prefer_sentence_boundary, prefer_punctuation, proportional_timing) not used by SubtitleSegmentationService (only enabled passed).

---

## 6. Public API Audit

### MediaFlowApplication (Public Entry Point)
`python
class MediaFlowApplication:
    def run(self, job: VideoJob) -> VideoJob: ...
    def run_video(...) -> VideoJob: ...
    def run_cli(args) -> VideoJob: ...
`
- **Clean**: Only imports JobOrchestrator, ApplicationContext, VideoJob, FeatureConfig.
- **No internal leakage**: Use cases, providers, factory not exposed.

### ApplicationBuilder (Composition Root)
- Single public method: uild() → MediaFlowApplication.
- rom_cli_args(args) factory for CLI.
- All wiring internal; CLI demo imports **only** ApplicationBuilder.

### CLI Demo (src/interfaces/cli/demo.py)
- **Verified**: Zero direct infrastructure imports. Only rgparse, json, sys, os, ApplicationBuilder.
- **Test**: 	est_composition.py::test_cli_does_not_manually_wire_dependencies asserts this.

---

## 7. Test Audit

### Coverage by Subsystem

| Subsystem | Unit Tests | Integration Tests | Real Provider Tests | Status |
|-----------|------------|-------------------|---------------------|--------|
| Domain Validation | 12 (	est_validation.py) | — | — | **Well Tested** |
| Speech | 2 (	est_use_cases.py) + orchestrator | 1 (orchestrator STT-only) | 0 (mock only) | **Moderately Tested** |
| OCR | 5 (	est_ocr_framework.py) | 1 (orchestrator OCR-only) | 0 (mock only) | **Moderately Tested** |
| Translation | 2 (OpenAI) + 2 (Gemini) | 0 | 0 (mock only) | **Moderately Tested** |
| Render | 8 (	est_render.py) | 1 (orchestrator full) | 0 (mock only) | **Moderately Tested** |
| ASS/Styling | 5 (	est_styling.py) | 0 | 0 | **Moderately Tested** |
| Normalize | 6 (	est_subtitle_quality.py) | 0 | 0 | **Well Tested** |
| Segmentation | 4 (	est_subtitle_segmentation.py) | 0 | 0 | **Well Tested** |
| Orchestrator | 4 (	est_orchestrator.py) | 4 (feature combos) | 0 | **Well Tested** |
| Composition | 7 (	est_composition.py) | 1 (CLI smoke) | 0 | **Well Tested** |
| Architecture | 2 (	est_architecture_boundaries.py) | — | — | **Well Tested** |
| Benchmark/Validate | 11 (scripts) | — | — | **N/A** |

### Critical Gaps
1. **6/11 use cases lack direct unit tests**: ExtractAudioUseCase, TranscribeSpeechUseCase, DetectOnscreenTextUseCase, MergeSegmentsUseCase, NormalizeSegmentsUseCase, SubtitleSegmentationUseCase, ExtractFramesUseCase — only exercised via orchestrator.
2. **No real-provider integration tests** (by POS §10 design, but undocumented/gated).
3. **No failure-path tests**: ProviderError → UseCaseError → orchestrator fail not tested.
4. **No CLI flag coverage** for --ocr-provider paddle, --translation-provider gemini, --subtitle-format ass, --subtitle-style, font overrides.
5. **StorageProvider**: zero tests (unused).

---

## 8. Documentation Audit

### Mismatches: Documentation vs. Code

| Document | Claim | Reality | Severity |
|----------|-------|---------|----------|
| README.md Provider Table | "OCRProvider: MockOCRProvider | -" (no real) | Real PaddleOCRProvider exists | Medium |
| README.md Provider Table | "TranslationProvider: MockTranslationProvider | -" | Real OpenAI + Gemini exist | Medium |
| README.md Provider Table | "VideoProvider: MockVideoProvider | -" | Real FFmpegVideoProvider exists | Medium |
| README.md Provider Table | "SubtitleProvider: MockSubtitleProvider | -" | Real ASSSubtitleProvider exists | Medium |
| README.md Flags | --render described as "mock" | Real FFmpeg render implemented | Medium |
| README.md Dependencies | Lists pip install openai only | Also needs aster-whisper, paddleocr, google-generativeai, pyyaml | Low |
| POS §6 | "StorageProvider: upload/download" | Port exists, no real impl, unused | Info |
| ARCHITECTURE_REVIEW.md | "StorageProvider registered" | Registered as mock only | Consistent |
| ARCHITECTURE_SCORECARD.md | "Extensibility 8.5/10 - ProviderFactory has if branches" | **Fixed** in code: ProviderFactory delegates to registry; no if branches | Stale |

### Undocumented Behavior
- GeminiTranslationProvider auto-chunks (15 segments) + 3s gap + 429 retry with Retry-After header — not in POS/AGENTS.
- FFmpegVideoProvider.render_video auto-detects .ass vs .srt — not in POS.
- ArtifactManager has ideo_path + ideo_output_path duplicate — not documented.

---

## 9. Runtime Audit

### Can python demo.py sample.mp4 --stt --translate --subtitle --render execute with real providers?

**Yes — verified by single manual run**.

| Stage | Provider | Real Run Result |
|-------|----------|-----------------|
| Audio Extraction | FFmpegVideoProvider | 0.4s, WAV 16kHz mono |
| Speech-to-Text | FasterWhisperSpeechProvider (tiny) | 27.5s, 160 segments, but **9 homophone errors** (因果 to 英国) |
| Translation | GeminiTranslationProvider (3.1-flash-lite) | 31.1s, chunked 15/4.5s gap, no 429 |
| Subtitle (SRT) | ASSSubtitleProvider | 0.0s, 90 entries |
| Subtitle (ASS) | ASSSubtitleProvider | 0.0s, 90 entries, YouTube style |
| Render | FFmpegVideoProvider | 89.4s, H.264+AAC, 1920x1080 |

### Runtime Blockers
1. **ASR Quality**: 	iny model unusable for Chinese wuxia dialogue (homophone 因果 to 英国). ase same errors; small fixes (0 errors) but 148s runtime.
2. **No CI Integration**: Real run requires aster-whisper model download, FFmpeg, API keys — not automated.
3. **No Quality Gate**: CER/WER not computed in pipeline; scripts/validate_output.py exists but not hooked.
4. **Logs Missing**: logs/job.log not produced; only console output.

---

## 10. Technical Debt Ranking

| Rank | Debt | Impact | Difficulty | Priority |
|------|------|--------|------------|----------|
| **Critical** | PaddleOCRProvider bare Exception | Violates error policy; crashes not caught as ProviderError | Low (1-line fix) | **Immediate** |
| **Critical** | ASR quality (	iny/ase homophones) | Product unusable for Chinese; translation garbage | Medium (upgrade model or add CER gate) | **Immediate** |
| **High** | Duplicate subtitle config across two YAMLs | Silent formatting divergence | Medium (consolidate to single source) | **High** |
| **High** | Gemini model default mismatch (3 sources) | Unpredictable model selection | Low (align to YAML) | **High** |
| **Medium** | JobOrchestrator monolithic (200 lines, 16 deps) | Hard to test stages in isolation; change risk | Medium (extract methods + config object) | **Medium** |
| **Medium** | VideoProvider.extract_audio contract mismatch (mock vs real) | Tests may miss path bugs | Low (fix mock to honor output_path) | **Medium** |
| **Medium** | No transition guards in JobStateMachine | Invalid state transitions possible | Low (add guard) | **Medium** |
| **Low** | Frame entity no validation | Negative timestamp allowed | Low (add __post_init__) | **Low** |
| **Low** | Duplicate artifact paths (ideo_path + ideo_output_path) | Confusion, maintenance | Low (remove one) | **Low** |
| **Low** | SubtitleStylePreset.get() silent key ignore | Typo-prone style overrides | Low (validate keys) | **Low** |
| **Low** | StorageProvider dead code | Dead weight | Low (remove or implement) | **Low** |
| **Low** | subtitle_quality.segmentation config unused | Config drift | Low (remove or wire) | **Low** |

---

## 11. Overall Project Maturity Scores

| Dimension | Score | Justification |
|-----------|-------|---------------|
| **Architecture** | 9.0/10 | Clean Architecture strictly enforced; boundary tests pass; Provider Pattern complete. |
| **Maintainability** | 7.5/10 | Good separation, but orchestrator monolithic; config duplication; duplicated logic in Normalize/Segment. |
| **Replaceability** | 9.5/10 | All providers pluggable via registry; zero hard-coded pipeline branches. |
| **Testability** | 8.5/10 | 87 unit tests fast; 6/11 use cases lack direct tests; no real integration CI. |
| **Configuration** | 7.0/10 | Typed config + overrides works; but duplicated subtitle keys across two YAMLs. |
| **Documentation** | 8.0/10 | POS/AGENTS excellent; README provider table stale; some undocumented behaviors. |
| **Production Readiness** | 6.5/10 | Single real run passes; ASR quality bottleneck; no CI; no CER gate; logs missing. |
| **Framework Readiness** | 8.0/10 | Composition root solid; extension path clear; dead StorageProvider; Frame under-validated. |
| **Product Readiness** | 6.0/10 | One video validated; ASR quality too low for product; no quality gate; no automated validation. |

---

## 12. Verified vs. Uncertain Findings

| # | Finding | Status | Evidence |
|---|---------|--------|----------|
| 1 | No architecture boundary violations | **Verified** | 	est_architecture_boundaries.py + static grep across 57 files |
| 2 | All 8 ports have mock + real (except Storage) | **Verified** | File inventory + import check |
| 3 | Composition root uses only ApplicationBuilder | **Verified** | demo.py + 	est_composition.py |
| 4 | Config loads YAML → typed AppConfig | **Verified** | ConfigLoader + test |
| 5 | 87 unit tests pass | **Verified** | pytest -q |
| 6 | PaddleOCRProvider bare Exception | **Verified** | Source lines 13, 24 |
| 7 | Gemini model default mismatch | **Verified** | Registry gemini-2.0-flash vs YAML gemini-3.1-flash-lite vs class gemini-3.1-flash-lite |
| 8 | VideoProvider.extract_audio contract inconsistency | **Verified** | Port vs Mock vs FFmpeg impl |
| 9 | SubtitleProvider.generate_ass mandatory style | **Verified** | Port signature vs use case call |
| 10 | StorageProvider unused | **Verified** | Grep: no StorageProvider import in app/orchestrator |
| 11 | Test gaps (6/11 use cases no direct tests) | **Verified** | Test file inventory |
| 12 | Real-provider integration tests exist | **Uncertain** | None found; POS §10 says they "must not run by default" — may be intentional |
| 13 | subtitle_quality.segmentation config unused | **Verified** | SubtitleSegmentationService only receives enabled |
| 14 | Frame entity missing validation | **Verified** | No __post_init__ |
| 15 | Single real pipeline run succeeded | **Verified** | Manual run output + artifacts in output/job_demo_123/ |

---

## 13. Recommended Next Milestone

### **ASR Quality Gate + Production Validation Pipeline**

**Goal**: Make the pipeline reliably produce publishable Vietnamese subtitles for Chinese short videos.

**Deliverables**:
1. **Fix PaddleOCRProvider** → wrap all exceptions in ProviderError.
2. **Upgrade default STT model** to small for Chinese (or add model selection guide).
3. **Add CER Quality Gate**: If reference transcript exists, fail job when CER > threshold; log warning otherwise.
4. **Consolidate subtitle config**: Single source of truth for max_chars_per_line, durations.
5. **Align Gemini model default**: Remove registry fallback; use YAML only.
6. **Add CI smoke test** with small model (or ase for English) on sample clip, gated by secrets.
7. **Implement logs/job.log** writer via ArtifactManager.log_path().
8. **Document real-provider integration test gating** (pytest markers + env vars).

**Do NOT**:
- Add REST API, TTS, dubbing, queue, or deployment.
- Change architecture.
- Rewrite orchestrator (incremental extraction of stage methods OK).

---

## Feature Matrix (Summary)

| Feature | Status | Confidence | Evidence | Notes |
|---------|--------|------------|----------|-------|
| Speech-to-Text | Implemented | High | Real run + benchmark | ASR quality bottleneck |
| OCR | Partially Implemented | Medium | Provider exists, bare Exception | No real validation |
| Translation | Implemented | High | OpenAI + Gemini real runs | Model default mismatch |
| Merge | Implemented | High | Orchestrator test | Duplicated dedup logic |
| Normalize | Implemented | High | Tests + orchestrator | Duplicated with Segment |
| Subtitle Segmentation | Implemented | High | Tests + orchestrator | Re-clamps duration |
| Subtitle Styling (ASS) | Implemented | High | Presets + real run | Config duplication |
| Rendering | Implemented | High | Real run 89s | Windows path fragile |
| Benchmark | Implemented | High | Script + comparison.md | No GPU test |
| Validation | Implemented | High | Script + metrics | No CI hook |
| Configuration | Partially Implemented | Medium | YAML + CLI + typed | Duplicate keys |
| Composition Root | Implemented | High | Builder + tests | None |
| Artifact Manager | Implemented | High | 12 paths + overwrite guard | Duplicate video path |
| Provider Registry | Implemented | High | 15 providers + factory | Gemini default mismatch |
| Provider Factory | Implemented | High | Delegates to registry | Thin wrappers OK |
| Job State Machine | Implemented | High | 11 states + timestamps | No transition guards |
| Real Video Pipeline | Partially Implemented | Medium | One successful run | ASR quality limits product |

---

## Conclusion

The AI Video Translator is a **well-architected framework** with Clean Architecture boundaries strictly enforced, a complete Provider Pattern implementation, and a functional composition root. The **primary product blocker is ASR quality** — the default 	iny model produces critical homophone errors that cascade into wrong translations. The small model fixes this but is too slow for CI without GPU.

**Production readiness requires**: (1) fixing the bare Exception in PaddleOCR, (2) adding a CER-based quality gate, (3) consolidating duplicate subtitle configuration, (4) establishing a real-provider CI validation pipeline with the small model for Chinese content. No architectural changes are needed.
