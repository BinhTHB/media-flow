# Project Operating Specification (POS)

## Table of Contents
1. [Vision](#1-vision)
2. [Engineering Principles](#2-engineering-principles)
3. [Core Architecture](#3-core-architecture)
4. [Feature System](#4-feature-system)
5. [Job Orchestrator Rules](#5-job-orchestrator-rules)
6. [Provider Specification](#6-provider-specification)
7. [Data Contract Rules](#7-data-contract-rules)
8. [Error Handling Policy](#8-error-handling-policy)
9. [Coding Standards](#9-coding-standards)
10. [Testing Strategy](#10-testing-strategy)
11. [Extension Guide](#11-extension-guide)
12. [Roadmap Philosophy](#12-roadmap-philosophy)
13. [Internal Framework Components](#13-internal-framework-components)

---

## 1. Vision

### Long-term Goals
The AI Video Translator project aims to build an industrial-grade, highly-configurable, and decoupled pipeline for translating short video contents (less than 5 minutes) from source languages (primarily English/Chinese) into Vietnamese.

### Scope
- Decoupled video process orchestration: Audio extraction, Speech Transcription (STT), On-screen text recognition (OCR), Translation, Subtitle generation, and Video subtitle burning.
- Feature-based configuration allowing selective execution of parts of the translation pipeline.
- Provider abstraction to support plug-and-play swaps of underlying AI frameworks (e.g., Whisper vs. SenseVoice, GPT vs. Gemini).

### Out of Scope
- Direct deployment or VPS integration logic inside Domain/Application layers.
- Real-time video translation streaming (the focus is on batch/short-form asynchronous translation).
- In-memory video/audio editing logic (must be delegated to specialized external providers).

---

## 2. Engineering Principles

### Feature-first
Every processing capability (STT, OCR, translation, etc.) is represented as an independent, pluggable module. The codebase is organized such that adding, removing, or modifying a feature has minimal footprint on other features.

### Clean Architecture
We enforce a strict boundaries layout:
```
[ Interfaces ] ──> [ Application (Use Cases & Ports) ] ──> [ Domain (Entities) ]
     ▲                                                          ▲
     └─────────────────── [ Infrastructure ] ───────────────────┘
```
No inner layer can import or depend on outer layers.

### Provider Pattern
External services (like translation APIs, transcription models, storage, or video rendering engines) must implement defined ports (interfaces) in the Application layer. Their concrete classes belong in the Infrastructure layer.

### Dependency Inversion
Higher-level policies (Use Cases) do not depend on lower-level details (Infrastructure). Both depend on abstractions (Ports). Abstractions are defined at the application layer and implemented at the infrastructure layer.

### Domain Independence
The core domain entities represent pure business logic and models. They do not know about Whisper, GPT, FFmpeg, GitHub Actions, Supabase, or any other framework.

### Interface Stability
Use case inputs, use case outputs, and Port interfaces must remain stable. Modifications to an implementation must not alter the interface, protecting callers from cascading changes.

### Composition over Inheritance
Features and execution pipelines are composed dynamically at runtime via the `JobOrchestrator` rather than inheriting from rigid pipeline classes.

### Data Contracts as the Source of Truth
We enforce strict data schema contracts for `VideoJob` and its nested components. Any modification to these structures must maintain backward compatibility.

---

## 3. Core Architecture

### Domain
- **Location**: `src/domain/`
- **Responsibility**: Houses core business structures (`VideoJob`, `Segment`, `SubtitleItem`, `FeatureConfig`).
- **Constraints**: 0 external dependencies (except Python standard library). No business orchestrations or database calls. Only domain-level data validation during object initialization.

### Application
- **Location**: `src/application/`
- **Responsibility**: Contains use cases (`ExtractAudioUseCase`, `TranscribeSpeechUseCase`, etc.) and ports (interface definitions).
- **Constraints**: Depend only on Domain layer and internal ports. Under no circumstance should this layer import from `infrastructure` or external libraries.

### Infrastructure
- **Location**: `src/infrastructure/`
- **Responsibility**: Concrete implementations of ports (e.g., `FasterWhisperSpeechProvider`, `MockTranslationProvider`). Handles integration with system tools (FFmpeg), databases, filesystems, and APIs.
- **Constraints**: May import third-party libraries (`faster-whisper`, `openai`, etc.). Must catch low-level errors and map them to domain-compatible exceptions defined in the shared module.

### Interfaces
- **Location**: `src/interfaces/`
- **Responsibility**: Exposes the system to the outside world (e.g., CLI parsing, API controllers/routers). Instantiates and injects concrete providers into use cases, and invokes the orchestrator.
- **Constraints**: No direct domain logic. CLI/API controllers should only parse requests, invoke use cases, and serialize responses.

---

## 4. Feature System

The processing capabilities of the system are split into independent, optional features governed by `FeatureConfig`.

### Speech-to-Text (`speech_to_text`)
- **Purpose**: Extracts spoken words from the video's audio track.
- **Input**: `audio_path` (string)
- **Output**: List of `Segment` entities (source="speech")
- **Dependencies**: Requires audio extraction from `VideoProvider`.
- **Pluggability**: Optional. Can be bypassed if only OCR translation is needed.

### OCR (`ocr`)
- **Purpose**: Recognizes on-screen texts/titles in the video frames.
- **Input**: List of extracted `Frame` entities
- **Output**: List of `Segment` entities (source="ocr")
- **Dependencies**: Requires `FrameExtractionProvider`.
- **Pluggability**: Optional. OCR must not require STT to run.

### Translation (`translate`)
- **Purpose**: Translates segment text into the target language (Vietnamese).
- **Input**: List of `Segment` entities
- **Output**: List of `Segment` entities with populated `translated_text`
- **Dependencies**: Depends on results from Speech-to-Text or OCR.
- **Pluggability**: Optional (can transcribe only without translating).

### Subtitle (`subtitle`)
- **Purpose**: Formats translated segments into an SRT/VTT file structure.
- **Input**: List of `Segment` entities (with `translated_text` populated)
- **Output**: Subtitle content string (SRT format)
- **Dependencies**: None.
- **Pluggability**: Optional.

### Rendering (`render_video`)
- **Purpose**: Burns subtitles or translations back into the video file.
- **Input**: `input_video` (string), `subtitle_path` (string)
- **Output**: `output_video_path` (string)
- **Dependencies**: Requires `Subtitle` and `VideoProvider` components.
- **Pluggability**: Optional.

---

## 5. Job Orchestrator Rules

The `JobOrchestrator` is responsible for running use cases sequentially, conditional on the `FeatureConfig` flags.

### Execution Flow
```
[Video Job Input]
       │
       ├──► (speech_to_text=True) ──► Extract Audio ──► Transcribe ──┐
       │                                                             ├─► [Merge Segments] ──► (translate=True) ──► Translate ──► (subtitle=True) ──► Generate SRT ──► (render_video=True) ──► Render Video
       └──► (ocr=True) ─────────────► Detect On-screen Text ─────────┘
```

### Dependency Handling
- The orchestrator will dynamically route output paths. For instance, if `render_video` is enabled but `subtitle` is disabled, the orchestrator must raise a `ValidationError` or use default outputs.
- It does not instantiate concrete providers; all dependencies must be pre-injected into use cases and passed to the orchestrator's constructor.

### Skip & Failure Policies
- **Skip Policy**: If a feature flag is set to `False`, the orchestrator immediately skips that step.
- **Failure Policy**: Any exception thrown by use cases must propagate out of the orchestrator to ensure data consistency, unless a retry policy is active. Use cases must fail fast and cleanly.

---

## 6. Provider Specification

A Provider is an adapter in the Infrastructure layer that bridges external tools or APIs to our Application layer interfaces.

### Lifecycle & Strategy
- Providers must be stateless or manage their states transparently.
- Dependencies such as model sizes, API keys, and device specifications must be injected during Provider initialization.
- Every provider must implement a clean replacement path: developers must be able to switch providers simply by swapping the concrete instance in the interface entry point.

### Interfaces Table
- **SpeechProvider**: `transcribe(audio_path: str, source_language: str) -> List[Segment]`
- **FrameExtractionProvider**: `extract_frames(video_path: str, interval: float, output_dir: str) -> List[Frame]`
- **OCRProvider**: `detect_text(frames: List[Frame]) -> List[Segment]`
- **TranslationProvider**: `translate(segments: List[Segment], target_language: str) -> List[Segment]`
- **VideoProvider**: `extract_audio(video_path: str) -> str`, `render_video(video_path: str, subtitle_path: str) -> str`
- **SubtitleProvider**: `generate_srt(segments: List[Segment]) -> str`
- **StorageProvider**: `upload(file_path: str) -> str`, `download(file_url: str) -> str`

---

## 7. Data Contract Rules

Our contracts guarantee API stability and reliable serialization.

### Entities
- **VideoJob**: Holds metadata about the translation request, execution configs, output paths, and processed segments.
- **Frame**: A video frame artifact containing a timestamp and image path.
- **Segment**: A transcript/OCR unit containing timestamps, text, translation, source, and confidence score.
- **SubtitleItem**: Single caption item with sequence number and formatted timestamps.
- **FeatureConfig**: Configuration flags mapping boolean flags to execution paths.

### Serialization and Rules
- All entities must expose a `to_dict()` method returning standard JSON-serializable types.
- **Immutable Fields**: `job_id`, `input_video`. Once instantiated, they must not change.
- **Nullable / Optional Fields**: `audio_path`, `output_video_path`, `subtitle_path`, `subtitle_content` must default to empty strings.
- **Backward Compatibility**: Adding new optional fields is permitted. Deleting fields or changing their type is forbidden without a formal RFC.

---

## 8. Error Handling Policy

- **Recoverable Errors**: Network flakiness during translation. Providers must handle retries internally (e.g., exponential backoff) up to a threshold before raising an error.
- **Fatal Errors**: `ValidationError` (schema mismatch, missing input files), `ImportError` (missing optional providers like `faster-whisper`), or unrecoverable API keys. These must fail immediately.
- **Exceptions Mapping**: Infrastructure exceptions must be wrapped and re-thrown as `ProviderError`. Use case validation errors must raise `UseCaseError` or `ValidationError`.

---

## 9. Coding Standards

### Forbidden
- Importing modules from `src/infrastructure/` or `src/interfaces/` into `src/domain/` or `src/application/`.
- Accessing external libraries (like `faster_whisper` or `openai`) or system commands (like `ffmpeg`) inside use cases.
- Writing business-level orchestrations (such as switching use cases) inside concrete providers.

### Required
- Use dependency injection to supply use cases with providers.
- Maintain type hints across all function signatures.
- Write a unit test for every added use case and validation rule.

---

## 10. Testing Strategy

### Unit Tests
- Placed in `tests/`.
- Test domain validation rules (e.g., negative timestamps, invalid segment source).
- Test use cases in isolation using mock providers to avoid network and model overhead.

### Integration Tests
- Verify orchestrator pipeline execution with combinations of flags (e.g., STT only, STT + OCR + Translate).
- Test concrete provider setups with local files where appropriate.

---

## 11. Extension Guide

To add a new feature (e.g., **Voice Cloning / Dubbing**):

1. **Entity**: Define `DubbingConfig` in `src/domain/entities/` and add validation.
2. **Port**: Define `DubbingProvider` interface in `src/application/ports/`.
3. **Provider**: Implement `MockDubbingProvider` and the real provider in `src/infrastructure/providers/dubbing/`.
4. **Use Case**: Create `GenerateDubbedAudioUseCase` in `src/application/use_cases/`.
5. **Orchestrator**: Update `JobOrchestrator` to coordinate the dubbing use case if the feature is enabled.
6. **Tests**: Add unit tests verifying validation and orchestrator integrations.
7. **Documentation**: Update the POS features list and the README.

---

## 12. Roadmap Philosophy

- **Stable Interfaces over Rapid Features**: We prioritize interface stability. Changing an interface requires updating all implementing providers, which increases regression risk.
- **Modular Providers**: Core capabilities must be provider-agnostic. We replace models, not architecture.
- **Strict Decoupling**: Keep domain layers simple and free of frameworks so that we can easily migrate platforms (e.g., running on GitHub Actions runner vs. a VPS) without refactoring the application.


---

## 13. Internal Framework Components

The internal framework exists to keep feature and provider growth safe without weakening Clean Architecture.

### ProviderRegistry

- **Location**: `src/infrastructure/registry/`
- **Responsibility**: Maintains provider mappings by category and name.
- **Allowed Dependencies**: Concrete infrastructure providers.
- **Forbidden**: Domain/Application imports from registry.

### ProviderFactory

- **Location**: `src/infrastructure/factory/`
- **Responsibility**: Creates provider instances from `config/providers.yaml` and CLI/API overrides.
- **Allowed Dependencies**: ProviderRegistry and infrastructure providers.
- **Usage**: Composition roots such as CLI/API only.

### ArtifactManager

- **Location**: `src/infrastructure/artifacts/`
- **Responsibility**: Owns artifact paths and file I/O for job outputs.
- **Application Boundary**: Application talks to it only through `ArtifactProvider` port.
- **Default Layout**:

```text
output/<job_id>/
├── audio/audio.wav
├── transcripts/transcript.json
├── transcripts/translated.json
├── subtitles/subtitle.srt
├── videos/output.mp4
└── logs/job.log
```

### JobStateMachine

- **Domain Part**: `JobStatus` lives in `src/domain/entities/`.
- **Application Part**: Transition policy lives in `src/application/orchestrator/`.
- **Responsibility**: Makes execution observable through `created`, `running`, feature-specific states, `completed`, `failed`, and `skipped`.
- **Forbidden**: Infrastructure must not control state transitions.


---

## 14. OCR Framework Rules

OCR is a first-class optional subsystem parallel to Speech-to-Text.

- Frame extraction and OCR must remain separate responsibilities.
- OCR providers receive `Frame` entities, never raw video paths or OpenCV image objects.
- OCR providers return `Segment` entities with `source="ocr"`.
- Merge policy belongs to Application via `MergeSegmentsUseCase`.
- Providers must not sort, translate, deduplicate, or subtitle text.
- Initial merge policy is timestamp sort with stable ordering and no deduplication.
- PaddleOCR belongs only in Infrastructure and must not leak into Domain/Application.

Default OCR artifact layout:

```text
output/<job_id>/
└── frames/
    ├── frame_0000.jpg
    └── frame_0001.jpg
```


---

## 15. Subtitle Quality Layer

Subtitle quality is an Application-layer normalization step between translation and subtitle generation.

```text
MergeSegmentsUseCase
      ↓
TranslateSegmentsUseCase (optional)
      ↓
NormalizeSegmentsUseCase
      ↓
GenerateSubtitleUseCase
```

Rules:

- Normalization must not call external APIs.
- Normalization must not live inside providers.
- Subtitle providers only format already-normalized text into subtitle formats.
- If translation is enabled, normalization operates on `translated_text`.
- If translation is disabled, normalization operates on original `text`.
- The normalized artifact is `output/<job_id>/transcripts/normalized.json`.

The initial normalization policy is rule-based: trim, remove empty entries, sort by timestamp, deduplicate similar text, prefer speech over OCR when overlapping text is similar, clamp durations, merge safe short neighbors, and wrap long text into subtitle-friendly lines.


---

## 16. Subtitle Segmentation Layer

Subtitle segmentation is an Application-layer step after normalization and before subtitle generation.

```text
NormalizeSegmentsUseCase
      ↓
SubtitleSegmentationUseCase
      ↓
GenerateSubtitleUseCase
```

Rules:

- Segmentation must never lose text or drop words.
- Timestamps must be distributed proportionally or deterministic.
- Minimum duration (e.g. 0.8s) and maximum duration (e.g. 6.0s) must be respected.
- Output artifact path is `output/<job_id>/transcripts/segmented.json`.
- Formatting of subtitle items is still delegated to `SubtitleTextFormatter` but slicing/truncation of text lines is disabled in normalization.


---

## 17. Video Render Engine

The Video Render Engine burns subtitles directly into the video stream using FFmpeg.

```text
GenerateSubtitleUseCase
      ↓
RenderVideoUseCase
      ↓
output/<job_id>/videos/output.mp4
```

### Rules & Architecture:

- Render is optional, governed by `FeatureConfig.render_video`.
- If `render_video` is enabled, the system enforces that the `subtitle` feature must also be enabled and the subtitle file must successfully exist. Otherwise, it throws a `ValidationError`.
- The orchestrator passes the input video, generated subtitle path, and designated output video path (`self.artifact_provider.video_path(job.job_id)`) to `RenderVideoUseCase`.
- If a custom `output_video_path` is pre-set on the job, the orchestrator respects and passes that path instead.
- FFmpeg parameters escape Windows paths and double backslashes correctly, formatting path references specifically for FFmpeg video filters (`subtitles='C\:/path/to/subtitle.srt'`).
- Low-level FFmpeg process timeout errors or bad returns are caught and raised as `ProviderError`.


---

## 18. Subtitle Styling Framework

The Subtitle Styling Framework introduces ASS (Advanced SubStation Alpha) subtitle generation with full styling control, while preserving SRT compatibility.

### Domain Layer

- **SubtitleStyle** (`src/domain/entities/subtitle_style.py`):
  A pure data entity describing presentation only:
  ```python
  font_family: str
  font_size: int
  primary_color: str       # Hex #RRGGBB or ASS &HBBGGRR&
  outline_color: str
  outline_width: float
  shadow: float
  alignment: int           # 1-9 numpad positions (2 = bottom-center)
  margin_v: int
  margin_l: int
  margin_r: int
  bold: bool
  italic: bool
  spacing: float
  line_height: float
  background: str          # "outline", "box", "none"
  opacity: float           # 0.0 - 1.0
  ```

- **SubtitleStylePreset** (`src/domain/entities/subtitle_style_preset.py`):
  Built-in presets: `default`, `youtube`, `tiktok`, `netflix`, `minimal`.
  Extensible via `config/subtitle_styles.yaml`.

### Application Layer

- **ASSGenerator** (`src/application/services/ass_generator.py`):
  Stateless service. `generate(segments, style) -> str` (ASS content).
  No I/O, no rendering, no external dependencies.

- **SubtitleProvider** Port (`src/application/ports/subtitle_provider.py`):
  Extended with `generate_ass(segments: List[Segment], style: SubtitleStyle) -> str`.

### Infrastructure Layer

- **ASSSubtitleProvider** (`src/infrastructure/providers/subtitle/ass_subtitle_provider.py`):
  Delegates to `ASSGenerator` for ASS content; generates SRT via standard logic.

- **MockSubtitleProvider** (`src/infrastructure/providers/subtitle/mock_subtitle_provider.py`):
  Also implements `generate_ass` via `ASSGenerator` for testing.

- **ProviderRegistry**:
  Registered under category `"subtitle"` with name `"ass"`.

### Artifact Layout

Both formats coexist:
```
output/<job_id>/subtitles/
├── subtitle.srt
└── subtitle.ass
```

### Rendering Strategy

`FFmpegVideoProvider.render_video` auto-detects `subtitle.ass` in the same directory as the provided `subtitle_path` (which is `subtitle.srt`). If present, it uses:
```bash
ffmpeg -i input.mp4 -vf "ass='subtitle.ass'" -y output.mp4
```
Otherwise falls back to:
```bash
ffmpeg -i input.mp4 -vf "subtitles='subtitle.srt'" -y output.mp4
```

No Application-layer logic decides the filter; the provider decides.

### CLI Integration

```bash
python src/interfaces/cli/demo.py video.mp4 \
  --stt --translate --subtitle --render \
  --subtitle-format ass \
  --subtitle-style youtube \
  --font "Arial" --font-size 30 --outline 3.5 --shadow 2.0 --margin-v 20
```

- `--subtitle-format` selects output format (`srt`/`ass`).
- `--subtitle-style` selects built-in or custom preset from `config/subtitle_styles.yaml`.
- Optional overrides merge on top of the selected preset.

### Configuration

`config/subtitle_styles.yaml`:
```yaml
styles:
  default:
    font_family: Arial
    font_size: 24
    primary_color: "#FFFFFF"
    ...
  youtube:
    font_size: 28
    bold: true
    ...
```
Custom presets added here are automatically available via `--subtitle-style`.

### Non-Goals

The following are explicitly out of scope for this milestone:
- Animated karaoke / word highlighting
- Per-word timing
- Animated transitions
- TTS / Dubbing / Voice cloning
- Speaker color coding
- Template system
- Cloud rendering

### Future Extensions

- Custom font loading via FFmpeg fontconfig.
- Style inheritance / layering.
- Speaker-aware styles (multiple styles per job).
- ASS text animation tags (`{\fad}`, `{\t}`).


---

## 19. Application Composition Root

All entry points (CLI, REST API, Bot, Workers, GitHub Actions) must compose their dependencies via the centralized `ApplicationBuilder`. No entry point is allowed to manually build the orchestrator, use cases, or providers.

### Composition Architecture

```text
CLI / API / Worker
       │
       ▼
ApplicationBuilder (config + overrides)
       │
       ├─► ProviderFactory & Registry
       ├─► UseCases & ArtifactManager
       ├─► OrchestratorDependencies
       ▼
MediaFlowApplication (run / run_cli / run_video)
       │
       ▼
JobOrchestrator (sequential stages execution)
```

### Dependency Ownership

- **ApplicationContext**: Owns the configuration dicts, registry, factory, and `ArtifactManager` instance.
- **OrchestratorDependencies**: Explicitly packs all use cases, ports, and configuration configurations into a single data object.
- **MediaFlowApplication**: Exposes unified high-level public APIs and handles interface-level validations (e.g., CLI overwrite check).


---

## 20. Configuration Consolidation & Auto Registration

To simplify registry extensions, providers register themselves with the `ProviderRegistry` using explicit factory callbacks.

### Configuration Layer

- **ConfigLoader**:
  Loads configurations from filesystem, validates required properties, and populates the typed `AppConfig` entity.
- **AppConfig**:
  Contains typed configuration mappings: `defaults`, `providers`, `subtitle_quality`, `styles`. Enables future workers/endpoints to invoke pipeline overrides cleanly.

### Open/Closed Provider Registry

Instead of nesting creation conditionals inside `ProviderFactory`, the registration logic maps `(category, name) -> factory_fn`:
```python
registry.register("video", "ffmpeg", lambda opt: FFmpegVideoProvider(output_dir=opt.get("output_dir")))
```
- `ProviderFactory` delegates creation directly to `ProviderRegistry.create(category, name, options)`.
- Explicit registration is executed during registry instantiation (`register_default_providers`), preventing compiler dependency loops.
