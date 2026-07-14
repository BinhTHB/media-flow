# Architecture Scorecard

**Project**: AI Video Translator
**Maturity Assessment Date**: 2026-07-10
**Overall Architecture Maturity Score**: **8.5 / 10**

---

## 1. Metrics and Scores

### 1.1 Architecture Maturity (Clean Architecture)
- **Score**: **9.0 / 10**
- **Justification**: Strict layering is enforced: Domain → Application → Infrastructure → Interfaces. Domain has 0 dependencies. Application only imports internal ports and domain. Realized via boundary tests checking import graphs.
- **Deduction**: Minor coupling in orchestrator directly importing 10 concrete use cases instead of a pipeline abstraction.

### 1.2 Maintainability
- **Score**: **7.5 / 10**
- **Justification**: Code is generally clean, with clear domain boundaries.
- **Deduction**: JobOrchestrator has a constructor with 16 parameters and a single 200-line `execute()` method with nested conditionals. Duplicate config keys in both `providers.yaml` and `subtitle_styles.yaml`.

### 1.3 Extensibility
- **Score**: **8.5 / 10**
- **Justification**: Feature-based modularity allows adding features (e.g. Dubbing) by registering configurations and inserting a new use-case step into the orchestrator.
- **Deduction**: Adding a new provider requires updating hardcoded `if` branches in `ProviderFactory` (violates Open/Closed).

### 1.4 Replaceability
- **Score**: **9.5 / 10**
- **Justification**: Decoupled using the Provider Pattern. Switching speech providers (e.g., mock to faster-whisper) or subtitle format (SRT to ASS) is a simple configuration change or CLI flag. Standardized exception mapping (`ProviderError`).

### 1.5 Testability
- **Score**: **9.0 / 10**
- **Justification**: Unit tests run 100% locally with mock providers. Orchestrator tests cover feature combinations. 66 tests passing in under 0.5s.
- **Deduction**: Lack of direct unit tests for individual orchestrator pipeline stages in isolation due to monolithic `execute()` method.

### 1.6 Provider Architecture
- **Score**: **9.0 / 10**
- **Justification**: All providers implement clean, abstract ports defined in `src/application/ports/`. The implementation lives strictly in `src/infrastructure/providers/`.
- **Deduction**: `ProviderFactory` has some provider-specific construction details.

### 1.7 Configuration Quality
- **Score**: **7.0 / 10**
- **Justification**: YAML configurations are clear. Providers, Whisper config, OpenAI, and subtitle quality are separated logically.
- **Deduction**: Overlapping formatting configuration between `providers.yaml` (under `subtitle_quality`) and `subtitle_styles.yaml` (under presets like `default`, `youtube`). This duplicates `max_chars_per_line` and durations.

### 1.8 Documentation Quality
- **Score**: **9.0 / 10**
- **Justification**: Handbooks (`AGENTS.md` and `PROJECT_OPERATING_SPECIFICATION.md`) are comprehensive and strictly adhered to by developers.
- **Deduction**: No documentation for developer-facing API or details on how to register new custom providers without modifying `ProviderFactory`.

---

## 2. Known Risks

- **FFmpeg dependency path issues on Windows**: Paths with spaces or drive colons (e.g. `C:\path`) can cause FFmpeg filters (e.g., `subtitles='C\:/path'`) to fail. Though escaped, Windows/UNIX path translation in docker/WSL is a potential point of failure.
- **OpenAI Translation Cost & Rate Limits**: Real API translation has no fallback or caching. High translation volumes could fail mid-job.
- **ProviderFactory Open/Closed Violation**: The risk of developers directly updating `ProviderFactory` with large conditional logic, leading to complex merge conflicts.
- **Configuration Drift**: Keeping duplicate values in `providers.yaml` and `subtitle_styles.yaml` in sync is a human error risk.

---

## 3. Recommended Next Milestone

### Milestone: Real Video Smoke Test & Validation
- **Goal**: Validate the complete pipeline against a real short video file (< 5 minutes) to confirm production readiness before adding new entry points or AI features.
- **Key Changes**:
  1. Run full pipeline with `faster-whisper` real transcription, `openai` real translation, and `ffmpeg` real subtitle burn-in on a sample video.
  2. Verify Vietnamese Unicode subtitle rendering, ASS/SRT artifact correctness, Windows path escaping, and output video playback.
  3. Establish CI-friendly real-video smoke test that runs on demand (not every commit due to cost/time).
  4. Document timing accuracy thresholds and acceptable error margins for subtitles.
