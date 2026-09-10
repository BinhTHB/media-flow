# Changelog

All notable changes to MediaFlow are documented in this file.

## [Unreleased]

### Added

- P2 offline quality workflow for Python 3.12 on Windows/Linux: dependency install, compile, architecture checks, dependency-free secret scan, and full pytest.
- Reproducible local `scripts/quality.py` gate, quality/operations documentation, and a reviewed benchmark manifest/schema/fixture contract separated from one-off scripts.
- Structured safe job provenance/metrics plus deterministic lifecycle tests for cancellation, duplicate-job claims, bounded events, and cleanup quotas.
- Durable atomic job state, event, and result documents under `outputs/jobs/<job_id>` with restart-safe list/get recovery and secret redaction.
- Validated provider configuration resolution with explicit runtime > environment > project config > code default precedence and duplicate-key rejection.
- Offline P1 contract tests for configuration precedence, invalid YAML, status aggregation, persistence/recovery, and preflight artifact safety.
- Local web studio and dubbing surfaces with truthful capability/preflight reporting, plus audio dubbing, subtitle refinement, and provider-routing adapters.

### Changed

- Unified CLI and local web execution behind one Python `JobRunner`/composition path with structured progress and failure events; the web runtime no longer parses CLI stdout.
- Standardized processing outcomes as `SUCCESS`, `PARTIAL`, `FAILED`, or `BLOCKED`, including warnings and artifact provenance; added lightweight timeout/cancellation checkpoints.
- Moved input/output/runtime preflight ahead of job and output directory creation where possible, and made provider construction conditional on enabled capabilities.
- Preserved `quality`, `balanced`, and `fast` runtime profiles across entry points.
- Enforced explicit audio extraction injection, strict BBox/SpeechSegment validation, and filesystem-based architecture guards.
- Expanded media pipeline, provider routing, detection/rendering, and offline contract coverage while keeping provider/media E2E opt-in.

## Session record — repository completion

> **Privacy/redaction:** This audit record intentionally excludes prompts/instructions, personal or team identifiers, email addresses, internal orchestration details, credentials and secret values, API keys, transient terminal identifiers/paths, and other sensitive data. Command output is summarized rather than copied verbatim.

### Scope and decisions

- The session began with a request to bootstrap the repository working rules. The requested `akidevrule` utility was not installed, so its bootstrap command could not run; repository-native guidance was used instead.
- Rules, memory, and project documentation were reviewed by name/topic: `AGENTS.md` (safe workflow and dependency rules), `docs/PROJECT_OPERATING_SPECIFICATION.md` (feature-first modular monolith and source-of-truth ownership), `docs/ARCHITECTURE.md` (Audio/Visual boundaries and ports), `docs/MIGRATION_PLAN.md` (phased migration), `docs/FEATURE_CONTRACTS.md`, `docs/PROVIDERS.md`, `docs/GEMINI_RUNTIME.md`, `docs/E2E_VALIDATION.md`, `docs/QUALITY_AND_OPERATIONS.md`, and the root/docs/scripts README material. Available project/rule memory was used only at the topic level; no private instruction text is reproduced here.
- The repository-analysis request covered architecture, runtime paths, contracts, providers, configuration, tests, operations, frontend/API surfaces, and delivery readiness.
- Git clarification: the repository already had Git metadata and `.gitignore`; the working tree was not clean. Existing generated/untracked work was preserved rather than reset or deleted.
- After the analysis and gap review, the user selected the complete roadmap: P0 baseline, P0 architecture, P1 runtime, P1 contracts, P2 quality, and final ship.

### Analysis findings

| Area | Finding / gap |
|---|---|
| Architecture | Clean/feature-oriented intent existed, but legacy policy-driven orchestration and overlapping entry paths weakened ownership and dependency boundaries. |
| Processing flow | Audio, Visual, subtitle enrichment, rendering, and mux behavior needed explicit independent pipelines and one thin composition path. |
| Runtime | CLI and local web execution were not yet consistently backed by one structured job runner; progress, cancellation, timeout, persistence, and restart recovery needed stronger semantics. |
| Contracts | Validation, serialization, status aggregation, artifact provenance, provider configuration precedence, and failure/block reporting needed stable, tested contracts. |
| Providers | Optional providers needed lazy capability-aware construction, explicit routing, shared ports, and truthful unavailable/blocked outcomes rather than implicit success. |
| Operations | Architecture enforcement, secret scanning, reproducible offline quality checks, benchmark governance, CI, and lifecycle cleanup needed a documented gate. |
| Delivery | Real-provider/media validation remained environment-dependent and had to stay opt-in rather than being represented as completed offline verification. |

### Implemented roadmap

| Phase | Main work completed |
|---|---|
| P0 baseline | Captured the existing automated baseline and repository state before structural work; preserved the dirty working tree and existing artifacts. |
| P0 architecture | Strengthened core contracts and validation; separated Audio and Visual feature ownership; kept detector/ASR/OCR/translation/TTS/video implementations behind ports; added filesystem architecture guards. |
| P1 runtime | Unified CLI/E2E and local web execution around `src/interfaces/jobs/runner.py` and composition wiring; added structured progress/failure events, preflight checks, cancellation/timeout checkpoints, safe job workspaces, and web studio/dubbing surfaces. |
| P1 contracts | Standardized `ProcessingResult` outcomes (`SUCCESS`, `PARTIAL`, `FAILED`, `BLOCKED`), warnings, provenance, config precedence, duplicate-key rejection, atomic job documents, restart recovery, and artifact-safety checks. |
| P2 quality | Added `scripts/quality.py`, `scripts/secret_scan.py`, offline CI, architecture checks, compile checks, lifecycle/cleanup coverage, benchmark manifest/schema/fixture governance, and quality/operations documentation. |
| Final ship | Re-ran focused and full verification, checked diffs/secrets/compilation/smokes, updated public documentation and changelog, and created the local completion commit. |

### Principal files and capabilities

- Core/application: `src/core/contracts.py`, validated bbox/speech/subtitle models, `src/application/process_video.py`, `src/application/result.py`, and proposal DTO contracts.
- Feature flows: independent `src/features/audio/` pipeline/composer/ports and `src/features/visual/` pipeline, subtitle event segmentation/refinement/planning/rendering, and anchored text overlay.
- Infrastructure: YOLO detection/proposals; Gemini/Groq/OpenRouter speech and correction routes; Gemini/OpenRouter translation; Edge, ElevenLabs, Fish, Gemini Live, and OpenRouter TTS routing; FFmpeg/video scene handling; stage/visual caches; and ElevenLabs dubbing integration.
- Interfaces/runtime: composition/config/provider resolution, durable job runner/store, local web server, and Studio/Dubbing frontend pages.
- Quality/contracts: `.github/workflows/quality.yml`, benchmark manifest/schema/lifecycle fixture, `scripts/quality.py`, `scripts/secret_scan.py`, expanded architecture/runtime/provider/web/audio/visual/P0/P1/P2 tests, and updated README/architecture/E2E/feature/operations documentation.

### Verification chronology

| Order | Run | Result |
|---:|---|---|
| 1 | Initial full baseline | **87 passed, 3 skipped**. |
| 2 | Post-architecture/runtime suite | **112 passed, 5 skipped**. |
| 3 | Post-contract suite | **114 passed, 5 skipped**. |
| 4 | Focused P1/P2 contract and operations checks | **12 passed**. |
| 5 | Focused lifecycle checks | **8 passed**. |
| 6 | Final full suite | **128 passed, 5 skipped**. |
| 7 | Architecture boundary suite | **5 passed**. |
| 8 | Additional gates | `compileall`, AST/dependency-boundary checks, dependency-free secret scan, Markdown/basic diff checks, `git diff --check`, and applicable CLI/web/preflight smokes completed successfully. |

### Failed or limited attempts

- Kiro integration was unavailable (`ENOENT`), and `akidevrule` was not installed; repository-native rules and documents were followed instead.
- Some commands were rejected because they targeted paths outside the allowed project scope.
- One PowerShell-form command was accidentally run by `cmd.exe`, and one `tree` invocation failed because too many parameters were supplied; both were replaced with compatible inspection methods.
- System Python lacked `opencc`; project virtual-environment/offline paths were used where available rather than changing global Python.
- One focused P2 test failed on its first run, the defect was corrected, and the focused plus full suites then passed.
- An attempted read of a missing `demo.py` failed; active entry points were located and verified instead.
- Real media/provider/network/GPU runs and manual audiovisual review were not executed in the final offline verification. These remain explicit opt-in validation limits, not implied successes.

### Delivery and residual state

- Local completion commit: `484fa243eec76170d5c6638f09f491d51ad33238` (`Complete MediaFlow runtime roadmap`), recording **98 files changed, 11,756 insertions, and 447 deletions**.
- Residual untracked content was deliberately not included or deleted: Postman globals/workspace material and approximately **95 one-off analysis, benchmark, inspection, review, and smoke scripts** under `scripts/`.
- No push, tag, network publication, destructive clean, or reset was performed during delivery.
