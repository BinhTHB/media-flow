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
