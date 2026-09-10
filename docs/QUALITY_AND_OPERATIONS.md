# Quality and operations

## Offline local gate

Use the project interpreter. No provider credentials, network calls, GPU, or model execution are needed.

```powershell
.venv\Scripts\python.exe scripts\quality.py
```

Focused architecture gate:

```powershell
.venv\Scripts\python.exe scripts\quality.py --architecture-only
```

The gate compiles `src/` and `tests/`, scans tracked text files for obvious credentials, checks architecture boundaries, and runs the full pytest suite. Integration tests remain opt-in.

No lint/type dependency is added in P2: the current dependency set contains no dedicated linter or static type checker. CI therefore uses compile, AST architecture checks, the secret guard, and pytest rather than silently relying on undeclared tools.

## CI

`.github/workflows/quality.yml` runs the same offline checks on Windows and Linux with Python 3.12 after installing `requirements.txt`. It grants read-only repository permissions and has a job timeout.

## Job lifecycle

`JobRunner` timeout and cancellation are cooperative checkpoints between composition, processing, and finalization. Provider adapters retain their own I/O timeouts. `JobStore` supplies an exclusive per-job claim, cancellation marker, atomic redacted JSON state, bounded event history, and cleanup by age/count/bytes. Cleanup removes terminal jobs only and never an actively claimed job.

Structured events and metrics contain identifiers, stage/status, elapsed time, counts, profile, and provider names only. Credential-like keys are redacted before persistence/logging; raw media, prompts, environment values, and provider responses are not logged.

## Benchmarks

`benchmarks/manifest.json` is the reviewed source of benchmark cases and thresholds. Fixtures are small and offline. One-off/provider experiments remain outside CI and are classified in `scripts/README.md`; their generated reports belong under `outputs/`.
