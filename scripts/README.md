# Script policy

## Production entry points

- `run_e2e.py`: supported local pipeline entry point.
- `run_final_dubbing.py`: supported quality dubbing entry point.
- `quality.py`: offline compile, secret scan, architecture and pytest gate.
- `secret_scan.py`: dependency-free committed-file credential guard.
- `get_youtube_refresh_token.py`: one-time OAuth helper for publishing (see `docs/PUBLISHING.md`).

Production code must not import `scripts` or `benchmarks`.

## Adding a script

A new script is only justified when it is a **supported entry point** — something
a user or CI is expected to run repeatedly. Name it for the operation it performs
and register it in this file.

Do not add `benchmark_*`, `probe_*`, `inspect_*`, `review_*`, `smoke_*`,
`debug_*`, `analyze_*`, `compare_*`, `diagnose_*`, or `make_*` scripts. Those
are one-off investigations: run them from a scratch directory, or promote the
reusable behavior into `src/` behind a port with deterministic tests. Git
history is the archive.

`.gitignore` still blocks those prefixes so a stray experiment cannot be
committed by accident.

## Benchmark contract

Canonical benchmark metadata lives in `benchmarks/manifest.json`, validated
conceptually by `benchmarks/schema.json`, with small committed fixtures under
`benchmarks/fixtures/`. Threshold changes require review and a changelog note.
Generated media and reports stay under gitignored `outputs/`.
