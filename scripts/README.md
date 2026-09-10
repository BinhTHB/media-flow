# Script policy

## Production entry points

- `run_e2e.py`: supported local pipeline entry point.
- `run_final_dubbing.py`: supported quality dubbing entry point.
- `quality.py`: offline compile, secret scan, architecture and pytest gate.
- `secret_scan.py`: dependency-free committed-file credential guard.

Production code must not import `scripts` or `benchmarks`.

## One-off and provider scripts

The remaining `benchmark_*`, `probe_*`, `inspect_*`, `review_*`, `smoke_*`, and `debug_*` files are local experiments. They may call providers, require credentials, or depend on local media. They are not part of CI and must not be bulk-staged. Promote reusable behavior into `src/` behind a port and deterministic tests instead of growing a one-off script into production.

## Benchmark contract

Canonical benchmark metadata lives in `benchmarks/manifest.json`, validated conceptually by `benchmarks/schema.json`, with small committed fixtures under `benchmarks/fixtures/`. Threshold changes require review and a changelog note. Generated media and reports stay under gitignored `outputs/`.
