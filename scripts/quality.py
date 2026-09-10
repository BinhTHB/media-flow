#!/usr/bin/env python3
"""Reproducible offline quality entry point using repository dependencies only."""
from __future__ import annotations

import argparse
import compileall
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--architecture-only", action="store_true")
    args = parser.parse_args()
    if not compileall.compile_dir(ROOT / "src", quiet=1):
        return 1
    if not compileall.compile_dir(ROOT / "tests", quiet=1):
        return 1
    run([sys.executable, "scripts/secret_scan.py"])
    target = ["tests/test_architecture_boundaries.py"] if args.architecture_only else []
    run([sys.executable, "-m", "pytest", "-q", *target])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
