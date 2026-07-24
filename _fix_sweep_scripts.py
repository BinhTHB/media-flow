"""Fix syntax errors in sweep scripts and test (PowerShell escaping artifacts)."""
from __future__ import annotations
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent


def fix_file(rel_path, fixes):
    src = (ROOT / rel_path).read_text(encoding="utf-8")
    for old, new in fixes:
        src = src.replace(old, new)
    (ROOT / rel_path).write_text(src, encoding="utf-8")
    print(f"  Fixed: {rel_path}")


# 1. test_sweep_scripts.py
fix_file("tests/test_sweep_scripts.py", [
    ('without data.""""""', 'without data."""'),
    ('output.""""""', 'output."""'),
    ('"Unexpected status: %%s" %% out', 'f"Unexpected status: {out}"'),
    ('"Expected DATA_NEEDED, got: %%s" %% out', 'f"Expected DATA_NEEDED, got: {out}"'),
])

# 2. sweep_r1.py
fix_file("scripts/sweep_r1.py", [
    ("%%", "%"),
])

# 3. sweep_r2.py
fix_file("scripts/sweep_r2.py", [
    ('first).""""', 'first)."""'),
    ("%%", "%"),
])

# 4. sweep_r3.py
fix_file("scripts/sweep_r3.py", [
    ('frames."""""', 'frames."""'),
    ("%%", "%"),
])

print("All files fixed.")
