"""Test that all sweep scripts import correctly and report BLOCKED/DATA_NEEDED without data."""
from __future__ import annotations
import json, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _run_script(name: str) -> dict:
    """Run a sweep script and return its JSON output."""
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / name)],
        capture_output=True, text=True, timeout=30,
        cwd=str(ROOT),
    )
    # Find last JSON dict in stdout (skip progress lines)
    idx = result.stdout.find("{")
    if idx >= 0:
        try:
            return json.loads(result.stdout[idx : result.stdout.rfind("}") + 1])
        except json.JSONDecodeError:
            pass
    return {"status": "NO_OUTPUT", "error": result.stderr[:200] if result.stderr else ""}


def test_sweep_r1_imports():
    """sweep_r1.py should import and report BLOCKED (no video)."""
    out = _run_script("sweep_r1.py")
    assert out.get("status") in ("BLOCKED", "COMPLETED"), f"Unexpected status: {out}"
    if out["status"] == "BLOCKED":
        assert "reason" in out


def test_sweep_r2_imports():
    """sweep_r2.py should import and report BLOCKED (no OCR data)."""
    out = _run_script("sweep_r2.py")
    assert out.get("status") in ("BLOCKED", "COMPLETED", "NO_DATA"), f"Unexpected status: {out}"


def test_sweep_r3_imports():
    """sweep_r3.py should import and report BLOCKED (no frames)."""
    out = _run_script("sweep_r3.py")
    assert out.get("status") in ("BLOCKED", "COMPLETED", "NO_DATA"), f"Unexpected status: {out}"


def test_sweep_step7_imports():
    """sweep_step7.py should report DATA_NEEDED."""
    out = _run_script("sweep_step7.py")
    assert out.get("status") == "DATA_NEEDED", f"Expected DATA_NEEDED, got: {out}"
