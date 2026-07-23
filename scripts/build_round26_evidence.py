"""Build Round 26 offline evidence from frozen Round 25 artifacts.

This script creates a frozen evidence directory for Round 26 by:
  1. Creating artifacts/round26/ directory structure
  2. Copying relevant artifacts from round25
  3. Creating package_report.json with current git SHA
  4. Scanning for ledger files (crop_classifier_ledger.jsonl or
     crop_classifier_v21_ledger.jsonl)
  5. Creating blocked_status.json if E2E hasn't run yet
  6. Recording SHA256 of key files
  7. Listing all files with sizes
"""
from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
ROUND25 = ROOT / "artifacts" / "round25"
ROUND26 = ROOT / "artifacts" / "round26"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_sha() -> str:
    import subprocess
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True, text=True, cwd=str(ROOT)
    )
    return result.stdout.strip()


def _copy_artifacts() -> None:
    """Copy relevant Round 25 artifacts into the Round 26 evidence directory."""
    files_to_copy = [
        "fn_diagnosis.json",
        "rule_sweep.json",
        "evaluation_v2_1_baseline.json",
        "package_report.json",
        "blocked_status.json",
        "crop_classifier_ledger.jsonl",
    ]
    for fname in files_to_copy:
        src = ROUND25 / fname
        if src.exists():
            shutil.copy2(src, ROUND26 / fname)
            print(f"  [COPY] {fname}")
        else:
            print(f"  [SKIP] {fname} (not found in round25)")


def _ledger_scan() -> dict[str, Any]:
    """Scan for crop_classifier_ledger files and report findings."""
    findings: dict[str, Any] = {}
    output_dir = ROOT / "output" / "round26-sample5" / "debug"
    for name in ("crop_classifier_ledger.jsonl", "crop_classifier_v21_ledger.jsonl"):
        locations = [
            ROUND26 / name,
            ROUND25 / name,
            output_dir / name,
            ROOT / "output" / name,
        ]
        for loc in locations:
            if loc.exists():
                entries = loc.read_text(encoding="utf-8").strip().splitlines()
                records = [json.loads(line) for line in entries if line.strip()]
                findings[name] = {
                    "path": str(loc),
                    "entry_count": len(records),
                    "line_count": len(entries),
                    "size_bytes": loc.stat().st_size,
                    "sha256": _sha256(loc),
                    "models": list({r.get("model", "unknown") for r in records}),
                    "earliest": min(r.get("timestamp", 0) for r in records),
                    "latest": max(r.get("timestamp", 0) for r in records),
                }
                print(f"  [LEDGER] Found {name} at {loc} ({len(records)} entries)")
                break
        else:
            print(f"  [LEDGER] {name} not found in any scanned location")
    return findings


def _record_file_hashes() -> dict[str, str]:
    """Record SHA256 of key evidence files."""
    key_files = [
        ROUND26 / "blocked_status.json",
        ROUND26 / "package_report.json",
        ROUND26 / "fn_diagnosis.json",
        ROUND26 / "rule_sweep.json",
        ROUND26 / "evaluation_v2_1_baseline.json",
    ]
    hashes: dict[str, str] = {}
    for path in key_files:
        if path.exists():
            hashes[path.name] = _sha256(path)
    return hashes


def _file_inventory() -> list[dict[str, Any]]:
    """List all files in the round26 directory with sizes."""
    inventory: list[dict[str, Any]] = []
    for path in sorted(ROUND26.rglob("*")):
        if path.is_file():
            inventory.append({
                "path": str(path.relative_to(ROUND26)),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
                "modified_utc": datetime.fromtimestamp(
                    path.stat().st_mtime, tz=timezone.utc
                ).isoformat(),
            })
    return inventory


def main() -> None:
    ROUND26.mkdir(parents=True, exist_ok=True)
    print(f"=== Round 26 Evidence Builder ===")
    print(f"Source: {ROUND25}")
    print(f"Target: {ROUND26}")
    print()

    # print("--- Copying artifacts from round25 ---")
    # _copy_artifacts()
    # print()

    print("--- Ledger scan ---")
    ledger_findings = _ledger_scan()
    print()

    git_sha = _git_sha()
    print(f"Git SHA: {git_sha}")
    print()

    print("--- Creating blocked status ---")
    blocked = {
        "status": "PENDING_E2E",
        "stage": "pending",
        "job_id": "round26-sample5",
        "round25_git_sha": git_sha,
        "note": "Round 26 E2E has not been executed yet.",
        "ledger_findings": ledger_findings,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    (ROUND26 / "blocked_status.json").write_text(
        json.dumps(blocked, indent=2), encoding="utf-8"
    )
    print("  Created blocked_status.json (status: PENDING_E2E)")
    print()

    print("--- Creating package report ---")
    carried_over = ["crop_classifier_ledger.jsonl", "evaluation_v2_1_baseline.json", "fn_diagnosis.json", "rule_sweep.json"]
    package_report = {
        "git_sha": git_sha,
        "blocked": True,
        "round": 26,
        "source": "round25",
        "carried_over_from_round25": carried_over,
        "included": {
            "round25_artifacts": False,
            "round26_ledger": bool(ledger_findings.get("crop_classifier_ledger.jsonl")),
            "round26_v21_ledger": bool(ledger_findings.get("crop_classifier_v21_ledger.jsonl")),
        },
        "file_hashes": _record_file_hashes(),
        "note": "Evidence directory for Round 26. E2E has not been run.",
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    (ROUND26 / "package_report.json").write_text(
        json.dumps(package_report, indent=2), encoding="utf-8"
    )
    print("  Created package_report.json")
    print()

    print("--- File inventory ---")
    inventory = _file_inventory()
    (ROUND26 / "file_inventory.json").write_text(
        json.dumps(inventory, indent=2), encoding="utf-8"
    )
    for entry in inventory:
        size_kb = entry["size_bytes"] / 1024
        print(f"  {entry['path']:55s} {size_kb:>8.2f} KB")
    print()

    total_kb = sum(e["size_bytes"] for e in inventory) / 1024
    print(f"=== Summary ===")
    print(f"  Files: {len(inventory)}")
    print(f"  Total size: {total_kb:.2f} KB")
    print(f"  Ledger found: {bool(ledger_findings.get('crop_classifier_ledger'))}")
    print(f"  V21 Ledger found: {bool(ledger_findings.get('crop_classifier_v21_ledger'))}")
    print(f"  Status: PENDING_E2E (blocked)")
    print(f"  Git SHA: {git_sha}")


if __name__ == "__main__":
    main()
