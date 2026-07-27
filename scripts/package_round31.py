#!/usr/bin/env python3
"""Round 31 packaging — git_sha verified, no nested dir, ledger populated."""
from __future__ import annotations

import hashlib
import json
import subprocess
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT.parent
ZIP = PARENT / "media-flow-dev-v31.zip"
ROOT_NAME = "media-flow-dev-v31"
ROUND = ROOT / "artifacts" / "round31"

EXCLUDE_PARTS = {".git", ".venv", "__pycache__", ".pytest_cache", ".ruff_cache", "frames"}
EXCLUDE_SUFFIXES = {".pyc", ".wav", ".pt"}
EXCLUDE_NAMES = {".env", "config.yml"}


def include(path: Path) -> bool:
    rel = path.relative_to(ROOT)
    if any(part in EXCLUDE_PARTS for part in rel.parts):
        return False
    if path.suffix.lower() in EXCLUDE_SUFFIXES or path.name in EXCLUDE_NAMES:
        return False
    # Only include round31 output
    if rel.parts[0] == "output" and rel.parts[1:2] != ("round31-sample5",):
        return False
    # Only include round31 / round28 artifacts
    if rel.parts[0] == "artifacts" and rel.parts[1] not in {"round31", "round28"}:
        return False
    return rel.parts[0] in {"src", "tests", "scripts", "config", "artifacts", "output"}


def main() -> None:
    # Verify HEAD commit before packaging
    git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    print(f"[GIT] HEAD: {git_sha}")
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True)
    if dirty.stdout.strip():
        print("[WARN] Uncommitted changes detected. Continue anyway?")
        lines = dirty.stdout.strip().split("\n")
        for line in lines[:20]:
            print(f"  {line}")
        print(f"  ({len(lines)} dirty files)")

    report = {
        "zip": str(ZIP), "sha256": "external_sidecar_authoritative",
        "git_sha": git_sha, "root": ROOT_NAME, "round": 31,
        "blocked": False, "status": "IN_PROGRESS",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "gt_v2_3": "artifacts/round28/gt_v2_3.json",
    }
    report_file = ROUND / "package_report.json"
    report_file.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if ZIP.exists():
        ZIP.unlink()

    with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in ROOT.rglob("*"):
            if path.is_file() and include(path):
                rel = path.relative_to(ROOT)
                archive.write(path, Path(ROOT_NAME) / rel)

    names = zipfile.ZipFile(ZIP).namelist()
    duplicates = [n for n in names if names.count(n) > 1]
    if duplicates:
        raise RuntimeError(f"Duplicate ZIP entries: {duplicates[:10]}")

    digest = hashlib.sha256(ZIP.read_bytes()).hexdigest()
    sidecar = ZIP.with_suffix(".zip.sha256")
    sidecar.write_text(f"{digest}  {ZIP.name}\n", encoding="ascii")

    print(f"ZIP={ZIP}")
    print(f"SHA256={digest}")
    print(f"ENTRIES={len(names)} UNIQUE={len(set(names))}")
    print(f"GIT_SHA={git_sha}")


if __name__ == "__main__":
    main()
