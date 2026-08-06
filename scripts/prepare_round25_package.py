"""Copy Round 25 E2E evidence and create the required source package."""
from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT.parent
ROUND25 = ROOT / "artifacts" / "round25"
JOB = ROOT / "output" / "round25-sample5"
ZIP = PARENT / "media-flow-dev-v25.zip"
EXCLUDED_PARTS = {".git", ".venv", "__pycache__", ".pytest_cache", ".ruff_cache", "frames"}
EXCLUDED_NAMES = {".env", "config.yml", "best.pt", "models"}
EXCLUDED_SUFFIXES = {".wav", ".pt"}
INCLUDED_ROOTS = {"src", "tests", "scripts", "artifacts", "output"}


def _allowed(path: Path) -> bool:
    relative = path.relative_to(ROOT)
    if not relative.parts or relative.parts[0] not in INCLUDED_ROOTS:
        return False
    if relative.parts[0] == "artifacts" and relative.parts[1:2] not in {
        ("round24",),
        ("round25",),
    }:
        return False
    if relative.parts[0] == "output" and relative.parts[1:2] != ("round25-sample5",):
        return False
    if any(part in EXCLUDED_PARTS for part in relative.parts):
        return False
    if path.name in EXCLUDED_NAMES or path.suffix.lower() in EXCLUDED_SUFFIXES:
        return False
    return True


def _copy_e2e_ledger() -> None:
    source = JOB / "debug" / "crop_classifier_ledger.jsonl"
    if source.exists():
        shutil.copy2(source, ROUND25 / source.name)


def main() -> None:
    _copy_e2e_ledger()
    root_name = "media-flow-dev-v25"
    with zipfile.ZipFile(ZIP, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in ROOT.rglob("*"):
            if not path.is_file() or not _allowed(path):
                continue
            archive.write(path, Path(root_name) / path.relative_to(ROOT))
    digest = hashlib.sha256(ZIP.read_bytes()).hexdigest()
    report = {
        "zip": str(ZIP),
        "sha256": digest,
        "root": root_name,
        "blocked": True,
        "included": {
            "code": True,
            "round24": True,
            "round25": True,
            "round25_output": JOB.exists(),
        },
    }
    (ROUND25 / "package_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
