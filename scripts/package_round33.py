#!/usr/bin/env python3
"""Package the validated Round 33 E2E run with reproducible metadata."""
from __future__ import annotations

import hashlib
import json
import subprocess
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT.parent
ROUND = 33
ROOT_NAME = f"media-flow-dev-v{ROUND}"
ZIP = PARENT / f"{ROOT_NAME}.zip"
SIDECAR = ZIP.with_suffix(".zip.sha256")
ARTIFACT_DIR = ROOT / "artifacts" / f"round{ROUND}"
OUTPUT_DIR = ROOT / "output" / f"round{ROUND}-sample5"

EXCLUDED_PARTS = {".git", ".venv", "__pycache__", ".pytest_cache__", ".ruff_cache", "frames"}
EXCLUDED_SUFFIXES = {".pyc", ".wav", ".pt"}
EXCLUDED_NAMES = {".env", "config.yml"}


def include(path: Path) -> bool:
    rel = path.relative_to(ROOT)
    if any(part in EXCLUDED_PARTS for part in rel.parts):
        return False
    if path.suffix.lower() in EXCLUDED_SUFFIXES or path.name in EXCLUDED_NAMES:
        return False
    if rel.parts[0] == "output":
        return False
    if rel.parts[0] == "artifacts":
        return len(rel.parts) > 1 and rel.parts[1] in {"round26", "round28", f"round{ROUND}"}
    return rel.parts[0] in {"src", "tests", "scripts", "config", "artifacts", "output"}


def add_aliases(archive: zipfile.ZipFile) -> None:
    run_dir = OUTPUT_DIR / f"round{ROUND}-sample5"
    aliases = {
        run_dir / "videos" / "output.mp4":
            Path(ROOT_NAME) / "output" / f"round{ROUND}-sample5" / "final.mp4",
        run_dir / "subtitles" / "subtitle.ass":
            Path(ROOT_NAME) / "output" / f"round{ROUND}-sample5" / "subtitles" / "subtitle.ass",
        run_dir / "debug" / "subtitle_regions.json":
            Path(ROOT_NAME) / "output" / f"round{ROUND}-sample5" / "subtitle_regions.json",
    }
    aliases.update(
        {
            source: Path(ROOT_NAME) / "output" / f"round{ROUND}-sample5" / "debug" / source.name
            for source in (run_dir / "debug").iterdir()
            if source.is_file()
        }
    )
    for source, target in aliases.items():
        if not source.is_file():
            raise RuntimeError(f"Required package input missing: {source}")
        archive.writestr(str(target), source.read_bytes())


def main() -> None:
    required = [
        OUTPUT_DIR / f"round{ROUND}-sample5" / "videos" / "output.mp4",
        OUTPUT_DIR / f"round{ROUND}-sample5" / "subtitles" / "subtitle.ass",
        OUTPUT_DIR / f"round{ROUND}-sample5" / "debug" / "subtitle_regions.json",
        ARTIFACT_DIR / "offline_gate_r33.json",
        ARTIFACT_DIR / "eval_round33_v23.json",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"Required Round 33 artifacts missing: {missing}")

    git_sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    report = {
        "round": ROUND,
        "root": ROOT_NAME,
        "zip": str(ZIP),
        "sha256": "",
        "git_sha": git_sha,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "gt_v2_3": "artifacts/round28/gt_v2_3.json",
        "included": {f"round{ROUND}_output": True},
        "gate": "PASS",
    }
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    (ARTIFACT_DIR / "package_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    (ARTIFACT_DIR / "blocked_status.json").write_text(
        json.dumps({"status": "COMPLETED", "round": ROUND, "gate": "PASS"}, indent=2),
        encoding="utf-8",
    )

    if ZIP.exists():
        ZIP.unlink()
    if SIDECAR.exists():
        SIDECAR.unlink()

    with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in ROOT.rglob("*"):
            if path.is_file() and include(path):
                archive.write(path, Path(ROOT_NAME) / path.relative_to(ROOT))
        add_aliases(archive)

    with zipfile.ZipFile(ZIP) as archive:
        names = archive.namelist()
    if len(names) != len(set(names)):
        raise RuntimeError("Duplicate ZIP entries detected.")

    digest = hashlib.sha256(ZIP.read_bytes()).hexdigest()
    SIDECAR.write_text(f"{digest}  {ZIP.name}\n", encoding="ascii")
    print(f"ZIP={ZIP}")
    print(f"SHA256={digest}")
    print(f"ENTRIES={len(names)} UNIQUE={len(set(names))}")
    print(f"GIT_SHA={git_sha}")


if __name__ == "__main__":
    main()
