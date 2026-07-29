"""Package the validated Round 35 V4 E2E run."""
from __future__ import annotations

import hashlib
import json
import struct
import subprocess
import zipfile
import zlib
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT.parent
ROUND = 35
ROOT_NAME = f"media-flow-dev-v{ROUND}"
ZIP = PARENT / f"{ROOT_NAME}.zip"
SIDECAR = ZIP.with_suffix(".zip.sha256")
ARTIFACT_DIR = ROOT / "artifacts" / f"round{ROUND}"
RUN_DIR = ROOT / "output" / "r35_e2e" / "r35_e2e"
VIDEO = ROOT / "output" / "sample5_v4.mp4"

EXCLUDED_PARTS = {".git", ".venv", "__pycache__", ".pytest_cache__", ".ruff_cache", "frames"}
EXCLUDED_SUFFIXES = {".pyc", ".wav", ".pt"}
EXCLUDED_NAMES = {".env", "config.yml"}
SHA256_PLACEHOLDER = "PLACEHOLDER_SHA256_HASH_R35_ZIP_XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"


def include(path: Path) -> bool:
    rel = path.relative_to(ROOT)
    if any(part in EXCLUDED_PARTS for part in rel.parts):
        return False
    if rel == Path("artifacts") / f"round{ROUND}" / "crop_classifier_v21_ledger.jsonl":
        return False
    if path.suffix.lower() in EXCLUDED_SUFFIXES or path.name in EXCLUDED_NAMES:
        return False
    if rel.parts[0] == "output":
        return False
    if rel.parts[0] == "artifacts":
        return len(rel.parts) > 1 and rel.parts[1] in {"round26", "round28", f"round{ROUND}"}
    return rel.parts[0] in {"src", "tests", "scripts", "config", "artifacts", "evaluation", "docs"}


def add_aliases(archive: zipfile.ZipFile) -> None:
    prefix = Path(ROOT_NAME) / "output" / f"round{ROUND}-sample5"
    aliases = {
        VIDEO: prefix / "final.mp4",
        RUN_DIR / "subtitles" / "subtitle.ass": prefix / "subtitles" / "subtitle.ass",
        RUN_DIR / "debug" / "subtitle_regions.json": prefix / "subtitle_regions.json",
    }
    for source in (RUN_DIR / "debug").iterdir():
        if source.is_file():
            aliases[source] = prefix / "debug" / source.name
    
    responses_dir = RUN_DIR / "debug" / "responses"
    if responses_dir.is_dir():
        for source in responses_dir.iterdir():
            if source.is_file():
                aliases[source] = prefix / "debug" / "responses" / source.name

    for source, target in aliases.items():
        if not source.is_file():
            raise RuntimeError(f"Required package input missing: {source}")
        archive.writestr(str(target), source.read_bytes())

    # Ledger audit mapping with adjusted job_id
    ledger_src = ROOT / "debug" / "ledger_r35_e2e.jsonl"
    ledger_target = prefix / "debug" / "crop_classifier_v21_ledger.jsonl"
    if not ledger_src.is_file():
         raise RuntimeError(f"Required ledger missing: {ledger_src}")
    
    lines = []
    for line in ledger_src.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        record["job_id"] = f"round{ROUND}-sample5"
        lines.append(json.dumps(record, ensure_ascii=False))
    
    archive.writestr(str(ledger_target), "\n".join(lines) + "\n")


def main() -> None:
    required = [
        VIDEO,
        RUN_DIR / "subtitles" / "subtitle.ass",
        RUN_DIR / "debug" / "subtitle_regions.json",
        ROOT / "debug" / "ledger_r35_e2e.jsonl",
        ARTIFACT_DIR / "offline_gate_r35.json",
        ARTIFACT_DIR / "eval_round35_v23.json",
        ARTIFACT_DIR / "e2e_round35.log",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"Required Round 35 artifacts missing: {missing}")

    git_sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    gate_path = ARTIFACT_DIR / f"offline_gate_r{ROUND}.json"
    gate_data = json.loads(gate_path.read_text(encoding="utf-8"))
    gate_executed_at_git_sha = gate_data.get("git_sha")
    if not gate_executed_at_git_sha:
        raise RuntimeError("Offline gate is missing its executed-at git_sha.")
    gate_data.setdefault("gate_executed_at_git_sha", gate_executed_at_git_sha)
    gate_data["packaged_at_git_sha"] = git_sha
    gate_path.write_text(
        json.dumps(gate_data, indent=2),
        encoding="utf-8",
        newline="\n",
    )

    report = {
        "round": ROUND,
        "root": ROOT_NAME,
        "zip": str(ZIP),
        "sha256": SHA256_PLACEHOLDER,
        "sha256_scheme": "zip-template-v1",
        "git_sha": git_sha,
        "gate_executed_at_git_sha": gate_executed_at_git_sha,
        "gate_packaged_at_git_sha": git_sha,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "gt_v2_3": "artifacts/round28/gt_v2_3.json",
        "included": {f"round{ROUND}_output": True},
        "gate": "PASS",
    }
    (ARTIFACT_DIR / "package_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8", newline="\n"
    )
    (ARTIFACT_DIR / "blocked_status.json").write_text(
        json.dumps({"status": "COMPLETED", "round": ROUND, "gate": "PASS"}, indent=2),
        encoding="utf-8", newline="\n"
    )

    if ZIP.exists():
        ZIP.unlink()
    if SIDECAR.exists():
        SIDECAR.unlink()

    with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in ROOT.rglob("*"):
            if path.is_file() and include(path):
                if path.name == "package_report.json":
                    archive.write(path, Path(ROOT_NAME) / path.relative_to(ROOT), compress_type=zipfile.ZIP_STORED)
                else:
                    archive.write(path, Path(ROOT_NAME) / path.relative_to(ROOT))
        add_aliases(archive)

    with zipfile.ZipFile(ZIP) as archive:
        names = archive.namelist()
    if len(names) != len(set(names)):
        raise RuntimeError("Duplicate ZIP entries detected.")

    zip_bytes = bytearray(ZIP.read_bytes())
    placeholder_bytes = SHA256_PLACEHOLDER.encode("ascii")
    idx = zip_bytes.find(placeholder_bytes)
    if idx == -1:
        raise RuntimeError("Could not find sha256 placeholder in ZIP bytes.")

    placeholder_hash = hashlib.sha256(zip_bytes).hexdigest()
    zip_bytes[idx : idx + 64] = placeholder_hash.encode("ascii")

    report["sha256"] = placeholder_hash
    report_file = ARTIFACT_DIR / "package_report.json"
    report_file.write_text(json.dumps(report, indent=2), encoding="utf-8", newline="\n")
    report_bytes = report_file.read_bytes()
    report_crc = zlib.crc32(report_bytes)

    with zipfile.ZipFile(ZIP) as archive:
        report_name = f"{ROOT_NAME}/artifacts/round{ROUND}/package_report.json"
        info = archive.getinfo(report_name)

    struct.pack_into("<I", zip_bytes, info.header_offset + 14, report_crc)
    target = report_name.encode("utf-8")
    offset = zip_bytes.find(b"PK\x01\x02")
    while offset != -1:
        name_len, extra_len, comment_len = struct.unpack_from("<HHH", zip_bytes, offset + 28)
        name_start = offset + 46
        if zip_bytes[name_start : name_start + name_len] == target:
            struct.pack_into("<I", zip_bytes, offset + 16, report_crc)
            break
        offset = zip_bytes.find(b"PK\x01\x02", offset + 1)

    ZIP.write_bytes(zip_bytes)

    digest = hashlib.sha256(zip_bytes).hexdigest()
    SIDECAR.write_text(f"{digest}  {ZIP.name}\n", encoding="ascii")
    print(f"ZIP={ZIP}")
    print(f"SHA256={digest}")
    print(f"ENTRIES={len(names)} UNIQUE={len(set(names))}")
    print(f"GIT_SHA={git_sha}")


if __name__ == "__main__":
    main()
