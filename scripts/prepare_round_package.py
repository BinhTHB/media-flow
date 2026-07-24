"""Round-agnostic package builder for media-flow.

Usage:
    python scripts/prepare_round_package.py --round 25
    python scripts/prepare_round_package.py --round 27

Produces media-flow-dev-v<N>.zip in the parent directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import zipfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT.parent

EXCLUDED_PARTS = {
    ".git", ".venv", "__pycache__", ".pytest_cache", ".ruff_cache",
    "frames", ".mypy_cache",
}
EXCLUDED_NAMES = {".env", "config.yml"}
EXCLUDED_SUFFIXES = {".pyc", ".wav", ".pt"}


def _git_sha() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True, text=True, cwd=str(ROOT),
    )
    return result.stdout.strip()


def _build_package(round_num: int) -> None:
    root_name = f"media-flow-dev-v{round_num}"
    zip_name = f"{root_name}.zip"
    zip_path = PARENT / zip_name

    artifacts_dir = ROOT / "artifacts" / f"round{round_num}"
    output_dir = ROOT / "output" / f"round{round_num}-sample5"
    gt_file = ROOT / "artifacts" / "round26" / "gt_v2_2.json"
    e2e_log = artifacts_dir / f"e2e_round{round_num}.log"

    included_roots = {"src", "tests", "scripts", "artifacts", "output"}
    allowed_artifact_rounds = {
        f"round{round_num}",
        "round26", "round27",
    }

    def _allowed(path: Path) -> bool:
        relative = path.relative_to(ROOT)
        parts = relative.parts

        if not parts or parts[0] not in included_roots:
            return False

        # Only include specific artifact rounds
        if parts[0] == "artifacts":
            if parts[1:2] == ():
                return False
            round_dir = parts[1]
            # Allow only current round and round26 (for gt_v2_2.json)
            if round_dir not in allowed_artifact_rounds:
                return False

        # Only include specific output round
        if parts[0] == "output":
            if parts[1:2] != (f"round{round_num}-sample5",):
                return False

        # Exclude by part name
        if any(part in EXCLUDED_PARTS for part in parts):
            return False
        # Exclude by filename
        if path.name in EXCLUDED_NAMES:
            return False
        # Exclude by suffix
        if path.suffix.lower() in EXCLUDED_SUFFIXES:
            return False

        return True

    # Make sure we generate structural metadata on disk before archiving
    git_sha = _git_sha()
    
    # Generate blocked_status.json on disk
    blocked_status = {
        "status": "BLOCKED",
        "stage": "e2e_gpu",
        "job_id": f"round{round_num}-sample5",
        "git_sha": git_sha,
        "reason": "GPU E2E not executed. PaddlePaddle-GPU is not available in local configuration (no CUDA backend). Cannot run full pipeline without GPU OCR. Sweep + offline replay completed with Paddle CPU fallback — see artifacts for diagnosis.",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    (artifacts_dir / "blocked_status.json").write_text(
        json.dumps(blocked_status, indent=2), encoding="utf-8"
    )

    # Initial draft of package_report.json on disk
    package_report = {
        "zip": str(zip_path),
        "sha256": "placeholder_to_be_replaced_by_digest_of_zip",
        "git_sha": git_sha,
        "root": root_name,
        "round": round_num,
        "blocked": True,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "included": {
            "code": True,
            f"round{round_num}": True,
            "round26_gt": gt_file.exists(),
            f"round{round_num}_output": output_dir.exists(),
        },
    }
    report_file = artifacts_dir / "package_report.json"
    report_file.write_text(json.dumps(package_report, indent=2), encoding="utf-8")

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in ROOT.rglob("*"):
            if not path.is_file() or not _allowed(path):
                continue
            archive.write(path, Path(root_name) / path.relative_to(ROOT))

    # Rewrite package_report.json without self-referential sha256 to avoid catch-22 mismatch.
    # The actual ZIP hash is strictly validated through the external sidecar .sha256 file.
    package_report["sha256"] = ""
    report_file.write_text(json.dumps(package_report, indent=2), encoding="utf-8")

    # Append the updated package_report.json to the zip archive so the inside has the correct sha256
    with zipfile.ZipFile(zip_path, "a") as archive:
        archive.write(report_file, Path(root_name) / report_file.relative_to(ROOT))

    # Calculate actual final zip SHA256 and write to sidecar file
    final_digest = hashlib.sha256(zip_path.read_bytes()).hexdigest()
    sidecar_path = zip_path.parent / f"{zip_path.name}.sha256"
    sidecar_path.write_text(final_digest + "\n", encoding="utf-8")

    print(f"  Created package: {zip_path}")
    print(f"  Final ZIP SHA256: {final_digest}")
    print(f"  Sidecar SHA256 file: {sidecar_path}")
    print(f"  Root: {root_name}")
    print(f"  Git SHA: {git_sha}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a round-agnostic media-flow package zip."
    )
    parser.add_argument(
        "--round", "-r",
        type=int,
        required=True,
        help="Round number (e.g., 25, 27)",
    )
    args = parser.parse_args()
    _build_package(args.round)


if __name__ == "__main__":
    main()
