"""Verify a media-flow round package zip with content-checks.

Usage:
    python scripts/verify_package.py --zip ../media-flow-dev-v28.zip --round 28

Checks each required artifact is present, inspects content, and prints PASS/FAIL per check.
Exits with code 1 if ANY check fails.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path


def _git_sha() -> str:
    """Return current HEAD SHA from the repo containing this script."""
    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True, text=True, cwd=str(repo_root),
    )
    return result.stdout.strip()


def _check(label: str, ok: bool) -> bool:
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {label}")
    return ok


def _has_blocked_file(names: list[str], suffix: str) -> bool:
    """Return True if any name ends with the given suffix."""
    return any(name.lower().endswith(suffix) for name in names)


def check_video_codec_h264(video_path: Path) -> bool:
    """Check if the video is H.264 codec using ffprobe."""
    try:
        cmd = [
            "ffprobe",
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=codec_name",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(video_path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        codec = result.stdout.strip().lower()
        print(f"        Detected video codec: {repr(codec)}")
        return codec == "h264"
    except Exception as e:
        print(f"        FFprobe error checking codec: {e}")
        return False


def verify(zip_path: Path, round_num: int) -> int:
    """Run all checks. Returns 0 if all pass, 1 if any fail."""
    all_pass = True

    if not zip_path.exists():
        print(f"  [FAIL] Zip file does not exist: {zip_path}")
        return 1

    root_name = f"media-flow-dev-v{round_num}"
    artifacts_prefix = f"{root_name}/artifacts/round{round_num}/"
    output_prefix = f"{root_name}/output/round{round_num}-sample5/"
    gt_path = f"{root_name}/artifacts/round26/gt_v2_2.json"
    e2e_log_path = f"{artifacts_prefix}e2e_round{round_num}.log"
    package_report_path = f"{artifacts_prefix}package_report.json"
    blocked_status_path = f"{artifacts_prefix}blocked_status.json"

    # Calculate actual zip SHA256
    sha256 = hashlib.sha256()
    with open(zip_path, "rb") as f:
        while chunk := f.read(8192):
            sha256.update(chunk)
    actual_zip_hash = sha256.hexdigest()
    print(f"  [INFO] Zip SHA256 checksum: {actual_zip_hash}")

    # Check matching sidecar .sha256 file
    sidecar_path = zip_path.parent / f"{zip_path.name}.sha256"
    sidecar_valid = False
    if sidecar_path.is_file():
        try:
            sidecar_content = sidecar_path.read_text("utf-8").strip()
            # extract first 64 chars matching sha256
            match = re.search(r"[a-fA-F0-9]{64}", sidecar_content)
            if match:
                sidecar_hash = match.group(0).lower()
                sidecar_valid = sidecar_hash == actual_zip_hash
                r = _check(f"Sidecar SHA256 matches ZIP hash ({actual_zip_hash})", sidecar_valid)
            else:
                r = _check("Sidecar SHA256 matches ZIP hash", False)
                print(f"        Invalid sidecar content pattern: {sidecar_content}")
            all_pass = all_pass and r
        except Exception as e:
            all_pass = _check(f"Reading sidecar .sha256 file failed: {e}", False) and all_pass
    else:
        # Check if another common checksum file name exists
        r = _check(f"Sidecar checksum file present at: {sidecar_path}", False)
        all_pass = all_pass and r

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        with zipfile.ZipFile(zip_path, "r") as zf:
            names = zf.namelist()

            # Determine actual presence of output in ZIP
            final_mp4_name = f"{output_prefix}videos/final.mp4"
            if final_mp4_name not in names:
                final_mp4_name = f"{output_prefix}final.mp4"
            
            actual_has_output = final_mp4_name in names
            print(f"  [INFO] ZIP output presence: {actual_has_output} (final.mp4 in zip: {final_mp4_name in names})")

            # --- Structural checks ---
            root_present = any(n.startswith(root_name + "/") or n == root_name + "/" for n in names)
            r = _check(f"Root folder '{root_name}/' present", root_present)
            all_pass = all_pass and r

            artifacts_present = any(n.startswith(artifacts_prefix) for n in names)
            r = _check(f"artifacts/round{round_num}/ present", artifacts_present)
            all_pass = all_pass and r

            gt_present = gt_path in names
            r = _check("artifacts/round26/gt_v2_2.json present", gt_present)
            all_pass = all_pass and r

            # If output is physically present, perform structural checks & content validation on output
            if actual_has_output:
                output_present = any(n.startswith(output_prefix) for n in names)
                r = _check(f"output/round{round_num}-sample5/ prefix present", output_present)
                all_pass = all_pass and r

                final_mp4_present = final_mp4_name in names
                r = _check(f"output/round{round_num}-sample5/final.mp4 present", final_mp4_present)
                all_pass = all_pass and r

                if final_mp4_present:
                    # Extract and ffprobe it!
                    temp_video = tmp_path / "final.mp4"
                    temp_video.write_bytes(zf.read(final_mp4_name))
                    is_h264 = check_video_codec_h264(temp_video)
                    r = _check(f"final.mp4 codec check (must be H.264)", is_h264)
                    all_pass = all_pass and r

                subtitle_json = (
                    f"{output_prefix}subtitle_regions.json" in names
                    or any(n.startswith(output_prefix) and n.endswith("subtitle_regions.json") for n in names)
                )
                r = _check(f"output/round{round_num}-sample5/subtitle_regions.json present", subtitle_json)
                all_pass = all_pass and r

                debug_present = any(n.startswith(f"{output_prefix}debug/") for n in names)
                r = _check(f"output/round{round_num}-sample5/debug/ present", debug_present)
                all_pass = all_pass and r

                if debug_present:
                    # Ledger audit
                    ledger_name = f"{output_prefix}debug/crop_classifier_v21_ledger.jsonl"
                    if ledger_name in names:
                        try:
                            ledger_bytes = zf.read(ledger_name)
                            lines = ledger_bytes.decode("utf-8").splitlines()
                            valid_ledger = True
                            checked_lines = 0
                            for line in lines:
                                if not line.strip():
                                    continue
                                record = json.loads(line)
                                job_id = record.get("job_id", "")
                                expected_job = f"round{round_num}-sample5"
                                if job_id != expected_job:
                                    print(f"        Ledger audit failed: entry job_id is '{job_id}' (expected '{expected_job}')")
                                    valid_ledger = False
                                    break
                                checked_lines += 1
                            r = _check(f"Ledger audit: all {checked_lines} entries belong to 'round{round_num}-sample5'", valid_ledger and checked_lines > 0)
                            all_pass = all_pass and r
                        except Exception as e:
                            print(f"        Ledger audit error: {e}")
                            all_pass = False
                    else:
                        r = _check("crop_classifier_v21_ledger.jsonl present for audit", False)
                        all_pass = all_pass and r
            else:
                # If output is physically absent, make sure we do NOT have output directories
                output_dirs_in_zip = any(n.startswith(output_prefix) for n in names)
                r = _check("Output folder absent (matching BLOCKED/No-Output scenario)", not output_dirs_in_zip)
                all_pass = all_pass and r

            # --- E2E log: present, size check, contents verification ---
            e2e_present = e2e_log_path in names
            r = _check(f"e2e_round{round_num}.log present", e2e_present)
            all_pass = all_pass and r
            if e2e_present:
                try:
                    e2e_bytes = zf.read(e2e_log_path)
                    e2e_text = e2e_bytes.decode("utf-8")
                    size_kb = len(e2e_bytes) / 1024
                    
                    # 1. Log file must be non-trivial (>1KB) and not just short prose
                    log_size_ok = size_kb > 1.0
                    r1 = _check(f"e2e log size check ({size_kb:.2f} KB > 1.0 KB)", log_size_ok)
                    
                    # 2. Check for real run traces (timestamp patterns, logs, etc.)
                    has_trace_markers = (
                        "Application" in e2e_text or 
                        "INFO" in e2e_text or 
                        "ERROR" in e2e_text or 
                        "WARN" in e2e_text or
                        "DEBUG" in e2e_text or 
                        "Command-line" in e2e_text or
                        "duration" in e2e_text or
                        re.search(r"\d{4}-\d{2}-\d{2}", e2e_text) is not None or
                        re.search(r"\d{2}:\d{2}:\d{2}", e2e_text) is not None
                    )
                    r2 = _check("e2e log contains execution signatures and timestamps", has_trace_markers)
                    all_pass = all_pass and r1 and r2
                except Exception as exc:
                    print(f"  [FAIL] e2e log inspection error: {exc}")
                    all_pass = False

            # --- package_report.json verification ---
            report_present = package_report_path in names
            r = _check("package_report.json present", report_present)
            all_pass = all_pass and r

            if report_present:
                try:
                    # Find all duplicate package_report.json in ZIP (Python lists zip contents in preservation order, so read the LAST entry)
                    report_entry = None
                    for n in names:
                        if n == package_report_path:
                            report_entry = n
                    
                    report_data = json.loads(zf.read(report_entry).decode("utf-8"))
                    
                    # 1. Git SHA matches HEAD
                    report_sha = report_data.get("git_sha", "")
                    current_sha = _git_sha()
                    sha_match = report_sha == current_sha
                    r1 = _check(f"package_report.json git_sha matches HEAD ({current_sha[:12]}…)", sha_match)
                    if not sha_match:
                        print(f"        report SHA:  {report_sha}")
                        print(f"        current SHA: {current_sha}")

                    gate_path = f"{artifacts_prefix}offline_gate_r{round_num}.json"
                    gate_match = False
                    if gate_path in names:
                        gate_data = json.loads(zf.read(gate_path).decode("utf-8"))
                        gate_match = gate_data.get("git_sha", "") == report_sha
                    r_gate = _check(
                        "offline gate git_sha matches package_report git_sha",
                        gate_match,
                    )
                    
                    # 2. Output verification matches round output flag
                    round_output_flag = report_data.get("included", {}).get(f"round{round_num}_output", False)
                    flags_match = round_output_flag == actual_has_output
                    r2 = _check(f"package_report round{round_num}_output field ({round_output_flag}) matches physical output's presence ({actual_has_output})", flags_match)
                    
                    # 3. Check correct ZIP SHA256 inside report if reported and non-empty
                    reported_zip_hash = report_data.get("sha256", "")
                    if reported_zip_hash:
                        hash_matches = reported_zip_hash.lower() == actual_zip_hash
                        r3 = _check(f"package_report ZIP hash matches actual computed hash", hash_matches)
                        if not hash_matches:
                            print(f"        reported inside ZIP: {reported_zip_hash}")
                            print(f"        actual ZIP hash    : {actual_zip_hash}")
                    else:
                        print("        [INFO] self-referential ZIP hash in package_report.json is empty (delegated to sidecar .sha256 validation)")
                        r3 = True
                        
                    all_pass = all_pass and r1 and r_gate and r2 and r3
                except Exception as exc:
                    print(f"  [FAIL] package_report.json check error: {exc}")
                    all_pass = False

            # --- blocked_status.json present ---
            blocked_present = blocked_status_path in names
            r = _check("blocked_status.json present", blocked_present)
            all_pass = all_pass and r

            if blocked_present:
                try:
                    blocked_data = json.loads(zf.read(blocked_status_path).decode("utf-8"))
                    status = blocked_data.get("status", "")
                    expected_status = "BLOCKED" if not actual_has_output else "COMPLETED" # or pending if not run
                    
                    if not actual_has_output:
                        r = _check(f"blocked_status.json status is '{status}' (expected BLOCKED)", status == "BLOCKED")
                    else:
                        r = True
                    all_pass = all_pass and r
                except Exception as e:
                    print(f"  [FAIL] blocked_status.json check error: {e}")
                    all_pass = False

            # --- Excluded items must NOT be present ---
            pyc_present = _has_blocked_file(names, ".pyc")
            r = _check(".pyc files NOT present", not pyc_present)
            if pyc_present:
                pyc_files = [n for n in names if n.lower().endswith(".pyc")]
                print(f"        Found .pyc files: {pyc_files[:5]}")
            all_pass = all_pass and r

            config_yml_present = any(n.endswith("/config.yml") or n == "config.yml" for n in names)
            r = _check("config.yml NOT present", not config_yml_present)
            all_pass = all_pass and r

            pt_present = _has_blocked_file(names, ".pt")
            r = _check(".pt model files NOT present", not pt_present)
            if pt_present:
                pt_files = [n for n in names if n.lower().endswith(".pt")]
                print(f"        Found .pt files: {pt_files[:5]}")
            all_pass = all_pass and r

            wav_present = _has_blocked_file(names, ".wav")
            r = _check(".wav files NOT present", not wav_present)
            if wav_present:
                wav_files = [n for n in names if n.lower().endswith(".wav")]
                print(f"        Found .wav files: {wav_files[:5]}")
            all_pass = all_pass and r

            env_present = any(n.endswith("/.env") or n == ".env" for n in names)
            r = _check(".env files NOT present", not env_present)
            all_pass = all_pass and r

    return 0 if all_pass else 1


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify a media-flow round package zip with content-checks."
    )
    parser.add_argument("--zip", required=True, type=Path, help="Path to the zip file")
    parser.add_argument("--round", "-r", required=True, type=int, help="Round number (e.g., 28)")
    args = parser.parse_args()

    print(f"\nVerifying: {args.zip}")
    print(f"Round: {args.round}")
    print("-" * 60)
    exit_code = verify(args.zip, args.round)
    print("-" * 60)
    if exit_code == 0:
        print("Result: ALL CHECKS PASSED ✓")
    else:
        print("Result: ONE OR MORE CHECKS FAILED ✗")
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
