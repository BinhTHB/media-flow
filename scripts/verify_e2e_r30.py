#!/usr/bin/env python3
"""Verify Round 30 E2E output; exits nonzero on any required self-check failure."""
from __future__ import annotations
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
VIDEO = ROOT / "output" / "round30-sample5" / "final.mp4"
ASS = ROOT / "output" / "round30-sample5" / "round30-sample5" / "subtitles" / "subtitle.ass"
OUTPUT = ROOT / "artifacts" / "round30" / "e2e_self_check.json"


def main() -> None:
    print("COMMAND: .\\.venv\\Scripts\\python.exe scripts\\verify_e2e_r30.py")
    print("TIMESTAMP_UTC:", datetime.now(timezone.utc).isoformat())
    print("NVIDIA-SMI:")
    smi = subprocess.run(["nvidia-smi"], capture_output=True, text=True, check=False)
    print(smi.stdout, end="")
    if smi.stderr:
        print(smi.stderr, file=sys.stderr, end="")

    probe = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=codec_name,width,height,pix_fmt",
            "-of", "json", str(VIDEO),
        ],
        capture_output=True, text=True, check=False,
    )
    probe_data = json.loads(probe.stdout) if probe.returncode == 0 else {}
    stream = (probe_data.get("streams") or [{}])[0]

    ass_text = ASS.read_text(encoding="utf-8") if ASS.exists() else ""
    provenance_count = ass_text.count("provenance=classification:T003")
    mask_count = ass_text.count(r"\p1")

    cap = cv2.VideoCapture(str(VIDEO))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 10)  # T003 ~= 0.35 s at 30 fps
    ok, frame = cap.read()
    cap.release()
    dark_ratio = (
        float(np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) < 30))
        if ok else None
    )

    checks = {
        "ass_has_t003_provenance": provenance_count > 0,
        "ass_mask_count": mask_count,
        "h264_720x1280": (
            stream.get("codec_name") == "h264"
            and stream.get("width") == 720
            and stream.get("height") == 1280
        ),
        "t003_frame_read": ok,
        "t003_dark_ratio": dark_ratio,
        "t003_dark_ratio_high": dark_ratio is not None and dark_ratio >= 0.90,
    }
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "video": str(VIDEO),
        "ass": str(ASS),
        "ffprobe_stream": stream,
        "checks": checks,
        "status": "PASS" if all(
            value is True for value in checks.values() if isinstance(value, bool)
        ) else "FAIL",
    }
    OUTPUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("SELF_CHECK:", json.dumps(payload, indent=2))
    raise SystemExit(0 if payload["status"] == "PASS" else 2)


if __name__ == "__main__":
    main()
