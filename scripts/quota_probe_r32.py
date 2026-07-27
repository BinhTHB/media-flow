#!/usr/bin/env python3
"""Probe: send 3 simple Gemini requests 25 seconds apart.
If all return 200, quota is RPM (not RPD) and gate can run today.
"""
from __future__ import annotations
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.infrastructure.gemini.client import GeminiClient
from src.infrastructure.gemini.contracts import GeminiRequest


def load_key() -> str:
    for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
        if line.strip().startswith("GEMINI_API_KEY="):
            return line.split("=", 1)[1].strip().strip("'\"").split(",")[0].strip()
    raise RuntimeError("GEMINI_API_KEY unavailable")


def main() -> None:
    client = GeminiClient(api_key=load_key(), model="gemini-3.5-flash-lite")

    for i in range(3):
        print(f"[PROBE {i+1}/3] Sending...")
        ts_start = time.monotonic()
        try:
            resp = client.generate_content(
                GeminiRequest(
                    contents=[{"parts": [{"text": "Reply with OK and nothing else."}]}],
                    model="gemini-3.5-flash-lite",
                    request_id=f"probe_r32_{i+1}",
                ),
                use_cache=False,
            )
            elapsed = time.monotonic() - ts_start
            print(f"[PROBE {i+1}/3] 200 OK  ({elapsed:.1f}s)")
            print(f"  response: {resp.text[:80]}")
            if i < 2:
                wait = 25
                print(f"  waiting {wait}s...")
                time.sleep(wait)
        except Exception as e:
            elapsed = time.monotonic() - ts_start
            err_text = str(e)
            print(f"[PROBE {i+1}/3] FAIL ({elapsed:.1f}s): {err_text[:200]}")
            sys.exit(1)

    print("\n[QUOTA] All 3 probes passed with 25s spacing. "
          "Conclusion: quota is RPM (per-minute), not RPD (per-day). "
          "Gate can run today with pacing.")


if __name__ == "__main__":
    main()
