#!/usr/bin/env python3
"""Small dependency-free guard against committing obvious credentials."""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {".env", ".ini", ".json", ".md", ".py", ".toml", ".txt", ".yaml", ".yml"}
SKIP_PARTS = {".git", ".venv", "node_modules", "outputs", "input", "models", "__pycache__"}
PRIVATE_KEY = re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----")
ASSIGNMENT = re.compile(
    r"(?i)(?:api[_-]?key|access[_-]?token|client[_-]?secret|password)\s*[:=]\s*['\"]?([A-Za-z0-9_./+\-=]{20,})"
)
PLACEHOLDERS = ("example", "replace", "your_", "dummy", "changeme", "redacted")


def candidate_files() -> list[Path]:
    try:
        output = subprocess.check_output(
            ["git", "ls-files"], cwd=ROOT, text=True, encoding="utf-8", errors="replace"
        )
        paths = [ROOT / line for line in output.splitlines() if line]
    except (OSError, subprocess.CalledProcessError):
        paths = list(ROOT.rglob("*"))
    return [
        path for path in paths
        if path.is_file() and not SKIP_PARTS.intersection(path.relative_to(ROOT).parts)
        and (path.suffix.lower() in TEXT_SUFFIXES or path.name == ".env.example")
    ]


def scan() -> list[str]:
    findings: list[str] = []
    for path in candidate_files():
        try:
            text = path.read_text(encoding="utf-8-sig")
        except (OSError, UnicodeError):
            continue
        relative = path.relative_to(ROOT).as_posix()
        for number, line in enumerate(text.splitlines(), 1):
            if PRIVATE_KEY.search(line):
                findings.append(f"{relative}:{number}: private key marker")
            match = ASSIGNMENT.search(line)
            if match and not any(word in match.group(1).lower() for word in PLACEHOLDERS):
                findings.append(f"{relative}:{number}: possible credential assignment")
    return findings


def main() -> int:
    findings = scan()
    if findings:
        print("Secret scan failed:\n" + "\n".join(findings), file=sys.stderr)
        return 1
    print("Secret scan passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
