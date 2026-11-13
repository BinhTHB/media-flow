"""Local multi-key Zernio config. One row is one API key aimed at one platform.

The file lives under secrets/ and is gitignored. List views return a masked
key. The full key stays on disk for the publish call.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

from src.infrastructure.publishing.zernio import ZERNIO_PLATFORMS


class ZernioAccountStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def list_public(self) -> list[dict[str, str]]:
        return [_public(row) for row in self._load()]

    def add(self, *, label: str, platform: str, api_key: str) -> dict[str, str]:
        name = label.strip() or platform
        target = platform.strip().lower()
        key = api_key.strip()
        if target not in ZERNIO_PLATFORMS:
            raise ValueError(f"Nền tảng không hỗ trợ: {platform}")
        if len(key) < 8:
            raise ValueError("API key quá ngắn.")
        rows = self._load()
        row = {
            "id": f"zernio_{uuid.uuid4().hex[:8]}",
            "label": name,
            "platform": target,
            "api_key": key,
        }
        rows.append(row)
        self._save(rows)
        return _public(row)

    def remove(self, account_id: str) -> bool:
        rows = self._load()
        kept = [row for row in rows if row["id"] != account_id]
        if len(kept) == len(rows):
            return False
        self._save(kept)
        return True

    def get(self, account_id: str) -> dict[str, str] | None:
        for row in self._load():
            if row["id"] == account_id:
                return row
        return None

    def _load(self) -> list[dict[str, str]]:
        if not self.path.is_file():
            return []
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return []
        rows = payload.get("accounts") if isinstance(payload, dict) else None
        if not isinstance(rows, list):
            return []
        clean: list[dict[str, str]] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            account_id = str(row.get("id") or "").strip()
            platform = str(row.get("platform") or "").strip().lower()
            key = str(row.get("api_key") or "").strip()
            if not account_id or platform not in ZERNIO_PLATFORMS or not key:
                continue
            clean.append({
                "id": account_id,
                "label": str(row.get("label") or platform),
                "platform": platform,
                "api_key": key,
            })
        return clean

    def _save(self, rows: list[dict[str, Any]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"accounts": rows}
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(self.path)


def _public(row: dict[str, str]) -> dict[str, str]:
    key = row.get("api_key") or ""
    masked = f"{'•' * 8}{key[-4:]}" if len(key) >= 4 else "••••"
    return {
        "id": row["id"],
        "label": row["label"],
        "platform": row["platform"],
        "api_key_masked": masked,
    }
