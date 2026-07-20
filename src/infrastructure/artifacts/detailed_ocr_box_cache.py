import json
import os
from typing import Any

from src.shared.exceptions import ProviderError


class DetailedOCRBoxCacheWriter:
    REQUIRED_FRAME_FIELDS = {
        "frame_timestamp", "frame_width", "frame_height", "source_image_path",
        "provider", "device", "coordinate_space", "boxes",
    }

    def write(self, path: str, records: list[dict[str, Any]]) -> None:
        for record in records:
            missing = self.REQUIRED_FRAME_FIELDS - record.keys()
            if missing:
                raise ProviderError(f"Detailed OCR frame record missing: {sorted(missing)}")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump({"schema_version": 1, "frames": records}, handle, indent=2, ensure_ascii=False)


class DetailedOCRBoxCacheReader:
    def read(self, path: str) -> list[dict[str, Any]]:
        try:
            with open(path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            raise ProviderError(f"Failed to read detailed OCR box cache: {path}") from exc
        if payload.get("schema_version") != 1 or not isinstance(payload.get("frames"), list):
            raise ProviderError("Unsupported detailed OCR box cache schema.")
        return payload["frames"]
