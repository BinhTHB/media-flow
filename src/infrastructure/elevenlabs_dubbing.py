"""ElevenLabs Dubbing v2 project API adapter.

Creates an Automatic Dubbing v2 project from local media, queues one target
language, waits for completion, and downloads the lossless dubbed audio.

Key-pool safety: project creation may consume dubbing quota. We rotate keys only
when ElevenLabs explicitly rejects a request before returning a project id
(401/402/403/429). We do not blindly retry ambiguous network/5xx failures.
"""
from __future__ import annotations

import json
import mimetypes
import os
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any


API_BASE = "https://api.elevenlabs.io/v1/dubbing"
_ROTATABLE_CREATE_STATUS = {401, 402, 403, 429}


def configured_elevenlabs_keys() -> list[str]:
    values: list[str] = []
    for name in ("ELEVENLABS_API_KEYS", "ELEVENLABS_API_KEY"):
        raw = os.getenv(name, "")
        if raw:
            values.extend(part.strip() for part in raw.replace(";", ",").split(","))
    for name, value in sorted(os.environ.items()):
        if name.startswith("ELEVENLABS_API_KEY_") and value.strip():
            values.append(value.strip())
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result


@dataclass(frozen=True)
class DubbingV2Result:
    project_id: str
    language_id: str
    target_language: str
    output_audio_path: str
    project_warnings: list[dict[str, Any]]
    language_warnings: list[dict[str, Any]]
    key_index: int


class ElevenLabsDubbingV2Adapter:
    def __init__(
        self,
        api_keys: list[str] | None = None,
        timeout_seconds: float = 120.0,
        poll_interval_seconds: float = 2.0,
        max_wait_seconds: float = 180.0,
    ) -> None:
        self.api_keys = api_keys or configured_elevenlabs_keys()
        self.timeout_seconds = max(10.0, float(timeout_seconds))
        self.poll_interval_seconds = max(0.5, float(poll_interval_seconds))
        self.max_wait_seconds = max(10.0, float(max_wait_seconds))

    def dub_file(
        self,
        source_path: str,
        *,
        target_language: str = "vi",
        source_language: str | None = None,
        output_audio_path: str,
        reference: str = "MediaFlow benchmark",
        keyterms: list[str] | None = None,
    ) -> DubbingV2Result:
        if not self.api_keys:
            raise RuntimeError("ELEVENLABS_API_KEY(S) is required for ElevenLabs Dubbing v2.")
        source = Path(source_path).resolve()
        if not source.is_file():
            raise FileNotFoundError(f"Dubbing source not found: {source}")

        create_payload: dict[str, Any] | None = None
        selected_key = ""
        selected_index = -1
        errors: list[str] = []
        for index, key in enumerate(self.api_keys):
            try:
                create_payload = self._create_project(
                    key,
                    source,
                    target_language=target_language,
                    source_language=source_language,
                    reference=reference,
                    keyterms=keyterms,
                )
                selected_key = key
                selected_index = index
                break
            except _RotatableCreateError as exc:
                errors.append(f"key#{index + 1}: HTTP {exc.status}: {exc.detail[:180]}")
                continue

        if not create_payload or not selected_key:
            raise RuntimeError("ElevenLabs Dubbing v2 key pool exhausted. " + " | ".join(errors))

        project_id = str(create_payload.get("project_id") or "")
        language_ids = create_payload.get("language_ids") or []
        if not project_id:
            raise RuntimeError("ElevenLabs Dubbing v2 create response did not include project_id.")

        project = self._wait_project_ready(selected_key, project_id)
        if not language_ids:
            languages = self._list_languages(selected_key, project_id)
            language_ids = [item.get("language_id") for item in languages if item.get("target_language") == target_language]
        if not language_ids:
            language = self._create_language(selected_key, project_id, target_language)
            language_ids = [language.get("language_id")]
        language_id = str(language_ids[0] or "")
        if not language_id:
            raise RuntimeError("ElevenLabs Dubbing v2 did not return a language_id.")

        language = self._wait_language_completed(selected_key, project_id, language_id)
        outputs = language.get("outputs") or {}
        audio_url = str(outputs.get("lossless_audio") or "")
        if not audio_url:
            raise RuntimeError(f"Completed dubbing target has no lossless_audio URL: {language}")
        output = Path(output_audio_path).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        self._download(audio_url, output)

        return DubbingV2Result(
            project_id=project_id,
            language_id=language_id,
            target_language=target_language,
            output_audio_path=str(output),
            project_warnings=list(project.get("warnings") or []),
            language_warnings=list(language.get("warnings") or []),
            key_index=selected_index,
        )

    def get_source_transcript(self, project_id: str, *, key_index: int = 0) -> dict[str, Any]:
        key = self._key_at(key_index)
        return self._get_json(key, f"{API_BASE}/project/{project_id}/transcript")

    def get_target_transcript(
        self,
        project_id: str,
        language_id: str,
        *,
        key_index: int = 0,
    ) -> dict[str, Any]:
        key = self._key_at(key_index)
        return self._get_json(
            key,
            f"{API_BASE}/project/{project_id}/language/{language_id}/transcript",
        )

    def _key_at(self, key_index: int) -> str:
        if not self.api_keys:
            raise RuntimeError("ELEVENLABS_API_KEY(S) is required for ElevenLabs Dubbing v2.")
        if key_index < 0 or key_index >= len(self.api_keys):
            raise IndexError(f"ElevenLabs key index out of range: {key_index}")
        return self.api_keys[key_index]

    def _create_project(
        self,
        key: str,
        source: Path,
        *,
        target_language: str,
        source_language: str | None,
        reference: str,
        keyterms: list[str] | None,
    ) -> dict[str, Any]:
        fields: dict[str, str | list[str]] = {
            "reference": reference[:500],
            "model_id": "dubbing_v2",
            "target_language": target_language,
        }
        if source_language:
            fields["source_language"] = source_language
        if keyterms:
            cleaned = [term.strip() for term in keyterms if term and term.strip()]
            if cleaned:
                fields["keyterms"] = cleaned[:1000]
        body, content_type = self._multipart(fields, source)
        request = urllib.request.Request(
            f"{API_BASE}/project",
            data=body,
            method="POST",
            headers={"xi-api-key": key, "Content-Type": content_type, "Accept": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            if exc.code in _ROTATABLE_CREATE_STATUS:
                raise _RotatableCreateError(exc.code, detail) from exc
            raise RuntimeError(f"ElevenLabs Dubbing v2 create HTTP {exc.code}: {detail[:500]}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(
                "ElevenLabs Dubbing v2 create network error; not auto-retrying to avoid duplicate quota use: "
                f"{exc}"
            ) from exc

    def _wait_project_ready(self, key: str, project_id: str) -> dict[str, Any]:
        deadline = time.monotonic() + self.max_wait_seconds
        while True:
            project = self._get_json(key, f"{API_BASE}/project/{project_id}")
            status = str(project.get("status") or "")
            if status == "ready":
                return project
            if status == "failed":
                raise RuntimeError(f"ElevenLabs dubbing project failed: {project.get('error')}")
            if time.monotonic() >= deadline:
                raise TimeoutError(f"ElevenLabs dubbing project did not become ready: {project_id}")
            time.sleep(self.poll_interval_seconds)

    def _list_languages(self, key: str, project_id: str) -> list[dict[str, Any]]:
        payload = self._get_json(key, f"{API_BASE}/project/{project_id}/language?page_size=20")
        return list(payload.get("languages") or [])

    def _create_language(self, key: str, project_id: str, target_language: str) -> dict[str, Any]:
        return self._post_json(
            key,
            f"{API_BASE}/project/{project_id}/language",
            {"target_language": target_language},
        )

    def _wait_language_completed(self, key: str, project_id: str, language_id: str) -> dict[str, Any]:
        deadline = time.monotonic() + self.max_wait_seconds
        url = f"{API_BASE}/project/{project_id}/language/{language_id}"
        while True:
            language = self._get_json(key, url)
            status = str(language.get("status") or "")
            if status == "completed":
                return language
            if status == "failed":
                raise RuntimeError(f"ElevenLabs dubbing language failed: {language.get('error')}")
            if time.monotonic() >= deadline:
                raise TimeoutError(f"ElevenLabs dubbing language did not complete: {language_id}")
            time.sleep(self.poll_interval_seconds)

    def _get_json(self, key: str, url: str) -> dict[str, Any]:
        request = urllib.request.Request(url, headers={"xi-api-key": key, "Accept": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"ElevenLabs Dubbing v2 GET HTTP {exc.code}: {detail[:500]}") from exc

    def _post_json(self, key: str, url: str, payload: dict[str, Any]) -> dict[str, Any]:
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            method="POST",
            headers={"xi-api-key": key, "Content-Type": "application/json", "Accept": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"ElevenLabs Dubbing v2 POST HTTP {exc.code}: {detail[:500]}") from exc

    def _download(self, url: str, output: Path) -> None:
        request = urllib.request.Request(url, headers={"Accept": "audio/*,application/octet-stream"})
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                output.write_bytes(response.read())
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Failed to download ElevenLabs dubbed audio: {exc}") from exc

    @staticmethod
    def _multipart(fields: dict[str, str | list[str]], source: Path) -> tuple[bytes, str]:
        boundary = f"----MediaFlowElevenDubbing{uuid.uuid4().hex}"
        parts: list[bytes] = []
        for name, value in fields.items():
            values = value if isinstance(value, list) else [value]
            for item in values:
                parts.extend([
                    f"--{boundary}\r\n".encode(),
                    f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode(),
                    str(item).encode("utf-8"),
                    b"\r\n",
                ])
        mime = mimetypes.guess_type(source.name)[0] or "application/octet-stream"
        parts.extend([
            f"--{boundary}\r\n".encode(),
            f'Content-Disposition: form-data; name="file"; filename="{source.name}"\r\n'.encode(),
            f"Content-Type: {mime}\r\n\r\n".encode(),
            source.read_bytes(),
            b"\r\n",
            f"--{boundary}--\r\n".encode(),
        ])
        return b"".join(parts), f"multipart/form-data; boundary={boundary}"


class _RotatableCreateError(RuntimeError):
    def __init__(self, status: int, detail: str) -> None:
        super().__init__(detail)
        self.status = status
        self.detail = detail
