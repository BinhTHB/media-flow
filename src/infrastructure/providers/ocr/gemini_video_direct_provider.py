from __future__ import annotations

import base64
import hashlib
import json
import mimetypes
import os
import re
import statistics
import subprocess
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence
from src.infrastructure.gemini.exceptions import GeminiRateLimitError, ProviderRateLimitError

from src.application.ports.ocr_provider import VideoSubtitleOCRProvider
from src.domain.entities.segment import Segment
from src.infrastructure.gemini.contracts import MODE_ROLE_PRIMARY, GeminiRequest
from src.infrastructure.gemini.exceptions import GeminiError
from src.infrastructure.gemini.runtime import GeminiRuntime
from src.infrastructure.gemini.structured_output import parse_structured
from src.shared.exceptions import ProviderError


_SUBTITLE_CLASSES = {"dialogue_subtitle", "narration_subtitle"}
_REPLACEMENT_SAFETY = {
    "SAFE_EQUIVALENT_OCCLUSION",
    "SAFE_MINOR_EXPANSION",
    "UNSAFE_INCREMENTAL_OCCLUSION",
    "REVIEW_REQUIRED",
}
_ACCEPTED_REPLACEMENT_SAFETY = {
    "SAFE_EQUIVALENT_OCCLUSION",
    "SAFE_MINOR_EXPANSION",
}
_ALL_CLASSES = _SUBTITLE_CLASSES | {
    "title_card", "scene_text", "watermark", "logo", "user_interface",
    "warning_overlay", "account_or_channel_label", "other", "review_required",
}


class GeminiDirectVideoOCRProvider(VideoSubtitleOCRProvider):
    """G.5 OCR adapter using direct video for V1-V3/V5 and Gemini images for V4."""

    def __init__(
        self,
        runtime: Optional[GeminiRuntime] = None,
        model: str = "gemini-3.1-flash-lite",
        api_key: Optional[str] = None,
        discovery_fps: float = 6,
        classification_fps: float = 6,
        segmentation_fps: float = 12,
        geometry_fps: float = 12,
        transcription_fps: float = 12,
        media_resolution: str = "high",
        clip_padding_seconds: float = 0.40,
        max_retry_per_stage: int = 1,
        prompt_version: str = "gemini_direct_video_g5_v1",
        schema_version: str = "gemini_direct_video_g5_v2",
        target_language: str = "vi",
        max_raw_candidates: int = 200,
        max_parseable_raw_candidates: int = 200,
        max_deduplicated_candidates: int = 10,
        max_accepted_subtitle_tracks: int = 10,
        max_events_per_track: int = 20,
        max_total_stage_requests: int = 100,
        max_counted_input_tokens_per_job: int = 500000,
        stage_timeout_seconds: float = 360,
        no_progress_timeout_seconds: float = 300,
        job_timeout_seconds: float = 600,
        max_quota_wait_seconds: float = 120,
        max_quota_retries_per_stage: int = 1,
        honor_retry_after: bool = True,
        fail_fast_on_daily_quota: bool = True,
        min_request_spacing_seconds: float = 5.0,
        direct_video_active_rpm: int = 15,
        quota_jitter_seconds: float = 0.5,
        unknown_429_backoff_seconds: float = 5.0,
        fps_certification_ttl_seconds: float = 86400.0,
        resume_enabled: bool = True,
        checkpoint_schema_version: str = "1.0.0",
        batch_size: int = 10,
        resume_manifest_dir: Optional[str] = None,
        fps_certification_path: Optional[str] = None,
        preflight_certification_path: Optional[str] = None,
        preflight_certification_ttl_seconds: float = 86400.0,
        preflight_credential_identity_hash: Optional[str] = None,
        stop_after_stage: Optional[str] = None,
        geometry_mode: str = "image",
    ) -> None:
        if model != "gemini-3.1-flash-lite":
            raise ValueError("G.5 direct-video OCR requires gemini-3.1-flash-lite.")
        if clip_padding_seconds < 0:
            raise ValueError("clip_padding_seconds must be non-negative.")
        if geometry_mode != "image":
            raise ValueError("G.5.11 requires image-level Gemini geometry.")
        self.runtime = runtime or GeminiRuntime.create()
        self.model = model
        self.api_key = api_key
        self.geometry_mode = geometry_mode
        self.fps = {
            "discovery": discovery_fps, "classification": classification_fps,
            "segmentation": segmentation_fps, "geometry": geometry_fps,
            "transcription": transcription_fps,
        }
        self.media_resolution = media_resolution
        self.clip_padding_seconds = clip_padding_seconds
        self.max_retry_per_stage = max_retry_per_stage
        self.prompt_version = prompt_version
        self.schema_version = schema_version
        self.target_language = target_language
        self.max_raw_candidates = max_raw_candidates
        self.max_parseable_raw_candidates = max_parseable_raw_candidates
        self.max_deduplicated_candidates = max_deduplicated_candidates
        self.max_accepted_subtitle_tracks = max_accepted_subtitle_tracks
        self.max_events_per_track = max_events_per_track
        self.max_total_stage_requests = max_total_stage_requests
        self.max_counted_input_tokens_per_job = max_counted_input_tokens_per_job
        self.stage_timeout_seconds = stage_timeout_seconds
        self.no_progress_timeout_seconds = no_progress_timeout_seconds
        self.job_timeout_seconds = job_timeout_seconds
        self.max_quota_wait_seconds = max_quota_wait_seconds
        self.max_quota_retries_per_stage = max_quota_retries_per_stage
        self.honor_retry_after = honor_retry_after
        self.fail_fast_on_daily_quota = fail_fast_on_daily_quota
        if direct_video_active_rpm <= 0:
            raise ValueError("direct_video_active_rpm must be positive.")
        self.direct_video_active_rpm = direct_video_active_rpm
        self.base_request_spacing_seconds = 60.0 / direct_video_active_rpm
        self.min_request_spacing_seconds = max(
            self.base_request_spacing_seconds,
            min_request_spacing_seconds,
        )
        self.quota_jitter_seconds = quota_jitter_seconds
        self.unknown_429_backoff_seconds = unknown_429_backoff_seconds
        self.fps_certification_ttl_seconds = fps_certification_ttl_seconds
        self.resume_enabled = resume_enabled
        self.checkpoint_schema_version = checkpoint_schema_version
        self.batch_size = batch_size
        self.resume_manifest_dir = Path(
            resume_manifest_dir or "output/gemini_direct_video_resume"
        )
        self.fps_certification_path = Path(
            fps_certification_path or "debug/g5_fps_certification.json"
        )
        self.preflight_certification_path = Path(
            preflight_certification_path or "debug/g5_authenticated_preflight.json"
        )
        self.preflight_certification_ttl_seconds = preflight_certification_ttl_seconds
        self.preflight_credential_identity_hash = preflight_credential_identity_hash
        if stop_after_stage not in {None, "V1"}:
            raise ValueError("stop_after_stage must be None or 'V1'.")
        self.stop_after_stage = stop_after_stage
        self.provenance: List[Dict[str, Any]] = []
        self.artifacts: Dict[str, Any] = {}
        self._render_eligibility: List[Dict[str, Any]] = []
        self._job_start_time: float = 0.0
        self._last_progress_time: float = 0.0
        self._resume_manifest_path: Optional[Path] = None

    # â”€â”€ public entry point â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def extract_subtitles(
        self, video_path: Path, source_language: str | None
    ) -> List[Segment]:
        if not video_path.is_file():
            raise ProviderError(f"Direct-video OCR input does not exist: '{video_path}'.")
        self._reset()
        self._job_start_time = time.monotonic()
        self._last_progress_time = self._job_start_time
        mime_type = mimetypes.guess_type(str(video_path))[0] or "video/mp4"
        if mime_type != "video/mp4":
            raise ProviderError("G.5 direct-video OCR currently requires video/mp4.")

        # --- read video metadata with FFprobe (allowed pre-inference) ---
        duration = self._probe_duration(str(video_path))
        if duration <= 0:
            raise ProviderError(f"Video duration must be positive, got {duration}.")
        self.artifacts["video_metadata"] = {
            "duration_seconds": duration,
            "source_fps": self._probe_fps(str(video_path)),
            "width": self._probe_width(str(video_path)),
            "height": self._probe_height(str(video_path)),
        }
        self._resume_manifest_path = self._manifest_path(video_path)
        resume = self._load_resume(video_path, duration)
        if resume:
            self.artifacts.update(resume.get("artifacts", {}))
            self.provenance = list(resume.get("provenance", []))
            self.artifacts["resume"] = {
                "manifest_path": str(self._resume_manifest_path),
                "resumed": True,
            }
        else:
            self.artifacts["resume"] = {
                "manifest_path": str(self._resume_manifest_path),
                "resumed": False,
            }

        if "authenticated_preflight" not in self.artifacts:
            self.artifacts["authenticated_preflight"] = (
                self._resolve_authenticated_preflight()
            )
            self._write_resume()
        self.runtime.get_client(self.model, api_key=self.api_key)
        # â”€â”€ Files API upload (once) â”€â”€
        try:
            file_handle = self.runtime.file_manager.upload_file(
                str(video_path), mime_type=mime_type
            )
        except GeminiError as exc:
            raise ProviderError(f"Gemini video upload failed: {exc}") from exc
        if not file_handle.uri:
            raise ProviderError("Gemini Files API upload returned no usable file URI.")
        # store upload_count / reuse_count snapshot at this point
        upload_count_snapshot = self.runtime.file_manager.upload_count
        reuse_count_snapshot = self.runtime.file_manager.reuse_count
        self.artifacts["video_upload"] = {
            "upload_count": upload_count_snapshot,
            "reuse_count": reuse_count_snapshot,
            "final_remote_state": file_handle.metadata.get("state"),
        }

        # â”€â”€ FPS smoke test (gate) â”€â”€
        if "fps_smoke_test" not in self.artifacts:
            self.artifacts["fps_smoke_test"] = self._resolve_fps_smoke_test(
                file_handle.uri, mime_type, duration
            )
            self._write_resume()
        if not self.artifacts["fps_smoke_test"]["effective"]:
            raise ProviderError("VIDEO_METADATA_FPS_NOT_EFFECTIVE")

        # â”€â”€ V1: Discovery â”€â”€
        raw_checkpoint = self.artifacts.get("checkpoints", {}).get("candidates_raw", {})
        candidates = raw_checkpoint.get("candidates")
        v1_reused = isinstance(candidates, list)
        if not isinstance(candidates, list):
            discovery = self._request(
                "v1_discovery", self._discovery_prompt(), self._discovery_schema(),
                file_handle.uri, mime_type, self.fps["discovery"],
                duration_seconds=duration,
            )
            raw_payload = discovery.get("candidates")
            if not isinstance(raw_payload, list):
                raise ProviderError("V1 response requires candidates.")
            raw_payload_count = len(raw_payload)
            raw_cap = min(
                self.max_raw_candidates, self.max_parseable_raw_candidates, 200
            )
            if raw_payload_count > raw_cap:
                raise ProviderError(
                    f"CANDIDATE_EXPLOSION: V1 raw={raw_payload_count} > max={raw_cap}"
                )
            candidates = self._validate_candidates(discovery, duration)
            self._checkpoint("candidates_raw", {"candidates": candidates})
        self.artifacts.setdefault("stage_status", {})["V1"] = {
            "status": "COMPLETED_REUSED" if v1_reused else "COMPLETED",
            "network_request_completed": not v1_reused,
            "network_request_skipped": v1_reused,
            "artifact_sha256": self._artifact_hash(candidates),
            "payload_sha256": (
                self.provenance[-1]["serialized_payload_sha256"]
                if not v1_reused and self.provenance else None
            ),
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }
        self._write_resume()
        raw_count = len(candidates)
        if raw_count > min(self.max_raw_candidates, 200):
            self._checkpoint("final", {
                "status": "CANDIDATE_EXPLOSION",
                "reason": f"V1 raw candidates ({raw_count}) exceed max_raw_candidates ({self.max_raw_candidates})",
                "v1_raw_count": raw_count,
            })
            raise ProviderError(
                f"CANDIDATE_EXPLOSION: V1 raw={raw_count} > max={self.max_raw_candidates}"
            )
        self._log_progress(f"[V1] raw={raw_count}")

        if not candidates:
            self._empty_artifacts()
            self._final_artifact(candidates)
            return []

        # â”€â”€ Deduplication â”€â”€
        dedup_checkpoint = self.artifacts.get("checkpoints", {}).get("candidates_deduplicated", {})
        merge_groups = dedup_checkpoint.get("merge_groups")
        if isinstance(dedup_checkpoint.get("candidates"), list) and isinstance(merge_groups, list):
            candidates = dedup_checkpoint["candidates"]
        else:
            candidates, merge_groups = self._deduplicate_candidates(candidates, duration)
            self._checkpoint("candidates_deduplicated", {"candidates": candidates, "merge_groups": merge_groups})
        dedup_count = len(candidates)
        self._log_progress(f"[V1] valid={raw_count} deduplicated={dedup_count}")
        if dedup_count > self.max_deduplicated_candidates:
            self._checkpoint("final", {
                "status": "CANDIDATE_EXPLOSION",
                "reason": f"Deduplicated candidates ({dedup_count}) exceed max ({self.max_deduplicated_candidates})",
                "v1_raw_count": raw_count,
                "deduplicated_count": dedup_count,
            })
            raise ProviderError(
                f"CANDIDATE_EXPLOSION: deduplicated={dedup_count} > max={self.max_deduplicated_candidates}"
            )
        self.artifacts["candidates"] = {"candidates": candidates}

        if self.stop_after_stage == "V1":
            self._checkpoint("final", {
                "status": "STOPPED_AFTER_V1",
                "resume_stage": "V2",
            })
            self._log_progress("[STOP] stop_after_stage=V1")
            return []

        # â”€â”€ V2: Classification â”€â”€
        classification_checkpoint = self.artifacts.get("checkpoints", {}).get("classifications", {})
        classifications = classification_checkpoint.get("payload")
        v2_reused = isinstance(classifications, dict)
        if not isinstance(classifications, dict):
            classifications = self._request(
                "v2_classification", self._classification_prompt(candidates),
                self._classification_schema(), file_handle.uri, mime_type,
                self.fps["classification"], duration_seconds=duration,
            )
            self._checkpoint("classifications", {"payload": classifications})
        accepted, rejected, review = self._validate_classifications(classifications, candidates)
        self.artifacts["classifications"] = {
            "classifications": classifications.get("classifications", [])
        }
        self._log_progress(
            f"[V2] accepted={len(accepted)} rejected={len(rejected)} review={len(review)}"
        )
        self.artifacts.setdefault("stage_status", {})["V2"] = {
            "status": "COMPLETED_REUSED" if v2_reused else "COMPLETED",
            "network_request_completed": not v2_reused,
            "network_request_skipped": v2_reused,
            "artifact_sha256": self._artifact_hash(classifications),
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }
        self._write_resume()

        # â”€â”€ Request planning + safety limits â”€â”€
        plan = self._compute_request_plan(accepted, duration)
        self._checkpoint("request_plan", plan)
        if plan["minimum_total_requests"] > self.max_total_stage_requests:
            self._checkpoint("final", {
                "status": "CANDIDATE_EXPLOSION",
                "reason": f"Planned requests ({plan['minimum_total_requests']}) exceed max ({self.max_total_stage_requests})",
                "plan": plan,
            })
            raise ProviderError(
                f"CANDIDATE_EXPLOSION: planned_requests={plan['minimum_total_requests']} > max={self.max_total_stage_requests}"
            )
        self._log_progress(
            f"[PLAN] V3={plan['v3_requests']} V4={plan['v4_requests']} "
            f"V5={plan['v5_requests']} TR={plan['translation_requests']} "
            f"min_total={plan['minimum_total_requests']}"
        )

        if len(accepted) > self.max_accepted_subtitle_tracks:
            self._checkpoint("final", {
                "status": "CANDIDATE_EXPLOSION",
                "reason": f"Accepted subtitle tracks ({len(accepted)}) exceed max ({self.max_accepted_subtitle_tracks})",
            })
            raise ProviderError(
                f"CANDIDATE_EXPLOSION: accepted={len(accepted)} > max={self.max_accepted_subtitle_tracks}"
            )

        if not accepted:
            self._empty_artifacts()
            self._final_artifact(candidates)
            return []

        # â”€â”€ V3: Temporal segmentation in same-stage batches â”€â”€
        temporal_checkpoint = self.artifacts.get("checkpoints", {}).get("temporal_events", {})
        events = temporal_checkpoint.get("events")
        v3_reused = isinstance(events, list)
        if not isinstance(events, list):
            events = []
        v3_inputs = [] if v3_reused else accepted
        for idx, batch in enumerate(self._batches(v3_inputs)):
            self._check_job_timeout()
            clip_start = max(0.0, min(item["start_seconds"] for item in batch) - self.clip_padding_seconds)
            clip_end = min(duration, max(item["end_seconds"] for item in batch) + self.clip_padding_seconds)
            if clip_start >= clip_end:
                continue
            payload = self._request(
                "v3_segmentation", self._batch_prompt("segment", batch),
                self._batch_schema("candidate_id", self._segmentation_schema()), file_handle.uri, mime_type,
                self.fps["segmentation"], start=clip_start, end=clip_end,
                duration_seconds=duration,
            )
            responses = self._batch_responses(payload, batch, "candidate_id")
            for candidate in batch:
                cand_events = self._validate_events(responses[candidate["candidate_id"]], candidate, clip_start, clip_end, duration)
                if len(cand_events) > self.max_events_per_track:
                    raise ProviderError(f"CANDIDATE_EXPLOSION: candidate {candidate['candidate_id']} events={len(cand_events)} > max={self.max_events_per_track}")
                events.extend(cand_events)
            self._log_progress(f"[V3] completed={idx+1}/{len(self._batches(accepted))}")
        self.artifacts["temporal_events"] = {"events": events}
        self._checkpoint("temporal_events", {"events": events})
        self.artifacts.setdefault("stage_status", {})["V3"] = {
            "status": "COMPLETED_REUSED" if v3_reused else "COMPLETED",
            "network_request_completed": not v3_reused,
            "network_request_skipped": v3_reused,
            "artifact_sha256": self._artifact_hash(events),
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }
        self._write_resume()

        # â”€â”€ V4: Geometry refinement in same-stage batches â”€â”€
        geometry_checkpoint = self.artifacts.get("checkpoints", {}).get("geometry", {})
        stored_geometry = geometry_checkpoint.get("events")
        v4_reused = isinstance(stored_geometry, list)
        geometry = (
            {item["event_id"]: item for item in stored_geometry}
            if v4_reused else {}
        )
        v4_inputs = [] if v4_reused else events
        for idx, batch in enumerate(self._batches(v4_inputs)):
            self._check_job_timeout()
            frame_sets = [
                {
                    "event": event,
                    "frames": [
                        {
                            "frame_role": role,
                            "timestamp_seconds": timestamp,
                            "image_bytes": self._extract_geometry_frame(
                                video_path, timestamp
                            ),
                        }
                        for role, timestamp in self._geometry_frame_times(event)
                    ],
                }
                for event in batch
            ]
            payload = self._request_image_geometry(
                self._image_geometry_prompt(frame_sets),
                self._batch_schema("event_id", self._image_geometry_schema()),
                frame_sets,
            )
            responses = self._batch_responses(payload, batch, "event_id")
            for event in batch:
                geometry[event["event_id"]] = self._validate_image_geometry(
                    responses[event["event_id"]], event
                )
            self._log_progress(f"[V4] completed={idx+1}/{len(self._batches(events))}")
        self.artifacts["geometry"] = {"events": list(geometry.values())}
        self._checkpoint("geometry", {"events": list(geometry.values())})
        v4_blocked = any(item.get("needs_review") for item in geometry.values())
        self.artifacts.setdefault("stage_status", {})["V4"] = {
            "request_status": "REUSED" if v4_reused else "COMPLETED",
            "validation_status": "REVIEW_REQUIRED" if v4_blocked else "ACCEPTED",
            "stage_status": "BLOCKED" if v4_blocked else "COMPLETED",
            "resume_stage": "V4" if v4_blocked else "V5",
            "network_request_completed": not v4_reused,
            "network_request_skipped": v4_reused,
            "artifact_sha256": self._artifact_hash(list(geometry.values())),
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }
        self._write_resume()

        # â”€â”€ V5: Transcription in same-stage batches â”€â”€
        transcription_checkpoint = self.artifacts.get("checkpoints", {}).get("transcriptions", {})
        transcriptions = transcription_checkpoint.get("events")
        v5_reused = isinstance(transcriptions, list) and bool(transcriptions)
        if not isinstance(transcriptions, list):
            transcriptions = []
        raw_v5_checkpoint = self.artifacts.get("checkpoints", {}).get(
            "v5_raw_transcription", {}
        )
        raw_v5_responses = raw_v5_checkpoint.get("batches")
        if not isinstance(raw_v5_responses, list):
            raw_v5_responses = []
        validation_audit: List[Dict[str, Any]] = []
        transcription_inputs = (
            [] if v5_reused else
            [{**event, "bbox": geometry[event["event_id"]]["representative_box_2d"]}
             for event in events if event["event_id"] in geometry]
        )
        for idx, batch in enumerate(self._batches(transcription_inputs)):
            self._check_job_timeout()
            clip_start = max(0.0, min(item["start_seconds_in_source"] for item in batch) - 0.25)
            clip_end = min(duration, max(item["end_seconds_in_source"] for item in batch) + 0.25)
            payload = self._request(
                "v5_transcription", self._batch_prompt("transcribe", batch),
                self._batch_schema("event_id", self._transcription_schema()), file_handle.uri, mime_type,
                self.fps["transcription"], start=clip_start, end=clip_end,
                duration_seconds=duration,
            )
            responses = self._batch_responses(payload, batch, "event_id")
            raw_v5_responses.append({
                "expected_event_ids": [item["event_id"] for item in batch],
                "payload": payload,
            })
            self._checkpoint("v5_raw_transcription", {
                "batches": raw_v5_responses,
                "artifact_sha256": self._artifact_hash(raw_v5_responses),
            })
            for event in batch:
                assessment = self._assess_transcription(
                    responses[event["event_id"]], event, source_language
                )
                validation_audit.append(assessment)
                transcription = assessment.get("transcription")
                if isinstance(transcription, dict):
                    transcriptions.append({**event, **geometry[event["event_id"]], **transcription})
            self._log_progress(f"[V5] completed={idx+1}/{len(self._batches(transcription_inputs))}")
        self.artifacts["transcriptions"] = {"events": transcriptions}
        validation_status = (
            "PASS" if transcriptions and len(transcriptions) == len(events) else
            "PARTIAL" if transcriptions else "REVIEW_REQUIRED"
        )
        self._checkpoint("transcriptions", {
            "events": transcriptions,
            "validation_audit": validation_audit,
            "request_status": "COMPLETED_REUSED" if v5_reused else "COMPLETED",
            "validation_status": validation_status,
            "stage_status": "COMPLETED" if transcriptions else "BLOCKED",
            "raw_artifact_hash": self._artifact_hash(raw_v5_responses),
            "validator_version": "v5_unicode_source_script_v1",
            "normalized_artifact_hash": self._artifact_hash(transcriptions),
        })
        self.artifacts.setdefault("stage_status", {})["V5"] = {
            "status": "COMPLETED_REUSED" if v5_reused and transcriptions else (
                "COMPLETED" if transcriptions else "BLOCKED"
            ),
            "network_request_completed": not v5_reused,
            "network_request_skipped": v5_reused,
            "artifact_sha256": self._artifact_hash(transcriptions),
            "request_status": "COMPLETED_REUSED" if v5_reused else "COMPLETED",
            "validation_status": validation_status,
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }
        self._write_resume()
        if not transcriptions:
            self._checkpoint("final", {
                "status": "REVIEW_REQUIRED",
                "resume_stage": "V5",
                "reason": "NO_READABLE_V5_TRANSCRIPTIONS",
            })
            return []

        # â”€â”€ Translation in same-stage batches â”€â”€
        translation_checkpoint = self.artifacts.get("checkpoints", {}).get("translations", {})
        translations = translation_checkpoint.get("events")
        translation_reused = isinstance(translations, list) and bool(translations)
        if not isinstance(translations, list):
            translations = []
        translation_inputs = [] if translation_reused else transcriptions
        for idx, batch in enumerate(self._batches(translation_inputs)):
            self._check_job_timeout()
            payload = self._text_request(
                "translation",
                self._batch_translation_prompt(batch, source_language),
                self._batch_schema("event_id", self._translation_schema()),
            )
            responses = self._batch_responses(payload, batch, "event_id")
            for item in batch:
                translated = self._validate_translation(responses[item["event_id"]], item["event_id"], item["text_verbatim"])
                translations.append({**item, "translated_text": translated or "", "translation_status": "ok" if translated else "REVIEW_REQUIRED"})
            self._log_progress(f"[TR] completed={idx+1}/{len(self._batches(transcriptions))}")
        self.artifacts["translations"] = {"events": translations}
        self._checkpoint("translations", {"events": translations})
        self.artifacts.setdefault("stage_status", {})["TRANSLATION"] = {
            "status": "COMPLETED_REUSED" if translation_reused else "COMPLETED",
            "network_request_completed": not translation_reused,
            "network_request_skipped": translation_reused,
            "artifact_sha256": self._artifact_hash(translations),
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }
        self._write_resume()

        # â”€â”€ Build render eligibility records and final segment list â”€â”€
        self._build_render_eligibility(candidates, accepted, events, geometry,
                                        transcriptions, translations, duration)
        self._final_artifact(candidates)

        return [
            Segment(
                id=item["event_id"], start=item["start_seconds_in_source"],
                end=item["end_seconds_in_source"], text=item["text_verbatim"],
                translated_text=item["translated_text"],
                source="ocr", confidence=1.0, confidence_available=False,
                bbox=self._pixel_free_bbox(item["representative_box_2d"]),
                metadata={"provider": "gemini_video_direct", "model": self.model,
                          "event_id": item["event_id"],
                          "source_language": source_language,
                          "geometry_source": "v4_direct_video"},
            )
            for item in translations
            if item["translated_text"]
        ]

    def debug_artifacts(self) -> Dict[str, Any]:
        return dict(self.artifacts)

    # â”€â”€ reset â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def _reset(self) -> None:
        self.provenance = []
        self.artifacts = {}
        self._render_eligibility = []
        self._job_start_time = 0.0
        self._last_progress_time = 0.0

    # â”€â”€ progress / checkpoint / timeout â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def _log_progress(self, message: str) -> None:
        self._last_progress_time = time.monotonic()
        # Safe console output (no secrets)
        print(message, flush=True)

    @staticmethod
    def _artifact_hash(value: Any) -> str:
        return hashlib.sha256(
            json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()

    def _audit_safe_slot(self, stage: str, counted_tokens: int) -> None:
        state = self.runtime.scheduler.ledger_state(counted_tokens)
        last_age = state["last_request_age_seconds"]
        spacing_wait = max(
            0.0,
            self.min_request_spacing_seconds - (last_age or 0.0),
        )
        wait = max(state["next_eligible_wait_seconds"], spacing_wait)
        audit = {
            "stage": stage,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "configured_rpm": state["configured_rpm"],
            "configured_tpm": state["configured_tpm"],
            "configured_rpd": state["configured_rpd"],
            "rolling_requests": state["rolling_requests"],
            "rolling_tokens": state["rolling_tokens"],
            "counted_tokens": counted_tokens,
            "minimum_request_spacing_seconds": self.min_request_spacing_seconds,
            "next_eligible_wait_seconds": wait,
        }
        self.artifacts["safe_slot_audit"] = audit
        if wait <= 0:
            return
        self._log_progress(
            f"[WAITING_FOR_SAFE_SLOT] stage={stage} wait={wait:.1f}s "
            f"requests={state['rolling_requests']} tokens={state['rolling_tokens']}"
        )
        if wait > self.max_quota_wait_seconds:
            raise ProviderError(
                f"WAITING_FOR_SAFE_SLOT: stage={stage} wait={wait:.1f}s "
                f"> max={self.max_quota_wait_seconds}s"
            )
        time.sleep(wait)
        self._log_progress(f"[SAFE_SLOT_READY] stage={stage}")

    def _checkpoint(self, name: str, data: Dict[str, Any]) -> None:
        self.artifacts.setdefault("checkpoints", {})[name] = data
        self._last_progress_time = time.monotonic()
        self._write_resume()

    def _mark_stage_blocked(self, stage: str, reason: str) -> None:
        self.artifacts.setdefault("stage_status", {})[stage] = {
            "status": "BLOCKED",
            "reason": reason,
            "completed_at": None,
        }
        self._checkpoint("final", {
            "status": "REVIEW_REQUIRED",
            "resume_stage": stage,
            "reason": reason,
        })

    def _manifest_path(self, video_path: Path) -> Path:
        digest = hashlib.sha256(video_path.read_bytes()).hexdigest()
        return self.resume_manifest_dir / f"{digest}.json"

    def _resume_identity(self, video_sha256: str, mime_type: str) -> Dict[str, Any]:
        return {
            "video_sha256": video_sha256,
            "mime_type": mime_type,
            "model": self.model,
            "media_resolution": self.media_resolution,
            "max_events_per_track": self.max_events_per_track,
            "batch_size": self.batch_size,
        }

    def _identity_matches(
        self, resume: Dict[str, Any], identity: Dict[str, Any]
    ) -> bool:
        stored = resume.get("identity")
        if not isinstance(stored, dict):
            return resume.get("video_sha256") == identity["video_sha256"]
        return stored == identity

    def _load_resume(self, video_path: Path, duration: float) -> Optional[Dict[str, Any]]:
        if not self.resume_enabled or self._resume_manifest_path is None:
            return None
        try:
            payload = json.loads(self._resume_manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        identity = self._resume_identity(
            hashlib.sha256(video_path.read_bytes()).hexdigest(), "video/mp4"
        )
        if (
            payload.get("checkpoint_schema_version") != self.checkpoint_schema_version
            or not self._identity_matches(payload, identity)
            or payload.get("duration_seconds") != duration
            or not isinstance(payload.get("artifacts"), dict)
        ):
            return None
        return payload

    def _write_resume(self) -> None:
        if not self.resume_enabled or self._resume_manifest_path is None:
            return
        self._resume_manifest_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "checkpoint_schema_version": self.checkpoint_schema_version,
            "video_sha256": self._resume_manifest_path.stem,
            "identity": self._resume_identity(
                self._resume_manifest_path.stem,
                self.artifacts.get("video_metadata", {}).get("mime_type", "video/mp4"),
            ),
            "duration_seconds": self.artifacts.get("video_metadata", {}).get("duration_seconds"),
            "artifacts": self.artifacts,
            "provenance": self.provenance,
        }
        temporary = self._resume_manifest_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True), encoding="utf-8")
        os.replace(temporary, self._resume_manifest_path)

    def _save_resume_manifest(
        self,
        video_sha256: str,
        identity: Dict[str, Any],
        duration: float,
        checkpoints: Dict[str, Any],
    ) -> None:
        if self._resume_manifest_path is None:
            raise ProviderError("Resume manifest path is not initialized.")
        self._resume_manifest_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "checkpoint_schema_version": self.checkpoint_schema_version,
            "video_sha256": video_sha256,
            "identity": identity,
            "duration_seconds": duration,
            "artifacts": {
                "checkpoints": checkpoints,
                "video_metadata": {"duration_seconds": duration},
            },
            "provenance": self.provenance,
        }
        temporary = self._resume_manifest_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True), encoding="utf-8")
        os.replace(temporary, self._resume_manifest_path)

    def _check_job_timeout(self) -> None:
        elapsed = time.monotonic() - self._job_start_time
        if elapsed > self.job_timeout_seconds:
            raise ProviderError(
                f"JOB_TIMEOUT: elapsed={elapsed:.1f}s > max={self.job_timeout_seconds}s"
            )
        no_progress = time.monotonic() - self._last_progress_time
        if no_progress > self.no_progress_timeout_seconds:
            raise ProviderError(
                f"NO_PROGRESS_TIMEOUT: no_progress={no_progress:.1f}s > max={self.no_progress_timeout_seconds}s"
            )

    # â”€â”€ quota management â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def _execute_with_quota_management(
        self, stage: str, request_fn: Callable[[], Any]
    ) -> Any:
        quota_attempts = 0
        quota_wait_remaining = self.max_quota_wait_seconds
        while True:
            try:
                return request_fn()
            except ProviderRateLimitError:
                raise
            except GeminiRateLimitError as exc:
                is_daily = self._is_daily_quota(exc)
                if is_daily and self.fail_fast_on_daily_quota:
                    raise ProviderRateLimitError(
                        f"Daily quota limit exceeded for stage {stage}: {exc}",
                        status_code=429,
                        error_status=exc.error_status,
                        error_code=exc.error_code,
                        error_details=exc.error_details,
                    ) from exc
                if quota_attempts >= self.max_quota_retries_per_stage:
                    raise ProviderRateLimitError(
                        f"Stage {stage} exhausted quota retries ({self.max_quota_retries_per_stage}): {exc}",
                    ) from exc
                delay = 2.0 * (2.0 ** quota_attempts)
                if self.honor_retry_after and exc.retry_after is not None:
                    delay = max(delay, exc.retry_after)
                if delay > quota_wait_remaining:
                    raise ProviderRateLimitError(
                        f"Stage {stage} bounded wait exceeded: waited {self.max_quota_wait_seconds - quota_wait_remaining:.1f}s, needs at least {delay:.1f}s more",
                    ) from exc
                self._log_progress(
                    f"[QUOTA:{stage}] Rate limited. Waiting {delay:.1f}s (remaining budget {quota_wait_remaining - delay:.1f}s)..."
                )
                time.sleep(delay)
                quota_wait_remaining -= delay
                quota_attempts += 1

    @staticmethod
    def _is_daily_quota(exc: GeminiRateLimitError) -> bool:
        msg = str(exc).lower()
        if "per day" in msg or "daily" in msg:
            return True
        if exc.error_status == "RESOURCE_EXHAUSTED" and "rate limit" not in msg:
            return True
        return False

    # â”€â”€ resume manifest â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def _save_legacy_resume_manifest(
        self, video_path: Path, completed_stages: List[str]
    ) -> None:
        if not self.resume_enabled:
            return
        path = video_path.parent / f".{video_path.name}.resume_manifest.json"
        data = {
            "checkpoint_schema_version": self.checkpoint_schema_version,
            "completed_stages": completed_stages,
            "video_path": str(video_path.resolve()),
            "video_size": video_path.stat().st_size,
            "video_mtime": video_path.stat().st_mtime,
            "artifacts": self.artifacts,
            "provenance": self.provenance,
        }
        with open(path, "w", encoding="utf-8") as fp:
            json.dump(data, fp, ensure_ascii=False, indent=2)

    def _load_resume_manifest(self, video_path: Path) -> Optional[Dict[str, Any]]:
        if not self.resume_enabled:
            return None
        path = video_path.parent / f".{video_path.name}.resume_manifest.json"
        if not path.is_file():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
        if data.get("checkpoint_schema_version") != self.checkpoint_schema_version:
            self._log_progress(f"[RESUME] Schema version mismatch: manifest={data.get('checkpoint_schema_version')}, current={self.checkpoint_schema_version}")
            return None
        try:
            manifest_video = Path(data.get("video_path", ""))
            st = video_path.stat()
        except (OSError, ValueError):
            return None
        if str(video_path.resolve()) != str(manifest_video.resolve()) or st.st_size != data.get("video_size") or st.st_mtime != data.get("video_mtime"):
            self._log_progress(f"[RESUME] Video file changed since manifest.")
            return None
        return data

    @staticmethod
    def _chunk(items: List[Any], n: int) -> List[List[Any]]:
        return [items[i:i + n] for i in range(0, len(items), n)]

    # â”€â”€ request helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def _quota_wait(
        self,
        stage: str,
        exc: GeminiRateLimitError,
        retries: int,
        payload_sha256: str,
        counted_tokens: int,
    ) -> None:
        details = json.dumps(exc.error_details, ensure_ascii=False).lower()
        message = f"{exc} {exc.error_status or ''} {details}".lower()
        if any(marker in message for marker in ("per day", "daily", "requests_per_day", "rpd")):
            classification = "RATE_LIMIT_RPD"
            metric = "generate_content_free_tier_requests_per_day"
        elif any(marker in message for marker in ("tokens", "tpm", "tokens_per_minute")):
            classification = "RATE_LIMIT_TPM"
            metric = "generate_content_free_tier_tokens"
        elif any(marker in message for marker in ("requests", "rpm", "requests_per_minute")):
            classification = "RATE_LIMIT_RPM"
            metric = "generate_content_free_tier_requests"
        elif any(marker in message for marker in ("overload", "unavailable", "503")):
            classification = "TRANSIENT_SERVER_OVERLOAD"
            metric = "server_overload"
        else:
            classification = "RESOURCE_EXHAUSTED_UNKNOWN"
            metric = "unknown"
        quota = {
            "stage": stage,
            "http_status": 429,
            "quota_classification": classification,
            "quota_metric": metric,
            "retry_after_seconds": exc.retry_after,
            "request_payload_sha256": payload_sha256,
            "counted_input_tokens": counted_tokens,
            "rolling_input_tokens": counted_tokens,
            "attempt": retries + 1,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self.artifacts["quota_failure"] = quota
        self.artifacts.setdefault("quota_events", []).append(quota)
        if classification == "RATE_LIMIT_RPD" and self.fail_fast_on_daily_quota:
            raise ProviderError(f"DAILY_QUOTA_EXHAUSTED: stage={stage} (RESUMABLE)") from exc
        if retries >= self.max_quota_retries_per_stage:
            raise ProviderError(f"QUOTA_BLOCKED: stage={stage} (RESUMABLE)") from exc
        wait = exc.retry_after if (self.honor_retry_after and exc.retry_after is not None) else None
        if wait is None or wait <= 0.0:
            wait = self.min_request_spacing_seconds
        if wait > self.max_quota_wait_seconds:
            raise ProviderError(
                f"QUOTA_BLOCKED: stage={stage} bounded_wait={wait:.1f}s "
                f"> max={self.max_quota_wait_seconds}s (RESUMABLE)"
            ) from exc
        quota["wait_seconds"] = wait
        self._log_progress(
            f"[WAITING_FOR_QUOTA] stage={stage} wait={wait:.1f}s "
            f"class={classification}"
        )
        self._write_resume()
        time.sleep(wait)

    def _request(
        self, stage: str, prompt: str, schema: Dict[str, Any], uri: str,
        mime_type: str, fps: float, start: Optional[float] = None,
        end: Optional[float] = None, duration_seconds: Optional[float] = None,
    ) -> Dict[str, Any]:
        clamp_start = start
        clamp_end = end
        if duration_seconds is not None:
            if clamp_start is not None:
                clamp_start = max(0.0, clamp_start)
            if clamp_end is not None:
                clamp_end = min(duration_seconds, clamp_end)
            else:
                clamp_end = duration_seconds

        part: Dict[str, Any] = {
            "fileData": {"fileUri": uri, "mimeType": mime_type},
            "videoMetadata": {"fps": fps},
        }
        video_metadata = {"fps": fps}
        if clamp_start is not None:
            video_metadata["startOffset"] = f"{clamp_start:.3f}s"
        if clamp_end is not None:
            video_metadata["endOffset"] = f"{clamp_end:.3f}s"
        part["videoMetadata"] = video_metadata

        request = GeminiRequest(
            contents=[{"parts": [{"text": prompt}, part]}], model=self.model,
            generation_config={"responseMimeType": "application/json",
                               "responseSchema": schema,
                               "mediaResolution": self._media_resolution_enum()},
            system_instruction=prompt, prompt_version=self.prompt_version,
            schema_version=self.schema_version, source_kind=stage,
            media_sha256=hashlib.sha256(uri.encode("utf-8")).hexdigest(),
            api_surface="generate_content",
        )
        client = self.runtime.get_client(self.model, role=MODE_ROLE_PRIMARY,
                                          api_key=self.api_key)

        last_error: Optional[str] = None
        response_payload: Optional[Dict[str, Any]] = None
        max_attempts = 1 + self.max_retry_per_stage

        stage_start_time = time.monotonic()
        quota_retries = 0
        for attempt in range(max_attempts):
            elapsed_stage = time.monotonic() - stage_start_time
            if elapsed_stage > self.stage_timeout_seconds:
                raise ProviderError(
                    f"STAGE_TIMEOUT: stage={stage} elapsed={elapsed_stage:.1f}s > "
                    f"max={self.stage_timeout_seconds}s"
                )
            try:
                # canonical serialized payload hash for countTokens + generate
                serialized_body = self._canonical_body(request)
                serialized_sha = hashlib.sha256(
                    json.dumps(serialized_body, sort_keys=True).encode("utf-8")
                ).hexdigest()
                request.predicted_tokens = client.count_tokens(request)
                counted_tokens = request.predicted_tokens
                if counted_tokens and counted_tokens > self.max_counted_input_tokens_per_job:
                    raise ProviderError(
                        f"TOKEN_BUDGET_EXCEEDED: predicted={counted_tokens} > max={self.max_counted_input_tokens_per_job}"
                    )
                self._audit_safe_slot(stage, counted_tokens or 0)
                response = client.generate_content(request)
                response_payload = parse_structured(response.text)
            except GeminiRateLimitError as exc:
                self._quota_wait(stage, exc, quota_retries, serialized_sha, counted_tokens)
                quota_retries += 1
                continue
            except TimeoutError as exc:
                self._mark_stage_blocked(
                    stage,
                    "NETWORK_TIMEOUT: generateContent response was not received",
                )
                raise ProviderError(f"NETWORK_TIMEOUT: stage={stage} (RESUMABLE)") from exc
            except GeminiError as exc:
                last_error = f"Gemini direct-video {stage} attempt {attempt+1} failed: {exc}"
                if attempt < max_attempts - 1:
                    continue
                raise ProviderError(last_error) from exc

            if response_payload is None or not isinstance(response_payload, dict):
                last_error = f"Gemini direct-video {stage} response was not an object on attempt {attempt+1}"
                if attempt < max_attempts - 1:
                    response_payload = None
                    continue
                raise ProviderError(last_error)

            # success
            if stage == "v5_transcription":
                self.artifacts.setdefault("raw_stage_responses", {}).setdefault(
                    stage, []
                ).append({
                    "serialized_payload_sha256": serialized_sha,
                    "response_text": response.text,
                })
                self._write_resume()
            self.provenance.append({
                "stage": stage, "input_type": "video", "model": self.model,
                "mime_type": mime_type, "file_data_present": True,
                "video_metadata_present": True, "requested_fps": fps,
                "clip_start": clamp_start, "clip_end": clamp_end,
                "media_resolution": self.media_resolution, "image_part_count": 0,
                "local_ocr_calls": 0,
                "serialized_payload_sha256": serialized_sha,
                "count_tokens": counted_tokens,
                "actual_usage": response.usage.to_dict() if response.usage else {},
                "cache_hit": getattr(response, "cache_hit", False),
                "retry_attempt": attempt,
            })
            return response_payload

        raise ProviderError(last_error or f"Gemini direct-video {stage} exhausted retries.")

    def _load_fps_certification(self) -> Optional[Dict[str, Any]]:
        try:
            certification = json.loads(self.fps_certification_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if certification.get("status") not in {"PARTIAL", "PASS"}:
            return None
        if certification.get("model") != self.model:
            return None
        if certification.get("fps_low") != 1 or certification.get("fps_high") != 6:
            return None
        if certification.get("payload_schema_version") != self.schema_version:
            return None
        if certification.get("fps_low_count_tokens") == certification.get("fps_high_count_tokens"):
            return None
        if not certification.get("video_metadata_serialized"):
            return None
        try:
            certified_at = datetime.fromisoformat(
                str(certification["certified_at"]).replace("Z", "+00:00")
            )
        except (KeyError, TypeError, ValueError):
            return None
        if (datetime.now(timezone.utc) - certified_at).total_seconds() > self.fps_certification_ttl_seconds:
            return None
        return certification

    def _load_authenticated_preflight_certification(
        self,
    ) -> Optional[Dict[str, Any]]:
        try:
            certification = json.loads(
                self.preflight_certification_path.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError):
            return None
        if certification.get("model") != self.model:
            return None
        if certification.get("http_status") != 200:
            return None
        if (
            certification.get("authentication_verdict")
            != "AUTHENTICATED_PREFLIGHT_PASS"
        ):
            return None
        credential_identity_hash = self._credential_identity_hash()
        if certification.get("credential_identity_hash") != credential_identity_hash:
            return None
        try:
            certified_at = datetime.fromisoformat(
                str(certification["timestamp"]).replace("Z", "+00:00")
            )
        except (KeyError, TypeError, ValueError):
            return None
        if (
            datetime.now(timezone.utc) - certified_at
        ).total_seconds() > self.preflight_certification_ttl_seconds:
            return None
        return certification

    def _credential_identity_hash(self) -> Optional[str]:
        key = self.preflight_credential_identity_hash
        if key is None:
            key = self.api_key or getattr(self.runtime._config, "api_key", None)
        if not isinstance(key, str) or not key:
            return None
        if len(key) == 64 and all(char in "0123456789abcdef" for char in key):
            return key
        return hashlib.sha256(key.encode("utf-8")).hexdigest()

    def _resolve_authenticated_preflight(self) -> Dict[str, Any]:
        certification = self._load_authenticated_preflight_certification()
        if certification is None:
            raise ProviderError("AUTHENTICATED_PREFLIGHT_NOT_REUSABLE")
        return {
            "status": "COMPLETED_REUSED",
            "authenticated": True,
            "network_request_skipped": True,
            "original_http_status": certification["http_status"],
            "certification_artifact_hash": hashlib.sha256(
                self.preflight_certification_path.read_bytes()
            ).hexdigest(),
            "certified_at": certification["timestamp"],
            "identity_validation": "MATCHED",
        }

    def _resolve_fps_smoke_test(
        self, uri: str, mime_type: str, duration: float
    ) -> Dict[str, Any]:
        certification = self._load_fps_certification()
        if certification is None:
            raise ProviderError("FPS_CERTIFICATION_NOT_REUSABLE")
        return {
            "status": "COMPLETED_REUSED",
            "effective": True,
            "network_request_skipped": True,
            "certification_artifact_hash": certification.get("project_identity_hash"),
            "identity_validation": "MATCHED",
            "fps_low_count_tokens": certification.get("fps_low_count_tokens"),
            "fps_high_count_tokens": certification.get("fps_high_count_tokens"),
            "certified_at": certification.get("certified_at"),
        }

    def _fps_smoke(self, uri: str, mime_type: str,
                    duration: float) -> Dict[str, Any]:
        records = []
        for fps in (1, 6):
            result = self._request(
                "fps_smoke",
                "Return only {\"ok\":true}. Do not transcribe, classify, or translate.",
                {"type": "object", "properties": {"ok": {"type": "boolean"}},
                 "required": ["ok"]},
                uri, mime_type, fps, 0.0, min(6.0, duration),
                duration_seconds=duration,
            )
            records.append({"fps": fps, "response": result,
                            **self.provenance[-1]})
        effective = (
            all(item["video_metadata_present"] for item in records)
            and (
                records[0].get("count_tokens") != records[1].get("count_tokens")
                or records[0].get("actual_usage") != records[1].get("actual_usage")
            )
        )
        return {"requests": records, "effective": effective}

    def _text_request(
        self, stage: str, prompt: str, schema: Dict[str, Any]
    ) -> Dict[str, Any]:
        request = GeminiRequest(
            contents=[{"parts": [{"text": prompt}]}],
            model=self.model,
            generation_config={"responseMimeType": "application/json",
                               "responseSchema": schema},
            system_instruction=prompt,
            prompt_version=self.prompt_version,
            schema_version=self.schema_version,
            source_kind=stage,
            api_surface="generate_content",
        )
        client = self.runtime.get_client(self.model, role=MODE_ROLE_PRIMARY,
                                          api_key=self.api_key)
        last_error: Optional[str] = None
        response_payload: Optional[Dict[str, Any]] = None
        max_attempts = 1 + self.max_retry_per_stage

        stage_start_time = time.monotonic()
        quota_retries = 0
        for attempt in range(max_attempts):
            elapsed_stage = time.monotonic() - stage_start_time
            if elapsed_stage > self.stage_timeout_seconds:
                raise ProviderError(
                    f"STAGE_TIMEOUT: stage={stage} elapsed={elapsed_stage:.1f}s > "
                    f"max={self.stage_timeout_seconds}s"
                )
            try:
                serialized_body = self._canonical_body(request)
                serialized_sha = hashlib.sha256(
                    json.dumps(serialized_body, sort_keys=True).encode("utf-8")
                ).hexdigest()
                request.predicted_tokens = client.count_tokens(request)
                counted_tokens = request.predicted_tokens
                if counted_tokens and counted_tokens > self.max_counted_input_tokens_per_job:
                    raise ProviderError(
                        f"TOKEN_BUDGET_EXCEEDED: predicted={counted_tokens} > max={self.max_counted_input_tokens_per_job}"
                    )
                self._audit_safe_slot(stage, counted_tokens or 0)
                response = client.generate_content(request)
                response_payload = parse_structured(response.text)
            except GeminiRateLimitError as exc:
                self._quota_wait(stage, exc, quota_retries, serialized_sha, counted_tokens)
                quota_retries += 1
                continue
            except TimeoutError as exc:
                self._mark_stage_blocked(
                    stage,
                    "NETWORK_TIMEOUT: generateContent response was not received",
                )
                raise ProviderError(f"NETWORK_TIMEOUT: stage={stage} (RESUMABLE)") from exc
            except GeminiError as exc:
                last_error = f"Gemini direct-video {stage} text attempt {attempt+1} failed: {exc}"
                if attempt < max_attempts - 1:
                    continue
                raise ProviderError(last_error) from exc

            if response_payload is None or not isinstance(response_payload, dict):
                last_error = f"Gemini direct-video {stage} text response invalid on attempt {attempt+1}"
                if attempt < max_attempts - 1:
                    response_payload = None
                    continue
                raise ProviderError(last_error)

            self.provenance.append({
                "stage": stage, "input_type": "text", "model": self.model,
                "image_part_count": 0, "local_ocr_calls": 0,
                "serialized_payload_sha256": serialized_sha,
                "count_tokens": counted_tokens,
                "actual_usage": response.usage.to_dict() if response.usage else {},
                "cache_hit": getattr(response, "cache_hit", False),
                "retry_attempt": attempt,
            })
            return response_payload

        raise ProviderError(last_error or f"Gemini direct-video {stage} text exhausted retries.")

    @staticmethod
    def _geometry_frame_times(event: Dict[str, Any]) -> List[tuple[str, float]]:
        start = float(event["start_seconds_in_source"])
        end = float(event["end_seconds_in_source"])
        duration = end - start
        middle = (start + end) / 2
        if duration < 0.8:
            return [("middle", middle)]
        early, late = start + duration * 0.15, end - duration * 0.15
        if duration <= 1.5:
            return [("early", early), ("late", late)]
        return [("early", early), ("middle", middle), ("late", late)]

    @staticmethod
    def _extract_geometry_frame(video_path: Path, timestamp: float) -> bytes:
        result = subprocess.run(
            [
                "ffmpeg", "-v", "error", "-i", str(video_path),
                "-ss", f"{timestamp:.3f}", "-frames:v", "1",
                "-f", "image2pipe", "-c:v", "mjpeg", "-",
            ],
            capture_output=True, check=False,
        )
        if result.returncode != 0 or not result.stdout:
            raise ProviderError("FFmpeg could not extract a V4 geometry frame.")
        return result.stdout

    def _image_geometry_prompt(self, frame_sets: Sequence[Dict[str, Any]]) -> str:
        mapping = [
            {
                "event_id": item["event"]["event_id"],
                "frames": [
                    {
                        "frame_role": frame["frame_role"],
                        "timestamp_seconds": frame["timestamp_seconds"],
                        "sha256": hashlib.sha256(frame["image_bytes"]).hexdigest(),
                    }
                    for frame in item["frames"]
                ],
            }
            for item in frame_sets
        ]
        return (
            "You are evaluating the safety of replacing an existing subtitle in original "
            "high-resolution frames from a 720x1280 portrait short video. The goal is not "
            "to avoid every overlap with a person. The original "
            "subtitle or its background may already cover part of a person or object. "
            "Replacing it in approximately the same visual region does not create new damage.\n\n"
            "For every frame separately, identify only the event's main dialogue subtitle. "
            "Do not confuse it with warning labels, creator notes, logos, account labels, "
            "stickers, UI overlays, or unrelated scene text.\n\n"
            "1. Existing occlusion: identify the person/object area already covered by "
            "source_text_bbox or source_container_bbox.\n"
            "2. Incremental occlusion: identify the additional person/object area covered by "
            "recommended_cover_bbox outside source_container_bbox.\n"
            "3. Unrelated content: determine whether the proposed cover newly hides a face, "
            "hand, important object, warning label, logo, or unrelated text.\n\n"
            "Return source_text_bbox, source_container_bbox, recommended_cover_bbox, "
            "source_container_overlaps_person and recommended_cover_overlaps_person for "
            "context, but do not use either as the final verdict. Set incremental_person_overlap, "
            "incremental_important_object_overlap, and incremental_unrelated_text_overlap only "
            "when the proposed cover newly hides that content outside the source container. "
            "List those new areas in incremental_overlap_regions. Return replacement_safety, "
            "geometry_confidence, safety_confidence, needs_review, and review_reason. "
            "replacement_safety must be one of "
            "SAFE_EQUIVALENT_OCCLUSION, SAFE_MINOR_EXPANSION, "
            "UNSAFE_INCREMENTAL_OCCLUSION, REVIEW_REQUIRED. Coordinates must be "
            "[y_min, x_min, y_max, x_max] on a 0-1000 scale. Do not mark an event unsafe "
            "only because both the source subtitle and replacement cover overlap the same "
            "already-occluded region. Mark unsafe only when the recommended cover introduces "
            "meaningful new occlusion outside the original subtitle container. The three supplied frames for "
            "each event must be evaluated together for stability and movement. Frame mapping:\n"
            + json.dumps(mapping, ensure_ascii=False)
        )

    def _request_image_geometry(
        self, prompt: str, schema: Dict[str, Any], frame_sets: Sequence[Dict[str, Any]]
    ) -> Dict[str, Any]:
        parts: List[Dict[str, Any]] = [{"text": prompt}]
        image_blobs: List[bytes] = []
        for item in frame_sets:
            for frame in item["frames"]:
                blob = frame["image_bytes"]
                image_blobs.append(blob)
                parts.extend([
                    {
                        "text": (
                            f"event_id={item['event']['event_id']} "
                            f"frame_role={frame['frame_role']} "
                            f"timestamp_seconds={frame['timestamp_seconds']:.3f}"
                        )
                    },
                    {
                        "inline_data": {
                            "mime_type": "image/jpeg",
                            "data": base64.b64encode(blob).decode("ascii"),
                        }
                    },
                ])
        request = GeminiRequest(
            contents=[{"parts": parts}], model=self.model,
            generation_config={"responseMimeType": "application/json", "responseSchema": schema},
            system_instruction=prompt, prompt_version=self.prompt_version,
            schema_version=self.schema_version, source_kind="v4_image_geometry",
            media_sha256=hashlib.sha256(b"".join(image_blobs)).hexdigest(),
            api_surface="generate_content",
        )
        client = self.runtime.get_client(self.model, role=MODE_ROLE_PRIMARY, api_key=self.api_key)
        request.predicted_tokens = client.count_tokens(request)
        self._audit_safe_slot("v4_image_geometry", request.predicted_tokens or 0)
        response = client.generate_content(request)
        payload = parse_structured(response.text)
        if not isinstance(payload, dict):
            raise ProviderError("V4 image geometry response was not an object.")
        self.provenance.append({
            "stage": "v4_image_geometry", "input_type": "image", "model": self.model,
            "image_part_count": len(image_blobs), "local_ocr_calls": 0,
            "count_tokens": request.predicted_tokens,
            "actual_usage": response.usage.to_dict() if response.usage else {},
            "cache_hit": getattr(response, "cache_hit", False),
        })
        return payload

    def _media_resolution_enum(self) -> str:
        resolution = self.media_resolution.upper()
        if resolution in {"LOW", "MEDIA_RESOLUTION_LOW"}:
            return "MEDIA_RESOLUTION_LOW"
        if resolution in {"MEDIUM", "MEDIA_RESOLUTION_MEDIUM"}:
            return "MEDIA_RESOLUTION_MEDIUM"
        if resolution in {"HIGH", "MEDIA_RESOLUTION_HIGH"}:
            return "MEDIA_RESOLUTION_HIGH"
        raise ProviderError(f"Unsupported Gemini media resolution: {self.media_resolution!r}.")

    def _batches(self, items: Sequence[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        return [list(items[index:index + self.batch_size]) for index in range(0, len(items), self.batch_size)]

    @staticmethod
    def _batch_schema(identifier: str, item_schema: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {"results": {"type": "array", "items": item_schema}},
            "required": ["results"],
        }

    @staticmethod
    def _batch_responses(
        payload: Dict[str, Any], expected: Sequence[Dict[str, Any]], identifier: str
    ) -> Dict[str, Dict[str, Any]]:
        results = payload.get("results")
        if not isinstance(results, list):
            results = []
        response_map = {
            item.get(identifier): item for item in results
            if isinstance(item, dict) and isinstance(item.get(identifier), str)
        }
        for item in expected:
            item_id = item[identifier]
            if item_id in response_map:
                continue
            if identifier == "candidate_id":
                response_map[item_id] = {"candidate_id": item_id, "events": []}
            else:
                response_map[item_id] = {
                    "event_id": item_id,
                    "representative_box_2d": item.get("representative_box_2d", [0, 0, 1, 1]),
                    "box_keyframes": [],
                    "text_verbatim": "",
                    "language": "unknown",
                    "script": "unknown",
                    "status": "review_required",
                    "translated_text": "",
                    "is_source_copy": False,
                    "reason_for_source_copy": "",
                }
        return response_map

    @staticmethod
    def _batch_prompt(stage: str, items: Sequence[Dict[str, Any]]) -> str:
        instructions = {
            "segment": "For every candidate, return stable subtitle events. Do not transcribe or translate.",
            "geometry": (
                "For every event, return text_bbox, container_bbox, and recommended_cover_bbox. "
                "recommended_cover_bbox must cover all source text plus its banner/bubble, border, "
                "outline, shadow, and safe padding. It may be asymmetrically wider than text_bbox, "
                "but must remain localized and exclude people, unrelated objects, logos, and scene text. "
                "For a visible banner or bubble, cover its full rendered width and height, not only the "
                "central text area, so no source characters remain visible at either edge. "
                "Use normalized [ymin, xmin, ymax, xmax] coordinates from 0 to 1000. "
                "Return overlay_type, geometry_confidence, and needs_review. Do not transcribe or translate."
            ),
            "transcribe": "For every event, transcribe only text within its supplied bbox. Do not translate.",
        }
        return (
            f"{instructions[stage]} Return one result for every supplied item in results. "
            + json.dumps({"items": list(items)}, ensure_ascii=False)
        )

    def _batch_translation_prompt(
        self, items: Sequence[Dict[str, Any]], source_language: Optional[str]
    ) -> str:
        return (
            f"Translate every supplied event from {source_language or 'unknown'} to "
            f"{self.target_language}. Return one result for every item. "
            "Preserve source-copy only with its reason.\n"
            + json.dumps({"items": list(items)}, ensure_ascii=False)
        )

    @staticmethod
    def _canonical_body(request: GeminiRequest) -> Dict[str, Any]:
        body: Dict[str, Any] = {"contents": request.contents}
        if request.generation_config is not None:
            body["generationConfig"] = request.generation_config
        if request.system_instruction is not None:
            body["systemInstruction"] = {"parts": [{"text": request.system_instruction}]}
        if request.safety_settings is not None:
            body["safetySettings"] = request.safety_settings
        return body

    # â”€â”€ bbox / geometry validation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    @staticmethod
    def _validate_box(box: Any) -> List[float]:
        if not isinstance(box, list) or len(box) != 4:
            raise ProviderError("Direct-video bbox must have four values.")
        values = [float(value) for value in box]
        ymin, xmin, ymax, xmax = values
        if min(values) < 0 or max(values) > 1000 or ymin >= ymax or xmin >= xmax:
            raise ProviderError("Direct-video bbox is invalid.")
        if (ymax - ymin) * (xmax - xmin) > 450000:
            raise ProviderError("Direct-video bbox exceeds cover safety policy.")
        return values

    @staticmethod
    def _box_area(box: Sequence[float]) -> float:
        ymin, xmin, ymax, xmax = box
        return (ymax - ymin) * (xmax - xmin)

    # â”€â”€ FFprobe helpers (pre-inference only) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    @staticmethod
    def _probe_duration(path: str) -> float:
        import subprocess
        try:
            result = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "default=noprint_wrappers=1:nokey=1", path],
                capture_output=True, text=True, timeout=30,
            )
            return float(result.stdout.strip())
        except Exception as exc:
            raise ProviderError(f"FFprobe duration failed: {exc}") from exc

    @staticmethod
    def _probe_fps(path: str) -> float:
        import subprocess
        try:
            result = subprocess.run(
                ["ffprobe", "-v", "error", "-select_streams", "v:0",
                 "-show_entries", "stream=r_frame_rate",
                 "-of", "default=noprint_wrappers=1:nokey=1", path],
                capture_output=True, text=True, timeout=30,
            )
            parts = result.stdout.strip().split("/")
            if len(parts) == 2:
                return float(parts[0]) / float(parts[1])
            return float(parts[0])
        except Exception:
            return 0.0

    @staticmethod
    def _probe_width(path: str) -> int:
        import subprocess
        try:
            result = subprocess.run(
                ["ffprobe", "-v", "error", "-select_streams", "v:0",
                 "-show_entries", "stream=width",
                 "-of", "default=noprint_wrappers=1:nokey=1", path],
                capture_output=True, text=True, timeout=30,
            )
            return int(result.stdout.strip())
        except Exception:
            return 0

    @staticmethod
    def _probe_height(path: str) -> int:
        import subprocess
        try:
            result = subprocess.run(
                ["ffprobe", "-v", "error", "-select_streams", "v:0",
                 "-show_entries", "stream=height",
                 "-of", "default=noprint_wrappers=1:nokey=1", path],
                capture_output=True, text=True, timeout=30,
            )
            return int(result.stdout.strip())
        except Exception:
            return 0

    # â”€â”€ V1 validation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def _validate_candidates(
        self, payload: Dict[str, Any], duration: float
    ) -> List[Dict[str, Any]]:
        items = payload.get("candidates")
        if not isinstance(items, list):
            raise ProviderError("V1 response requires candidates.")
        result, ids = [], set()
        for item in items:
            if not isinstance(item, dict):
                raise ProviderError("V1 candidate must be a dict.")
            cid = item.get("candidate_id")
            if not isinstance(cid, str):
                raise ProviderError("V1 candidate ID must be a string.")
            if cid in ids:
                raise ProviderError("V1 candidate IDs must be unique.")
            start, end = item.get("start_seconds"), item.get("end_seconds")
            if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
                raise ProviderError("V1 candidate timing must be numbers.")
            start_f, end_f = float(start), float(end)
            # Skip (do not kill the job) candidates with non-positive or
            # out-of-range duration; Gemini occasionally emits degenerate
            # intervals during discovery.
            if start_f < 0 or end_f <= start_f or end_f > duration:
                continue
            ids.add(cid)
            box = self._validate_box(item.get("representative_box_2d"))
            result.append({
                **item,
                "candidate_id": cid,
                "start_seconds": start_f,
                "end_seconds": end_f,
                "representative_box_2d": box,
            })
        return result

    # â”€â”€ Candidate deduplication â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    @staticmethod
    def _box_iou(a: Sequence[float], b: Sequence[float]) -> float:
        ymin_a, xmin_a, ymax_a, xmax_a = a
        ymin_b, xmin_b, ymax_b, xmax_b = b
        inter_ymin = max(ymin_a, ymin_b)
        inter_xmin = max(xmin_a, xmin_b)
        inter_ymax = min(ymax_a, ymax_b)
        inter_xmax = min(xmax_a, xmax_b)
        if inter_ymax <= inter_ymin or inter_xmax <= inter_xmin:
            return 0.0
        inter_area = (inter_ymax - inter_ymin) * (inter_xmax - inter_xmin)
        area_a = (ymax_a - ymin_a) * (xmax_a - xmin_a)
        area_b = (ymax_b - ymin_b) * (xmax_b - xmin_b)
        union_area = area_a + area_b - inter_area
        return inter_area / union_area if union_area > 0 else 0.0

    def _deduplicate_candidates(
        self, candidates: List[Dict[str, Any]], duration: float
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Merge frame-like detections into temporal tracks.

        Returns (deduplicated_candidates, merge_groups).
        """
        if not candidates:
            return [], []

        # Sort by start time for deterministic processing
        sorted_cands = sorted(candidates, key=lambda c: (c["start_seconds"], c["candidate_id"]))
        merged: List[List[Dict[str, Any]]] = []

        for cand in sorted_cands:
            placed = False
            for group in merged:
                last = group[-1]
                iou = self._box_iou(cand["representative_box_2d"], last["representative_box_2d"])
                temporal_gap = cand["start_seconds"] - last["end_seconds"]
                if iou > 0.5 and temporal_gap < 1.0:
                    group.append(cand)
                    placed = True
                    break
            if not placed:
                merged.append([cand])

        result: List[Dict[str, Any]] = []
        merge_groups: List[Dict[str, Any]] = []
        for idx, group in enumerate(merged):
            if not group:
                continue
            start = min(c["start_seconds"] for c in group)
            end = max(c["end_seconds"] for c in group)
            start = max(0.0, start)
            end = min(duration, end)
            # Representative bbox: median of each dimension
            ymins = [c["representative_box_2d"][0] for c in group]
            xmins = [c["representative_box_2d"][1] for c in group]
            ymaxs = [c["representative_box_2d"][2] for c in group]
            xmaxs = [c["representative_box_2d"][3] for c in group]
            try:
                med_box = [
                    statistics.median(ymins),
                    statistics.median(xmins),
                    statistics.median(ymaxs),
                    statistics.median(xmaxs),
                ]
            except statistics.StatisticsError:
                med_box = group[0]["representative_box_2d"]
            result.append({
                "candidate_id": f"merged_{idx}",
                "start_seconds": start,
                "end_seconds": end,
                "representative_box_2d": med_box,
            })
            merge_groups.append({
                "merged_id": f"merged_{idx}",
                "source_ids": [c["candidate_id"] for c in group],
                "count": len(group),
                "iou_threshold": 0.5,
                "gap_threshold": 1.0,
            })

        return result, merge_groups

    # â”€â”€ Request planning â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def _compute_request_plan(
        self, accepted: List[Dict[str, Any]], duration: float
    ) -> Dict[str, Any]:
        """Compute quota-safe same-stage batch plan before downstream dispatch."""
        v3 = max(1, len(self._batches(accepted))) if accepted else 0
        fixed = 4  # FPS smoke A/B, V1 discovery, V2 batch classification.
        max_events = len(accepted) * self.max_events_per_track
        v4 = v3
        v5 = v3
        tr = v3
        min_total = fixed + v3 + v4 + v5 + tr
        return {
            "old_plan_requests": fixed + len(accepted) * 4,
            "batched": True,
            "fixed_requests": fixed,
            "v3_requests": v3,
            "v4_requests": v4,
            "v5_requests": v5,
            "translation_requests": tr,
            "minimum_total_requests": min_total,
            "maximum_total_requests": fixed + v3 + v4 + v5 + tr,
            "accepted_subtitle_tracks": len(accepted),
            "max_events_per_track": self.max_events_per_track,
            "video_duration": duration,
        }

    # â”€â”€ V2 validation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def _validate_classifications(
        self, payload: Dict[str, Any], candidates: Sequence[Dict[str, Any]]
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        known = {item["candidate_id"]: item for item in candidates}
        accepted: List[Dict[str, Any]] = []
        rejected: List[Dict[str, Any]] = []
        review_required: List[Dict[str, Any]] = []
        entries = payload.get("classifications")
        if not isinstance(entries, list):
            raise ProviderError("V2 response requires classifications list.")

        seen_ids: set[str] = set()
        for item in entries:
            if not isinstance(item, dict):
                raise ProviderError("V2 classification entry must be a dict.")
            cid = item.get("candidate_id")
            classification = item.get("classification")
            decision = item.get("decision")
            if cid not in known or classification not in _ALL_CLASSES or decision not in {"accepted", "rejected"}:
                raise ProviderError(f"V2 classification for {cid} is invalid.")
            if cid in seen_ids:
                raise ProviderError(f"V2 duplicate classification for candidate {cid}.")
            seen_ids.add(cid)
            if decision == "accepted" and classification in _SUBTITLE_CLASSES:
                accepted.append(known[cid])
            elif decision == "rejected":
                rejected.append(known[cid])
            else:
                review_required.append(known[cid])

        # completeness: every V1 candidate must have exactly one V2 classification
        missing = set(known) - seen_ids
        if missing:
            raise ProviderError(
                f"V2 classification completeness: missing classifications for "
                f"candidates: {', '.join(sorted(missing))}"
            )
        return accepted, rejected, review_required

    # â”€â”€ V3 validation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    @staticmethod
    def _validate_events(
        payload: Dict[str, Any],
        candidate: Dict[str, Any],
        clip_start: float,
        clip_end: float,
        duration: float,
    ) -> List[Dict[str, Any]]:
        if payload.get("candidate_id") != candidate["candidate_id"]:
            raise ProviderError("V3 response candidate_id mismatch.")
        entries = payload.get("events")
        if not isinstance(entries, list):
            raise ProviderError("V3 response must contain events list.")
        clip_duration = clip_end - clip_start
        result: List[Dict[str, Any]] = []
        seen: set[str] = set()
        for item in entries:
            if not isinstance(item, dict):
                raise ProviderError("V3 event must be a dict.")
            raw_event_id = item.get("event_id")
            rel_start = item.get("start_seconds_in_source")
            rel_end = item.get("end_seconds_in_source")
            if not isinstance(raw_event_id, str) or not raw_event_id:
                raise ProviderError(
                    f"V3 event id is invalid: {raw_event_id!r}"
                )
            event_id = f"{candidate['candidate_id']}:{raw_event_id}"
            if event_id in seen:
                raise ProviderError(f"V3 event id is duplicated: {event_id!r}")
            seen.add(event_id)
            if not isinstance(rel_start, (int, float)) or not isinstance(rel_end, (int, float)):
                raise ProviderError("V3 event timing is invalid.")
            rs, re = float(rel_start), float(rel_end)

            if (
                0.0 <= rs <= clip_duration + 0.001
                and 0.0 <= re <= clip_duration + 0.001
            ):
                absolute_start = clip_start + rs
                absolute_end = clip_start + re
            elif rs >= clip_start - 0.001 and re <= duration + 0.001:
                absolute_start = rs
                absolute_end = re
            else:
                absolute_start = min(max(0.0, rs), duration)
                absolute_end = min(max(0.0, re), duration)

            # Skip (do not kill the job) events with non-positive or out-of-range
            # duration; model occasionally emits degenerate intervals.
            if absolute_end <= absolute_start:
                continue
            if absolute_start < -0.001 or absolute_end > duration + 0.001:
                continue

            clip_relative_start = absolute_start - clip_start
            clip_relative_end = absolute_end - clip_start
            result.append({
                **item,
                "candidate_id": candidate["candidate_id"],
                "event_id": event_id,
                "representative_box_2d": candidate["representative_box_2d"],
                "clip_source_start": clip_start,
                "clip_relative_start": clip_relative_start,
                "clip_relative_end": clip_relative_end,
                "source_absolute_start": absolute_start,
                "source_absolute_end": absolute_end,
                "start_seconds_in_source": absolute_start,
                "end_seconds_in_source": absolute_end,
            })
        return result

    # â”€â”€ V4 validation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def _validate_geometry(
        self, payload: Dict[str, Any], event: Dict[str, Any]
    ) -> Dict[str, Any]:
        if payload.get("event_id") != event["event_id"]:
            raise ProviderError("V4 event ID mismatch.")
        try:
            text_box = self._validate_box(payload.get("text_bbox"))
            container_box = self._validate_box(payload.get("container_bbox"))
            cover_box = self._validate_box(payload.get("recommended_cover_bbox"))
        except ProviderError as exc:
            raise ProviderError("V4 Gemini-only geometry is incomplete or invalid.") from exc
        if (
            cover_box[0] > text_box[0] or cover_box[1] > text_box[1]
            or cover_box[2] < text_box[2] or cover_box[3] < text_box[3]
            or cover_box[0] > container_box[0] or cover_box[1] > container_box[1]
            or cover_box[2] < container_box[2] or cover_box[3] < container_box[3]
        ):
            raise ProviderError(
                "V4 recommended cover must contain Gemini text and container boxes."
            )
        if self._box_area(cover_box) > 450000:
            raise ProviderError("V4 recommended cover exceeds safety policy.")
        confidence = payload.get("geometry_confidence")
        if not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
            raise ProviderError("V4 geometry confidence is invalid.")
        overlay_type = payload.get("overlay_type")
        if overlay_type not in {"banner", "bubble", "plain_text", "other"}:
            raise ProviderError("V4 overlay type is invalid.")
        if not isinstance(payload.get("needs_review"), bool):
            raise ProviderError("V4 needs_review must be boolean.")

        valid_keyframes = []
        for kf in payload.get("box_keyframes", []):
            if not isinstance(kf, dict):
                continue
            try:
                k_box = self._validate_box(kf.get("box_2d"))
                t = kf.get("timestamp_seconds")
                if not isinstance(t, (int, float)):
                    continue
                valid_keyframes.append({
                    "box_2d": k_box,
                    "timestamp_seconds": float(t)
                })
            except ProviderError:
                continue

        return {
            "event_id": event["event_id"],
            "text_bbox": text_box,
            "container_bbox": container_box,
            "recommended_cover_bbox": cover_box,
            "representative_box_2d": cover_box,
            "overlay_type": overlay_type,
            "geometry_confidence": float(confidence),
            "needs_review": payload["needs_review"],
            "box_keyframes": valid_keyframes,
        }

    def _validate_image_geometry(
        self, payload: Dict[str, Any], event: Dict[str, Any]
    ) -> Dict[str, Any]:
        if payload.get("event_id") != event["event_id"]:
            raise ProviderError("V4 image geometry event ID mismatch.")
        raw_observations = payload.get("frame_observations")
        if not isinstance(raw_observations, list):
            raise ProviderError("V4 image geometry observations are missing.")
        observations = []
        review_reasons = []
        unsafe = False
        expected_roles = {role for role, _ in self._geometry_frame_times(event)}
        for item in raw_observations:
            if not isinstance(item, dict):
                review_reasons.append("INVALID_OBSERVATION")
                continue
            try:
                text_box = self._validate_box(item.get("source_text_bbox"))
                container_box = self._validate_box(item.get("source_container_bbox"))
                cover_box = self._validate_box(item.get("recommended_cover_bbox"))
                if (
                    cover_box[0] > text_box[0] or cover_box[1] > text_box[1]
                    or cover_box[2] < text_box[2] or cover_box[3] < text_box[3]
                    or cover_box[0] > container_box[0] or cover_box[1] > container_box[1]
                    or cover_box[2] < container_box[2] or cover_box[3] < container_box[3]
                ):
                    raise ProviderError("Image cover does not contain source geometry.")
            except ProviderError:
                review_reasons.append("INVALID_OBSERVATION")
                continue
            replacement_safety = item.get("replacement_safety")
            if replacement_safety not in _REPLACEMENT_SAFETY:
                review_reasons.append("INVALID_SAFETY_VERDICT")
                continue
            if item.get("frame_role") not in expected_roles:
                review_reasons.append("INVALID_FRAME_ROLE")
                continue
            boolean_fields = (
                "source_container_overlaps_person",
                "recommended_cover_overlaps_person",
                "incremental_person_overlap",
                "incremental_important_object_overlap",
                "incremental_unrelated_text_overlap",
                "needs_review",
            )
            if any(not isinstance(item.get(field), bool) for field in boolean_fields):
                review_reasons.append("INVALID_SAFETY_OBSERVATION")
                continue
            confidence_fields = ("geometry_confidence", "safety_confidence")
            if any(
                not isinstance(item.get(field), (int, float))
                or not 0 <= float(item[field]) <= 1
                for field in confidence_fields
            ):
                review_reasons.append("INVALID_SAFETY_CONFIDENCE")
                continue
            if replacement_safety == "UNSAFE_INCREMENTAL_OCCLUSION":
                unsafe = True
                review_reasons.append("UNSAFE_INCREMENTAL_OCCLUSION")
                continue
            if (
                replacement_safety == "REVIEW_REQUIRED"
                or item["needs_review"]
                or any(item[field] for field in (
                    "incremental_person_overlap",
                    "incremental_important_object_overlap",
                    "incremental_unrelated_text_overlap",
                ))
            ):
                review_reasons.append("GEMINI_SAFETY_REVIEW")
                continue
            observations.append({
                "frame_role": item.get("frame_role"),
                "timestamp_seconds": item.get("timestamp_seconds"),
                "source_text_bbox": text_box,
                "source_container_bbox": container_box,
                "recommended_cover_bbox": cover_box,
                "visible_source_text": str(item.get("visible_source_text", "")),
                "overlay_type": str(item.get("overlay_type", "other")),
                "source_container_overlaps_person": item["source_container_overlaps_person"],
                "recommended_cover_overlaps_person": item["recommended_cover_overlaps_person"],
                "incremental_person_overlap": item["incremental_person_overlap"],
                "incremental_important_object_overlap": item["incremental_important_object_overlap"],
                "incremental_unrelated_text_overlap": item["incremental_unrelated_text_overlap"],
                "incremental_overlap_regions": item.get("incremental_overlap_regions", []),
                "replacement_safety": replacement_safety,
                "geometry_confidence": float(item.get("geometry_confidence", 0.0)),
                "safety_confidence": float(item.get("safety_confidence", 0.0)),
                "needs_review": False,
            })
        if (
            len(observations) != len(expected_roles)
            or {item["frame_role"] for item in observations} != expected_roles
        ):
            review_reasons.append("INCOMPLETE_FRAME_OBSERVATIONS")
        if payload.get("needs_review") is True:
            review_reasons.append("EVENT_REVIEW_REQUIRED")
        blocked = bool(review_reasons)
        if observations and not blocked:
            cover_box = [
                min(item["recommended_cover_bbox"][0] for item in observations),
                min(item["recommended_cover_bbox"][1] for item in observations),
                max(item["recommended_cover_bbox"][2] for item in observations),
                max(item["recommended_cover_bbox"][3] for item in observations),
            ]
            if self._box_area(cover_box) > 450000:
                review_reasons.append("COVER_SAFETY_REVIEW")
        else:
            cover_box = None
            if not review_reasons:
                review_reasons.append("NO_VALID_OBSERVATION")
        widths = [
            item["recommended_cover_bbox"][3] - item["recommended_cover_bbox"][1]
            for item in observations
        ]
        if widths and max(widths) - min(widths) > 150:
            review_reasons.append("INCONSISTENT_OBSERVATIONS")
        if review_reasons:
            cover_box = None
        confidence = payload.get("geometry_confidence")
        if not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
            raise ProviderError("V4 image geometry confidence is invalid.")
        return {
            "event_id": event["event_id"],
            "text_bbox": observations[0]["source_text_bbox"] if observations and not review_reasons else None,
            "container_bbox": observations[0]["source_container_bbox"] if observations and not review_reasons else None,
            "recommended_cover_bbox": cover_box,
            "representative_box_2d": cover_box,
            "image_level_recommended_cover_bbox": cover_box,
            "fallback_bbox": self._validate_box(event["representative_box_2d"]),
            "fallback_used_for_render": False,
            "geometry_confidence": float(confidence),
            "replacement_safety": (
                "UNSAFE_INCREMENTAL_OCCLUSION" if unsafe else
                "REVIEW_REQUIRED" if review_reasons else
                "SAFE_MINOR_EXPANSION" if any(
                    item["replacement_safety"] == "SAFE_MINOR_EXPANSION"
                    for item in observations
                ) else "SAFE_EQUIVALENT_OCCLUSION"
            ),
            "needs_review": bool(payload.get("needs_review")) or bool(review_reasons),
            "review_reason": ",".join(review_reasons) if review_reasons else None,
            "frame_observations": observations,
            "aggregation_method": "bounded_union",
            "box_keyframes": [],
        }

    # â”€â”€ V5 validation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    @staticmethod
    def _assess_transcription(
        payload: Dict[str, Any], event: Dict[str, Any],
        source_language: Optional[str] = None,
    ) -> Dict[str, Any]:
        event_id = event["event_id"]
        raw_text = payload.get("text_verbatim")
        audit: Dict[str, Any] = {
            "event_id": event_id,
            "raw_text_length": len(raw_text) if isinstance(raw_text, str) else None,
            "parsed_text_length": len(raw_text) if isinstance(raw_text, str) else None,
            "normalized_text_length": 0,
            "reject_reason": None,
            "transcription": None,
        }
        if payload.get("event_id") != event["event_id"]:
            audit["reject_reason"] = "UNKNOWN_EVENT_ID"
            return audit
        if not isinstance(raw_text, str):
            audit["reject_reason"] = "MALFORMED_STRUCTURED_OUTPUT"
            return audit
        cleaned = unicodedata.normalize("NFC", raw_text)
        cleaned = "".join(
            char for char in cleaned
            if not (
                char == "\ufffd"
                or (unicodedata.category(char) == "Cf")
                or (
                    unicodedata.category(char).startswith("C")
                    and char not in "\t\n\r"
                )
            )
        )
        cleaned = " ".join(cleaned.split()).strip()
        audit["normalized_text_length"] = len(cleaned)
        if not cleaned:
            audit["reject_reason"] = "EMPTY_AFTER_NORMALIZATION"
            return audit
        if not any(char.isalnum() for char in cleaned):
            audit["reject_reason"] = "PUNCTUATION_ONLY"
            return audit
        if len(cleaned) > 500:
            audit["reject_reason"] = "HALLUCINATED_LENGTH"
            return audit
        if payload.get("status") not in {"readable", "ok", "success", None}:
            audit["reject_reason"] = "MALFORMED_STRUCTURED_OUTPUT"
            return audit
        audit["transcription"] = {
            "text_verbatim": cleaned,
            "language": payload.get("language"),
            "script": payload.get("script"),
            "status": "readable",
        }
        return audit

    @staticmethod
    def _validate_transcription(
        payload: Dict[str, Any], event: Dict[str, Any],
        source_language: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        return GeminiDirectVideoOCRProvider._assess_transcription(
            payload, event, source_language
        ).get("transcription")

    # â”€â”€ translation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    @staticmethod
    def _validate_translation(
        payload: Dict[str, Any], event_id: str, source_text: str
    ) -> Optional[str]:
        if payload.get("event_id") != event_id:
            raise ProviderError("Translation event ID mismatch.")
        text = payload.get("translated_text")
        if not isinstance(text, str) or not text.strip():
            return None
        text = text.strip()
        source_copy = bool(payload.get("is_source_copy"))
        reason = payload.get("reason_for_source_copy")
        if source_copy and text != source_text:
            return None
        if text == source_text and (not source_copy or not isinstance(reason, str) or not reason.strip()):
            return None
        return text

    # â”€â”€ render eligibility â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def _build_render_eligibility(
        self,
        candidates: List[Dict[str, Any]],
        accepted_candidates: List[Dict[str, Any]],
        events: List[Dict[str, Any]],
        geometry: Dict[str, Dict[str, Any]],
        transcriptions: List[Dict[str, Any]],
        translations: List[Dict[str, Any]],
        duration: float,
    ) -> None:
        accepted_ids = {c["candidate_id"] for c in accepted_candidates}
        event_map = {e["event_id"]: e for e in events}
        geom_map = dict(geometry)
        tr_map = {t["event_id"]: t for t in transcriptions}
        tl_map = {t["event_id"]: t for t in translations}

        for c in candidates:
            cid = c["candidate_id"]
            c_valid = (
                "start_seconds" in c and "end_seconds" in c
                and 0 <= c["start_seconds"] < c["end_seconds"] <= duration
            )
            c_accepted = cid in accepted_ids
            # each event under this candidate
            for e in events:
                if e.get("candidate_id") != cid:
                    continue
                eid = e["event_id"]
                g = geom_map.get(eid)
                tr = tr_map.get(eid)
                tl = tl_map.get(eid)
                record = {
                    "event_id": eid,
                    "candidate_valid": c_valid,
                    "classification_valid": c_accepted,
                    "timing_valid": e["end_seconds_in_source"] > e["start_seconds_in_source"],
                    "geometry_valid": g is not None and "representative_box_2d" in g,
                    "transcription_valid": tr is not None and tr.get("status") == "readable",
                    "translation_valid": tl is not None and bool(tl.get("translated_text")),
                    "cover_safety_valid": g is not None
                        and self._box_area(g.get("representative_box_2d", [0, 0, 1, 1])) <= 450000,
                    "person_overlap_valid": False,
                    "render_eligible": False,
                    "reason_codes": [],
                }
                reasons = []
                if not c_valid:
                    reasons.append("INVALID_CANDIDATE")
                if not c_accepted:
                    reasons.append("CLASSIFICATION_REJECTED")
                if not record["timing_valid"]:
                    reasons.append("INVALID_TIMING")
                if not record["geometry_valid"]:
                    reasons.append("GEOMETRY_FAILED")
                if g is not None and g.get("needs_review"):
                    reasons.append("GEOMETRY_REVIEW_REQUIRED")
                if not record["transcription_valid"]:
                    reasons.append("TRANSCRIPTION_FAILED")
                if not record["translation_valid"]:
                    reasons.append("TRANSLATION_FAILED")
                if not record["cover_safety_valid"]:
                    reasons.append("COVER_SAFETY_FAILED")
                if not reasons:
                    record["render_eligible"] = True
                else:
                    record["reason_codes"] = reasons
                self._render_eligibility.append(record)

        self.artifacts["render_eligibility"] = {"events": self._render_eligibility}

    # â”€â”€ final artifact â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def _final_artifact(self, candidates: List[Dict[str, Any]]) -> None:
        paths = self.provenance
        eligible = [e for e in self._render_eligibility if e.get("render_eligible")]
        self.artifacts["final"] = {
            "total_requests": len(paths),
            "no_frame_input": all(
                p.get("image_part_count", 0) == 0 and p.get("input_type") == "video"
                for p in paths
                if p.get("input_type") == "video"
            ),
            "no_local_ocr": all(p.get("local_ocr_calls", 0) == 0 for p in paths),
            "total_accepted_events": len(eligible),
            "v1_candidate_count": len(candidates),
        }

    # â”€â”€ prompts â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    @staticmethod
    def _discovery_prompt() -> str:
        return (
            "Analyze the supplied video in chronological order. "
            "Locate every temporally changing text-overlay region that may contain "
            "viewer-facing subtitles. Return candidate text tracks with: candidate ID; "
            "approximate start and end time; representative bounding box. "
            "Do not classify the text as subtitle, title, watermark, or scene text. "
            "Do not transcribe the text. Do not translate. "
            "Do not combine spatially separate regions into one candidate."
        )

    @staticmethod
    def _classification_prompt(candidates: Sequence[Dict[str, Any]]) -> str:
        return (
            "Classify each supplied candidate text track using the original video. "
            "Choose exactly one classification: dialogue_subtitle, narration_subtitle, "
            "title_card, scene_text, watermark, logo, user_interface, warning_overlay, "
            "account_or_channel_label, other, review_required. "
            "Do not transcribe. Do not translate. "
            "Do not modify candidate timing or bounding boxes.\n"
            + json.dumps(candidates, ensure_ascii=False)
        )

    @staticmethod
    def _segmentation_prompt(candidate: Dict[str, Any]) -> str:
        return (
            "This video clip contains one accepted subtitle track. "
            "Identify each temporal interval during which the visible subtitle text "
            "remains unchanged. Return intervals only with source-absolute timestamps. "
            "Do not transcribe. Do not classify. Do not translate.\n"
            + json.dumps(candidate)
        )

    @staticmethod
    def _geometry_prompt(event: Dict[str, Any]) -> str:
        return (
            "Refine the bounding box of the supplied accepted subtitle event. "
            "Return a representative box and keyframes only when geometry materially "
            "changes. Do not transcribe. Do not classify. Do not translate.\n"
            + json.dumps(event)
        )

    @staticmethod
    def _transcription_prompt(
        event: Dict[str, Any], geometry: Dict[str, Any]
    ) -> str:
        return (
            "Transcribe only the subtitle text inside the supplied bounding box "
            "during this single stable subtitle event. Preserve original characters, "
            "punctuation, reading order, line breaks. "
            "Do not translate, classify, change timestamps, return a new bbox, "
            "or read text outside the supplied region. "
            "Return review_required if text is not stable or readable, and readable otherwise in the status field.\n"
            + json.dumps({"event": event, "bbox": geometry["representative_box_2d"]})
        )

    def _translation_prompt(
        self, event_id: str, text: str, source_language: Optional[str]
    ) -> str:
        lang = source_language or "unknown"
        return (
            f"Translate event {event_id} from {lang} to {self.target_language}. "
            "Translate naturally without summarizing or omitting content. "
            "Return source-copy only for intentional proper-name copy and explain why.\n"
            f"Source text: {text}"
        )

    # â”€â”€ schemas â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    @staticmethod
    def _discovery_schema() -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "video_id": {"type": "string"},
                "candidates": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "candidate_id": {"type": "string"},
                            "start_seconds": {"type": "number"},
                            "end_seconds": {"type": "number"},
                            "representative_box_2d": {
                                "type": "array", "items": {"type": "number"},
                            },
                        },
                        "required": ["candidate_id", "start_seconds",
                                      "end_seconds", "representative_box_2d"],
                    },
                },
            },
            "required": ["candidates"],
        }

    @staticmethod
    def _classification_schema() -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "classifications": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "candidate_id": {"type": "string"},
                            "classification": {"type": "string", "enum": ["dialogue_subtitle", "narration_subtitle", "title_card", "scene_text", "watermark", "logo", "user_interface", "warning_overlay", "account_or_channel_label", "other", "review_required"]},
                            "decision": {"type": "string", "enum": ["accepted", "rejected"]},
                        },
                        "required": ["candidate_id", "classification", "decision"],
                    },
                },
            },
            "required": ["classifications"],
        }

    @staticmethod
    def _segmentation_schema() -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "candidate_id": {"type": "string"},
                "events": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "event_id": {"type": "string"},
                            "start_seconds_in_source": {"type": "number"},
                            "end_seconds_in_source": {"type": "number"},
                        },
                        "required": ["event_id", "start_seconds_in_source",
                                      "end_seconds_in_source"],
                    },
                },
            },
            "required": ["candidate_id", "events"],
        }

    @staticmethod
    def _geometry_schema() -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "event_id": {"type": "string"},
                "text_bbox": {
                    "type": "array", "items": {"type": "number"},
                },
                "container_bbox": {
                    "type": "array", "items": {"type": "number"},
                },
                "recommended_cover_bbox": {
                    "type": "array", "items": {"type": "number"},
                },
                "overlay_type": {"type": "string", "enum": ["banner", "bubble", "plain_text", "other"]},
                "geometry_confidence": {"type": "number"},
                "needs_review": {"type": "boolean"},
                "box_keyframes": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "timestamp_seconds": {"type": "number"},
                            "box_2d": {"type": "array", "items": {"type": "number"}},
                        },
                    },
                },
            },
            "required": ["event_id", "text_bbox", "container_bbox",
                         "recommended_cover_bbox", "overlay_type",
                         "geometry_confidence", "needs_review"],
        }

    @staticmethod
    def _image_geometry_schema() -> Dict[str, Any]:
        observation = {
            "type": "object",
            "properties": {
                "frame_role": {"type": "string"},
                "timestamp_seconds": {"type": "number"},
                "source_text_bbox": {"type": "array", "items": {"type": "number"}},
                "source_container_bbox": {"type": "array", "items": {"type": "number"}},
                "recommended_cover_bbox": {"type": "array", "items": {"type": "number"}},
                "source_container_overlaps_person": {"type": "boolean"},
                "recommended_cover_overlaps_person": {"type": "boolean"},
                "incremental_person_overlap": {"type": "boolean"},
                "incremental_important_object_overlap": {"type": "boolean"},
                "incremental_unrelated_text_overlap": {"type": "boolean"},
                "incremental_overlap_regions": {"type": "array", "items": {"type": "string"}},
                "replacement_safety": {"type": "string", "enum": sorted(_REPLACEMENT_SAFETY)},
                "geometry_confidence": {"type": "number"},
                "safety_confidence": {"type": "number"},
                "needs_review": {"type": "boolean"},
                "review_reason": {"type": "string"},
            },
            "required": [
                "frame_role", "timestamp_seconds", "source_text_bbox",
                "source_container_bbox", "recommended_cover_bbox",
                "source_container_overlaps_person", "recommended_cover_overlaps_person",
                "incremental_person_overlap", "incremental_important_object_overlap",
                "incremental_unrelated_text_overlap", "incremental_overlap_regions",
                "replacement_safety", "geometry_confidence", "safety_confidence",
                "needs_review", "review_reason",
            ],
        }
        return {
            "type": "object",
            "properties": {
                "event_id": {"type": "string"},
                "frame_observations": {"type": "array", "items": observation},
                "event_recommended_cover_bbox": {
                    "type": "array", "items": {"type": "number"},
                },
                "geometry_confidence": {"type": "number"},
                "needs_review": {"type": "boolean"},
            },
            "required": [
                "event_id", "frame_observations",
                "event_recommended_cover_bbox", "geometry_confidence",
                "needs_review",
            ],
        }

    @staticmethod
    def _transcription_schema() -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "event_id": {"type": "string"},
                "text_verbatim": {"type": "string"},
                "language": {"type": "string"},
                "script": {"type": "string"},
                "status": {
                    "type": "string",
                    "enum": ["readable", "ok", "success", "review_required"],
                },
            },
            "required": ["event_id", "text_verbatim", "language", "script", "status"],
        }

    @staticmethod
    def _translation_schema() -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "event_id": {"type": "string"},
                "translated_text": {"type": "string"},
                "is_source_copy": {"type": "boolean"},
                "reason_for_source_copy": {"type": "string"},
            },
            "required": ["event_id", "translated_text"],
        }

    # â”€â”€ helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    @staticmethod
    def _pixel_free_bbox(box: Sequence[float]) -> Dict[str, float]:
        ymin, xmin, ymax, xmax = box
        return {"left": xmin, "top": ymin, "right": xmax, "bottom": ymax}

    def _empty_artifacts(self) -> None:
        for name in ("classifications", "temporal_events", "geometry",
                      "transcriptions", "translations", "render_eligibility"):
            self.artifacts.setdefault(name, {"events": []})

