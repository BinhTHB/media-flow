import json
import logging
import os
import shutil
import sys
import time
from typing import Any
from src.domain.entities.video_job import VideoJob
from src.domain.entities.job_status import JobStatus
from src.application.orchestrator.job_state_machine import JobStateMachine
from src.application.ports.artifact_provider import ArtifactProvider
from src.application.use_cases.extract_audio import ExtractAudioUseCase
from src.application.use_cases.filter_ocr_segments import FilterOCRSegmentsUseCase
from src.application.use_cases.group_segments import GroupSegmentsUseCase
from src.application.use_cases.transcribe_speech import TranscribeSpeechUseCase
from src.application.use_cases.extract_frames import ExtractFramesUseCase
from src.application.use_cases.detect_onscreen_text import DetectOnscreenTextUseCase
from src.application.use_cases.merge_segments import MergeSegmentsUseCase
from src.application.use_cases.normalize_segments import NormalizeSegmentsUseCase
from src.application.use_cases.subtitle_segmentation import SubtitleSegmentationUseCase
from src.application.use_cases.detect_subtitle_region import DetectSubtitleRegionUseCase
from src.application.use_cases.calculate_subtitle_placement import CalculateSubtitlePlacementUseCase
from src.application.use_cases.translate_segments import TranslateSegmentsUseCase
from src.application.use_cases.generate_subtitle import GenerateSubtitleUseCase
from src.application.use_cases.render_video import RenderVideoUseCase
from src.application.orchestrator.orchestrator_dependencies import OrchestratorDependencies
from src.shared.logging import FeatureLogger
from src.shared.exceptions import ValidationError
from src.domain.entities.subtitle_style import SubtitleStyle
from src.application.services.source_text_leakage_detector import SourceTextLeakageDetector
from src.application.services.content_quality_auditor import ContentQualityAuditor
from src.application.services.speech_quality_auditor import SpeechQualityAuditor
from src.application.services.vietnamese_text_quality_auditor import (
    VietnameseTextQualityAuditor,
)
from src.application.services.final_timeline_builder import FinalTimelineBuilder
from src.application.services.final_render_manifest_service import FinalRenderManifestService
from src.domain.entities.segment import Segment
from src.domain.entities.frame import Frame
from src.domain.entities.subtitle_region import SubtitleRegion
from src.application.services.artifact_cache_validator import ArtifactCacheValidator
from src.application.services.mask_authority import DecisionStore
from src.application.ports.ocr_provider import VideoSubtitleOCRProvider

audit_logger = logging.getLogger(__name__)


class JobOrchestrator:
    def __init__(self, dependencies: OrchestratorDependencies):
        self.extract_audio_uc = dependencies.extract_audio_uc
        self.filter_ocr_segments_uc = dependencies.filter_ocr_segments_uc
        self.group_segments_uc = dependencies.group_segments_uc
        self.get_video_duration_uc = dependencies.get_video_duration_uc
        self.transcribe_speech_uc = dependencies.transcribe_speech_uc
        self.detect_onscreen_text_uc = dependencies.detect_onscreen_text_uc
        self.translate_segments_uc = dependencies.translate_segments_uc
        self.generate_subtitle_uc = dependencies.generate_subtitle_uc
        self.render_video_uc = dependencies.render_video_uc
        self.extract_frames_uc = dependencies.extract_frames_uc
        self.merge_segments_uc = dependencies.merge_segments_uc
        self.normalize_segments_uc = dependencies.normalize_segments_uc
        self.subtitle_segmentation_uc = dependencies.subtitle_segmentation_uc
        self.detect_subtitle_region_uc = dependencies.detect_subtitle_region_uc
        self.calculate_subtitle_placement_uc = dependencies.calculate_subtitle_placement_uc
        self.generate_subtitle_placement_debug_uc = (
            dependencies.generate_subtitle_placement_debug_uc
        )
        self.generate_dubbed_audio_uc = dependencies.generate_dubbed_audio_uc
        self.mux_dubbed_video_uc = dependencies.mux_dubbed_video_uc
        self.final_timeline_builder = (
            dependencies.final_timeline_builder or FinalTimelineBuilder()
        )
        self.final_render_manifest_service = (
            dependencies.final_render_manifest_service or FinalRenderManifestService()
        )
        self.artifact_provider = dependencies.artifact_provider
        self.keep_audio = dependencies.keep_audio
        self.frame_interval = dependencies.frame_interval
        self.subtitle_format = dependencies.subtitle_format
        self.subtitle_style = dependencies.subtitle_style if dependencies.subtitle_style else SubtitleStyle()
        self.mask_decision_store = dependencies.mask_decision_store
        self.state_machine = JobStateMachine()

    def execute(self, job: VideoJob) -> VideoJob:
        self.state_machine.transition(job, JobStatus.RUNNING)
        debug_dir = self.artifact_provider.debug_dir(job.job_id)
        for entry in os.scandir(debug_dir):
            if entry.is_dir():
                shutil.rmtree(entry.path)
            else:
                os.remove(entry.path)
        self.mask_decision_store = DecisionStore(
            run_id=job.job_id,
            path=os.path.join(debug_dir, "mask_decisions.jsonl"),
        )
        speech_segments = []
        ocr_segments = []
        frames = []
        regions = []
        placements = []
        quality_metrics = {
            "ocr_segments_before": 0,
            "ocr_segments_after": 0,
            "dropped_by_confidence": 0,
            "dropped_by_length": 0,
            "dropped_by_abnormal_characters": 0,
            "dropped_by_repetition": 0,
            "source_text_leakage_count": 0,
            "source_text_leakage_examples": [],
            "quality_gate_status": "PASS",
        }
        ocr_metrics = {
            "dropped_by_confidence": 0,
            "dropped_by_length": 0,
            "dropped_by_abnormal_characters": 0,
            "dropped_by_repetition": 0,
            "dropped_by_garbage": 0,
            "filtered_tokens": [],
            "dropped_tokens": [],
        }
        controls = job.stage_controls
        direct_video_ocr = isinstance(
            self.detect_onscreen_text_uc.ocr_provider, VideoSubtitleOCRProvider
        )
        timings = {}
        cached_narration_segments = None
        video_duration = self.get_video_duration_uc.execute(job.input_video)
        video_width, video_height = self.get_video_duration_uc.video_provider.get_dimensions(
            job.input_video
        )
        placement_config = self.calculate_subtitle_placement_uc.placement_service.config
        placement_config.video_width = video_width
        placement_config.video_height = video_height
        replacement_required = (
            job.features.subtitle_placement
            and placement_config.placement_mode == "replace_source_subtitle"
        )
        cache_validator = ArtifactCacheValidator()
        try:
            if job.features.speech_to_text and controls.run_asr:
                started = time.monotonic()
                self.state_machine.transition(job, JobStatus.EXTRACTING_AUDIO)
                logger = FeatureLogger("ExtractAudio")
                logger.start()
                extracted_path = self.extract_audio_uc.execute(job.input_video)
                final_audio = self.artifact_provider.audio_path(job.job_id)
                if os.path.exists(extracted_path) and extracted_path != final_audio:
                    shutil.copy2(extracted_path, final_audio)
                    if not self.keep_audio:
                        try:
                            os.remove(extracted_path)
                        except OSError:
                            pass
                job.audio_path = final_audio
                logger.done()
                timings["asr"] = round(time.monotonic() - started, 3)

                self.state_machine.transition(job, JobStatus.TRANSCRIBING)
                logger = FeatureLogger("Speech")
                logger.start()
                transcript_path = self.artifact_provider.transcript_path(job.job_id)
                if os.path.exists(transcript_path):
                    speech_segments = [
                        Segment(**self._segment_fields(item))
                        for item in self.artifact_provider.read_json(transcript_path)
                    ]
                else:
                    speech_segments = self.transcribe_speech_uc.execute(
                        job.audio_path, job.speech_language
                    )
                    self.artifact_provider.write_json(
                        transcript_path, [s.to_dict() for s in speech_segments]
                    )
                logger.done()
                if not speech_segments:
                    raise ValidationError(
                        "Speech-to-text was requested but produced no transcript segments."
                    )

            needs_frames = (
                (job.features.ocr and controls.run_ocr and not direct_video_ocr)
                or (
                    job.features.subtitle_placement
                    and job.features.subtitle
                    and controls.run_subtitle_region_detection
                    and not direct_video_ocr
                )
            )
            if needs_frames:
                ocr_provider = self.detect_onscreen_text_uc.ocr_provider
                region_provider = self.detect_subtitle_region_uc.provider
                # Unwrap FallbackOCRProvider wrappers to reach the configured
                # hybrid provider without treating the wrapper as a credential failure.
                ocr_inner = getattr(ocr_provider, "primary", ocr_provider)
                region_inner = getattr(region_provider, "primary", region_provider)
                is_hybrid_ocr = ocr_inner.__class__.__name__ == "HybridGeminiOCRProvider"
                is_hybrid_region = (
                    region_inner.__class__.__name__ == "HybridSubtitleRegionDetectorProvider"
                )

                if is_hybrid_ocr or is_hybrid_region:
                    proposal = getattr(ocr_inner, "_proposal", None)
                    if proposal is None:
                        proposal = getattr(region_inner, "_proposal", None)
                    if proposal is None:
                        raise ValidationError(
                            "Hybrid OCR or Region Detector is enabled but YOLO text proposal provider is not configured. "
                            "Ensure the proposal provider is wired correctly in ProviderFactory."
                        )
                    if is_hybrid_ocr:
                        ocr_inner._proposal = proposal
                    if is_hybrid_region:
                        region_inner._proposal = proposal

                    audit_logger.info("[Proposal] Single-decode YOLO-on-video for job=%s", job.job_id)
                    bundle = proposal.detect_video(
                        job.input_video,
                        stride=1,
                        output_dir=self.artifact_provider.frames_dir(job.job_id),
                        min_pixel_std=1.0,
                    )
                    frames = [
                        Frame(timestamp=df.timestamp_sec, image_path=df.image_path or "")
                        for df in bundle.frames
                    ]
                    if is_hybrid_ocr:
                        ocr_inner.proposal_bundle = bundle
                        ocr_inner.set_run_context(
                            job.job_id, self.artifact_provider.debug_dir(job.job_id)
                        )
                    if is_hybrid_region:
                        region_inner.proposal_bundle = bundle
                        region_inner.set_decision_store(self.mask_decision_store)
                        region_inner.set_run_context(
                            job.job_id, self.artifact_provider.debug_dir(job.job_id)
                        )
                    self.artifact_provider.write_json(
                        os.path.join(
                            self.artifact_provider.debug_dir(job.job_id),
                            "proposal_bundle_metadata.json",
                        ),
                        bundle.to_metadata(),
                    )
                else:
                    frames = self.extract_frames_uc.execute(
                        job.input_video,
                        self.frame_interval,
                        self.artifact_provider.frames_dir(job.job_id),
                    )

            if (
                job.features.ocr
                and controls.run_ocr
                and not controls.reuse_ocr_cache
                and not replacement_required
            ):
                started = time.monotonic()
                self.state_machine.transition(job, JobStatus.DETECTING_TEXT)
                logger = FeatureLogger("OCR")
                logger.start()
                ocr_segments = self.detect_onscreen_text_uc.execute(
                    None if direct_video_ocr else frames,
                    video_path=job.input_video if direct_video_ocr else None,
                    source_language=job.source_language,
                )
                self._write_gemini_multistage_ocr_artifacts(job.job_id)
                self._write_gemini_direct_video_ocr_artifacts(job.job_id)
                raw_ocr = [s.to_dict() for s in ocr_segments]
                self.artifact_provider.write_json(
                    os.path.join(
                        self.artifact_provider.debug_dir(job.job_id),
                        "raw_ocr.json",
                    ),
                    {"segments": raw_ocr},
                )
                ocr_segments, ocr_metrics = self.filter_ocr_segments_uc.execute(ocr_segments, job.source_language)
                filtered_ids = {s.id for s in ocr_segments}
                raw_ocr_records = [
                    {
                        **segment,
                        "frame_id": segment["id"].rsplit("_", 1)[0],
                        "frame_timestamp": segment["start"],
                        "dropped_tokens": (
                            [segment["text"]] if segment["id"] not in filtered_ids else []
                        ),
                        "drop_reasons": (
                            ["ocr_quality_filter"]
                            if segment["id"] not in filtered_ids
                            else []
                        ),
                        "group_id": None,
                        "source_segment_ids": [segment["id"]],
                        "filtered_tokens": [],
                    }
                    for segment in raw_ocr
                ]
                self.artifact_provider.write_json(
                    os.path.join(
                        self.artifact_provider.debug_dir(job.job_id),
                        "raw_ocr.json",
                    ),
                    {
                        "segments": raw_ocr_records,
                        "dropped_tokens": ocr_metrics.get("dropped_tokens", []),
                        "drop_reasons": {
                            "confidence": ocr_metrics.get("dropped_by_confidence", 0),
                            "length": ocr_metrics.get("dropped_by_length", 0),
                            "abnormal_characters": ocr_metrics.get(
                                "dropped_by_abnormal_characters", 0
                            ),
                            "repetition": ocr_metrics.get("dropped_by_repetition", 0),
                            "garbage": ocr_metrics.get("dropped_by_garbage", 0),
                        },
                        "filtered_tokens": ocr_metrics.get("filtered_tokens", []),
                    },
                )
                self.artifact_provider.write_json(
                    os.path.join(
                        self.artifact_provider.debug_dir(job.job_id),
                        "filtered_ocr.json",
                    ),
                    {
                        "segments": [
                            {
                                **segment.to_dict(),
                                "frame_id": segment.id.rsplit("_", 1)[0],
                                "frame_timestamp": segment.start,
                                "dropped_tokens": [],
                                "drop_reasons": [],
                                "group_id": None,
                                "source_segment_ids": [segment.id],
                                "filtered_tokens": ocr_metrics.get(
                                    "filtered_tokens", []
                                ),
                            }
                            for segment in ocr_segments
                        ],
                        "dropped_tokens": ocr_metrics.get("dropped_tokens", []),
                        "drop_reasons": {
                            "confidence": ocr_metrics.get("dropped_by_confidence", 0),
                            "length": ocr_metrics.get("dropped_by_length", 0),
                            "abnormal_characters": ocr_metrics.get(
                                "dropped_by_abnormal_characters", 0
                            ),
                            "repetition": ocr_metrics.get("dropped_by_repetition", 0),
                            "garbage": ocr_metrics.get("dropped_by_garbage", 0),
                        },
                        "filtered_tokens": ocr_metrics.get("filtered_tokens", []),
                    },
                )
                quality_metrics.update(ocr_metrics)
                ocr_prov = self.detect_onscreen_text_uc.ocr_provider
                provider_name = getattr(ocr_prov, "primary_name", ocr_prov.__class__.__name__)
                self.artifact_provider.write_json(
                    os.path.join(self.artifact_provider.debug_dir(job.job_id), "ocr_cache_metadata.json"),
                    cache_validator.signature(
                        job.input_video, self.frame_interval,
                        provider_name,
                        "unknown",
                        getattr(self.detect_onscreen_text_uc.ocr_provider, "language", ""),
                        "gpu" if getattr(self.detect_onscreen_text_uc.ocr_provider, "use_gpu", False) else "cpu",
                        video_duration,
                        f"{self.calculate_subtitle_placement_uc.placement_service.config.video_width}x{self.calculate_subtitle_placement_uc.placement_service.config.video_height}",
                    ),
                )
                logger.done()
                timings["ocr_content"] = round(time.monotonic() - started, 3)
            elif (
                job.features.ocr
                and controls.reuse_ocr_cache
                and not replacement_required
            ):
                cache_root = controls.debug_from_artifacts or self.artifact_provider.debug_dir(job.job_id)
                metadata = self.artifact_provider.read_json(
                    os.path.join(cache_root, "debug", "ocr_cache_metadata.json")
                    if controls.debug_from_artifacts
                    else os.path.join(cache_root, "ocr_cache_metadata.json")
                )
                ocr_prov2 = self.detect_onscreen_text_uc.ocr_provider
                cached_provider_name = getattr(ocr_prov2, "primary_name", ocr_prov2.__class__.__name__)
                current = cache_validator.signature(
                    job.input_video, self.frame_interval,
                    cached_provider_name,
                    "unknown",
                    getattr(self.detect_onscreen_text_uc.ocr_provider, "language", ""),
                    "gpu" if getattr(self.detect_onscreen_text_uc.ocr_provider, "use_gpu", False) else "cpu",
                    video_duration,
                    f"{self.calculate_subtitle_placement_uc.placement_service.config.video_width}x{self.calculate_subtitle_placement_uc.placement_service.config.video_height}",
                )
                if controls.debug_from_artifacts:
                    current["video_resolution"] = metadata.get("video_resolution")
                if not cache_validator.is_valid(metadata, current):
                    raise ValidationError("OCR cache is missing or invalid for this input/configuration.")
                cached = self.artifact_provider.read_json(
                    os.path.join(cache_root, "debug", "filtered_ocr.json")
                    if controls.debug_from_artifacts
                    else os.path.join(cache_root, "filtered_ocr.json")
                )
                ocr_segments = [Segment(**self._segment_fields(item)) for item in cached["segments"]]

            if (
                job.features.subtitle_placement
                and job.features.subtitle
                and frames
                and controls.run_subtitle_region_detection
                and not controls.reuse_subtitle_region_cache
            ):
                started = time.monotonic()
                regions = self.detect_subtitle_region_uc.execute(frames)
                if replacement_required:
                    ocr_segments = self._ocr_segments_for_confirmed_tracks(
                        regions, frames, video_width, video_height
                    )
                self.artifact_provider.write_json(
                    os.path.join(
                        self.artifact_provider.debug_dir(job.job_id),
                        "subtitle_regions.json",
                    ),
                    [region.to_dict(include_provenance=True) for region in regions],
                )
                region_provider = self.detect_subtitle_region_uc.provider
                if hasattr(region_provider, "static_rule_ledger"):
                    self.artifact_provider.write_json(
                        os.path.join(
                            self.artifact_provider.debug_dir(job.job_id),
                            "static_content_rule_ledger.json",
                        ),
                        region_provider.static_rule_ledger(),
                    )
                self.artifact_provider.write_json(
                    os.path.join(self.artifact_provider.debug_dir(job.job_id), "subtitle_region_cache_metadata.json"),
                    cache_validator.signature(
                        job.input_video, self.frame_interval,
                        self.detect_subtitle_region_uc.provider.__class__.__name__,
                        "unknown",
                        getattr(self.detect_subtitle_region_uc.provider, "language", ""),
                        "gpu" if getattr(self.detect_subtitle_region_uc.provider, "use_gpu", False) else "cpu",
                        video_duration,
                        f"{self.calculate_subtitle_placement_uc.placement_service.config.video_width}x{self.calculate_subtitle_placement_uc.placement_service.config.video_height}",
                    ),
                )
                timings["subtitle_region_ocr"] = round(time.monotonic() - started, 3)
            elif job.features.subtitle_placement and controls.reuse_subtitle_region_cache:
                cache_root = controls.debug_from_artifacts or self.artifact_provider.debug_dir(job.job_id)
                metadata = self.artifact_provider.read_json(
                    os.path.join(cache_root, "debug", "subtitle_region_cache_metadata.json")
                    if controls.debug_from_artifacts
                    else os.path.join(cache_root, "subtitle_region_cache_metadata.json")
                )
                current = cache_validator.signature(
                    job.input_video, self.frame_interval,
                    self.detect_subtitle_region_uc.provider.__class__.__name__,
                    "unknown",
                    getattr(self.detect_subtitle_region_uc.provider, "language", ""),
                    "gpu" if getattr(self.detect_subtitle_region_uc.provider, "use_gpu", False) else "cpu",
                    video_duration,
                    f"{self.calculate_subtitle_placement_uc.placement_service.config.video_width}x{self.calculate_subtitle_placement_uc.placement_service.config.video_height}",
                )
                if controls.debug_from_artifacts:
                    current["video_resolution"] = metadata.get("video_resolution")
                if not cache_validator.is_valid(metadata, current):
                    raise ValidationError("Subtitle-region cache is missing or invalid for this input/configuration.")
                aggregate_path = (
                    os.path.join(cache_root, "debug", "subtitle_regions.json")
                    if controls.debug_from_artifacts
                    else os.path.join(cache_root, "subtitle_regions.json")
                )
                if os.path.exists(aggregate_path):
                    cached = self.artifact_provider.read_json(aggregate_path)
                else:
                    frame_dir = (
                        os.path.join(cache_root, "debug", "subtitle_region_frames")
                        if controls.debug_from_artifacts
                        else os.path.join(cache_root, "subtitle_region_frames")
                    )
                    cached = []
                    for filename in sorted(os.listdir(frame_dir)):
                        if filename.endswith(".json"):
                            checkpoint = self.artifact_provider.read_json(
                                os.path.join(frame_dir, filename)
                            )
                            cached.extend(checkpoint.get("regions", []))
                regions = [SubtitleRegion(**item) for item in cached]

            if controls.debug_from_artifacts or not (
                controls.run_asr or controls.run_ocr
            ):
                root = controls.debug_from_artifacts or self.artifact_provider.job_dir(job.job_id)
                path = os.path.join(
                    root,
                    "transcripts",
                    "normalized.json" if controls.run_segmentation else "segmented.json",
                )
                cached = self.artifact_provider.read_json(path)
                segments = [Segment(**self._segment_fields(item)) for item in cached]
                normalized_path = os.path.join(root, "transcripts", "normalized.json")
                if os.path.exists(normalized_path):
                    normalized = self.artifact_provider.read_json(normalized_path)
                    cached_narration_segments = [
                        Segment(**self._segment_fields(item))
                        for item in normalized
                        if item.get("source") == "speech"
                    ]
            else:
                segments = self.merge_segments_uc.execute(speech_segments, ocr_segments)

            # Translation Grouping
            segments = self.group_segments_uc.execute(segments, job.source_language)
            after_group_examples = [s.to_dict() for s in segments]
            self.artifact_provider.write_json(
                os.path.join(
                    self.artifact_provider.debug_dir(job.job_id),
                    "grouping_provenance.json",
                ),
                {
                    "groups": [
                        {
                            **segment,
                            "group_id": segment["id"],
                            "source_segment_ids": self.group_segments_uc.provenance_map.get(
                                segment["id"], [segment["id"]]
                            ),
                            "dropped_tokens": ocr_metrics.get("dropped_tokens", []),
                            "drop_reasons": [],
                            "filtered_tokens": ocr_metrics.get(
                                "filtered_tokens", []
                            ),
                        }
                        for segment in after_group_examples
                    ],
                    "dropped_tokens": ocr_metrics.get("dropped_tokens", []),
                    "filtered_tokens": ocr_metrics.get("filtered_tokens", []),
                },
            )

            grouping_examples = []
            for a in after_group_examples:
                source_ids = self.group_segments_uc.provenance_map.get(
                    a["id"], [a["id"]]
                )
                if len(source_ids) > 1:
                    grouping_examples.append({
                        "source_segment_ids": source_ids,
                        "after": a["text"],
                        "source_type": a["source"],
                        "start": a["start"],
                        "end": a["end"],
                    })

            if job.features.translate and controls.run_translation:
                started = time.monotonic()
                translation_input = list(segments)
                self.state_machine.transition(job, JobStatus.TRANSLATING)
                logger = FeatureLogger("Translate")
                logger.start()
                translation_provider = self.translate_segments_uc.translation_provider
                set_run_context = getattr(translation_provider, "set_run_context", None)
                if callable(set_run_context):
                    set_run_context(
                        job.job_id, self.artifact_provider.debug_dir(job.job_id)
                    )
                segments = self.translate_segments_uc.execute(segments, job.target_language)
                self.artifact_provider.write_json(
                    self.artifact_provider.translated_path(job.job_id),
                    [s.to_dict() for s in segments]
                )
                logger.done()
                timings["translation"] = round(time.monotonic() - started, 3)
                normalize_translated = True
            else:
                translation_input = list(segments)
                normalize_translated = any(
                    bool(segment.translated_text) for segment in segments
                )

            speech_quality_report = SpeechQualityAuditor.report(
                speech_segments,
                translation_input,
                segments,
                job.speech_language,
            )
            quality_metrics.update({
                key: value
                for key, value in speech_quality_report.items()
                if key not in ("asr_segments", "translated_speech_segments", "warnings")
            })
            quality_metrics["speech_quality_warnings"] = speech_quality_report["warnings"]
            self.artifact_provider.write_json(
                os.path.join(
                    self.artifact_provider.debug_dir(job.job_id),
                    "speech_quality_report.json",
                ),
                speech_quality_report,
            )

            source_counts_before = {
                source: sum(segment.source == source for segment in segments)
                for source in ("speech", "ocr")
            }
            if controls.run_normalization:
                started = time.monotonic()
                segments = self.normalize_segments_uc.execute(
                    segments,
                    is_translated=normalize_translated,
                    video_duration=video_duration,
                    replacement_required=replacement_required,
                )
                timings["normalization"] = round(time.monotonic() - started, 3)
            quality_metrics.update(self.normalize_segments_uc.metrics)
            self.artifact_provider.write_json(
                self.artifact_provider.normalized_path(job.job_id),
                [s.to_dict() for s in segments]
            )
            narration_segments = cached_narration_segments or list(segments)

            # Subtitle Segmentation Stage
            if controls.run_segmentation:
                started = time.monotonic()
                segments = self.subtitle_segmentation_uc.execute(
                    segments, is_translated=normalize_translated
                )
                timings["segmentation"] = round(time.monotonic() - started, 3)
            content_config = self.normalize_segments_uc.config
            if content_config.enable_content_quality_audit:
                report = []
                kept_segments = []
                for index, segment in enumerate(segments, start=1):
                    text = (
                        segment.translated_text
                        if normalize_translated
                        else segment.text
                    )
                    score, findings = ContentQualityAuditor.audit(text, segment.source)
                    recommended = "drop" if findings or score < content_config.min_caption_quality_score else "keep"
                    actual = "keep"
                    reason = "report-only audit"
                    if content_config.enable_content_quality_filtering:
                        should_drop = score < content_config.min_caption_quality_score
                        should_drop = should_drop or (
                            content_config.drop_placeholder_captions
                            and "placeholder_only" in findings
                        )
                        should_drop = should_drop or (
                            content_config.drop_malformed_ocr_captions
                            and "malformed_ocr_spelling" in findings
                        )
                        should_drop = should_drop or (
                            content_config.drop_incomplete_fragments
                            and "incomplete_vietnamese_fragment" in findings
                        )
                        if should_drop:
                            actual = "drop"
                            reason = "content quality filtering policy"
                    report.append({
                        "caption_index": index,
                        "timestamp": f"{segment.start:.3f} --> {segment.end:.3f}",
                        "original_text": text,
                        "score": score,
                        "findings": findings,
                        "recommended_action": recommended,
                        "actual_action": actual,
                        "reason": reason,
                    })
                    if actual == "keep":
                        kept_segments.append(segment)
                segments = kept_segments
                self.artifact_provider.write_json(
                    os.path.join(self.artifact_provider.debug_dir(job.job_id), "content_quality_report.json"),
                    {"captions": report},
                )
            vietnamese_report = []
            corrections_applied = 0
            for index, segment in enumerate(segments, start=1):
                text = segment.translated_text if normalize_translated else segment.text
                audit = VietnameseTextQualityAuditor.apply(
                    text, content_config.enable_vietnamese_text_correction
                )
                if audit["actual_action"] == "correct":
                    if normalize_translated:
                        segment.translated_text = audit["corrected_text"]
                    else:
                        segment.text = audit["corrected_text"]
                    corrections_applied += len(audit["corrections"])
                vietnamese_report.append({
                    "caption_index": index,
                    "timestamp": f"{segment.start:.3f} --> {segment.end:.3f}",
                    **audit,
                })
            for segment in narration_segments:
                text = segment.translated_text if normalize_translated else segment.text
                audit = VietnameseTextQualityAuditor.apply(
                    text, content_config.enable_vietnamese_text_correction
                )
                if audit["actual_action"] == "correct":
                    if normalize_translated:
                        segment.translated_text = audit["corrected_text"]
                    else:
                        segment.text = audit["corrected_text"]
            self.artifact_provider.write_json(
                os.path.join(
                    self.artifact_provider.debug_dir(job.job_id),
                    "vietnamese_text_quality_report.json",
                ),
                {
                    "captions": vietnamese_report,
                    "warnings_count": sum(
                        bool(item["corrections"]) for item in vietnamese_report
                    ),
                    "corrections_applied": corrections_applied,
                    "mode": (
                        "correction-enabled"
                        if content_config.enable_vietnamese_text_correction
                        else "report-first"
                    ),
                },
            )
            self.artifact_provider.write_json(
                self.artifact_provider.segmented_path(job.job_id),
                [s.to_dict() for s in segments]
            )

            job.segments = segments

            # Calculate metrics
            final_segments_count = len(segments)
            total_caption_duration = sum(s.end - s.start for s in segments)
            coverage_ratio = total_caption_duration / video_duration if video_duration > 0 else 0.0
            avg_caption_duration = total_caption_duration / final_segments_count if final_segments_count > 0 else 0.0

            gaps = []
            if final_segments_count > 0:
                gaps.append(segments[0].start)
                for i in range(1, final_segments_count):
                    gap = segments[i].start - segments[i-1].end
                    gaps.append(max(gap, 0.0))
                gaps.append(max(video_duration - segments[-1].end, 0.0))
            else:
                gaps.append(video_duration)
            longest_gap = max(gaps) if gaps else 0.0

            dropped_count = (len(speech_segments) + len(ocr_segments)) - final_segments_count

            quality_metrics.update({
                "video_duration": video_duration,
                "total_caption_duration": total_caption_duration,
                "caption_coverage_ratio": coverage_ratio,
                "number_of_final_captions": final_segments_count,
                "average_caption_duration": avg_caption_duration,
                "longest_gap_without_captions": longest_gap,
                "dropped_ocr_stt_segment_count": dropped_count,
                "grouping_examples": grouping_examples,
            })

            quality_config = self.normalize_segments_uc.config
            gate_failed = False
            gate_reasons = []

            if coverage_ratio < quality_config.min_coverage_ratio:
                gate_failed = True
                gate_reasons.append(f"Coverage ratio {coverage_ratio:.3f} below threshold {quality_config.min_coverage_ratio}")
            if longest_gap > quality_config.max_longest_gap:
                gate_failed = True
                gate_reasons.append(f"Longest gap {longest_gap:.3f} exceeds threshold {quality_config.max_longest_gap}")
            if final_segments_count < quality_config.min_caption_count:
                gate_failed = True
                gate_reasons.append(f"Caption count {final_segments_count} below threshold {quality_config.min_caption_count}")

            if normalize_translated:
                leak_examples = SourceTextLeakageDetector.detect(segments, job.source_language)
                quality_metrics["source_text_leakage_count"] = len(leak_examples)
                quality_metrics["source_text_leakage_examples"] = leak_examples

                max_leakage_allowed = quality_config.ocr_max_source_leakage
                if len(leak_examples) > max_leakage_allowed:
                    gate_failed = True
                    gate_reasons.append(f"Source leakage count {len(leak_examples)} exceeds threshold {max_leakage_allowed}")

            if gate_failed:
                quality_metrics["quality_gate_status"] = "FAIL"
                quality_metrics["gate_failure_reasons"] = gate_reasons
            else:
                quality_metrics["quality_gate_status"] = "PASS"

            if job.features.subtitle:
                started = time.monotonic()
                self.state_machine.transition(job, JobStatus.GENERATING_SUBTITLE)
                logger = FeatureLogger("Subtitle")
                logger.start()
                replacement_required = (
                                    job.features.subtitle_placement
                                    and self.calculate_subtitle_placement_uc.placement_service.config.placement_mode
                                    == "replace_source_subtitle"
                                )
                display_segments = (
                    [segment for segment in segments if segment.source == "ocr"]
                    if replacement_required
                    else segments
                )
                if replacement_required and not display_segments:
                    raw_filtered_count = len(ocr_segments) if 'ocr_segments' in locals() else 0
                    translated_count = source_counts_before.get("ocr", 0) if 'source_counts_before' in locals() else 0
                    normalized_count = sum(s.source == "ocr" for s in segments)
                    display_count = len(display_segments)
                    quality_metrics["dynamic_replacement_status"] = (
                        "NO_ACCEPTED_SUBTITLE_INTERVALS"
                    )
                    quality_metrics["render_replacement_status"] = "NOT_RUN"
                    job.error_message = "NO_ACCEPTED_SUBTITLE_INTERVALS"
                    audit_logger.warning(
                        "LOUD-WARN: Subtitle placement requested (replacement_required=True) "
                        "but NO OCR subtitle segments survived the pipeline for job %s. "
                        "Stage counts (OCR): filtered=%d -> translated=%d -> normalized=%d -> display=%d. "
                        "Output will contain NO burned-in subtitle replacement.",
                        job.job_id,
                        raw_filtered_count,
                        translated_count,
                        normalized_count,
                        display_count,
                    )
                    self.artifact_provider.write_json(
                        self.artifact_provider.final_timeline_path(job.job_id),
                        self.final_timeline_builder.build(
                            narration_segments, [], regions, [], True
                        ).to_dict(),
                    )
                    logger.done()
                    timings["placement"] = round(time.monotonic() - started, 3)
                    if controls.run_render:
                        return job
                if replacement_required and display_segments and regions:
                    region_start = min(r.start_time for r in regions)
                    region_end = max(r.end_time for r in regions)
                    first_seg = min(display_segments, key=lambda s: s.start)
                    last_seg = max(display_segments, key=lambda s: s.end)
                    if region_start < first_seg.start:
                        first_seg.start = region_start
                    if region_end > last_seg.end:
                        last_seg.end = region_end
                placements = self.calculate_subtitle_placement_uc.execute(
                    regions, display_segments, self.subtitle_style
                )
                placement_report = self.calculate_subtitle_placement_uc.quality_report(
                    regions, display_segments, placements, self.subtitle_style
                )
                self.artifact_provider.write_json(
                    self.artifact_provider.placement_quality_report_path(job.job_id),
                    placement_report,
                )
                if not os.path.exists(
                    self.artifact_provider.placement_quality_report_path(job.job_id)
                ):
                    raise ValidationError("placement_quality_report.json is missing.")
                final_timeline = self.final_timeline_builder.build(
                    narration_segments,
                    display_segments,
                    regions,
                    placements,
                    replacement_required,
                )
                job.final_timeline_id = final_timeline.timeline_id
                self.artifact_provider.write_json(
                    self.artifact_provider.final_timeline_path(job.job_id),
                    final_timeline.to_dict(),
                )
                srt_content = self.generate_subtitle_uc.execute(display_segments)
                job.subtitle_content = srt_content
                job.subtitle_path = self.artifact_provider.subtitle_path(job.job_id)
                self.artifact_provider.write_text(job.subtitle_path, srt_content)
                ass_content = self.generate_subtitle_uc.execute(
                                    display_segments,
                                    subtitle_format="ass",
                                    style=self.subtitle_style,
                                    placements=(
                                        placements if job.features.subtitle_placement and regions else None
                                    ),
                                    play_res_x=video_width,
                                    play_res_y=video_height,
                                    mask_regions=regions if replacement_required else None,
                                    decision_store=self.mask_decision_store,
                                )
                self.artifact_provider.write_text(
                    self.artifact_provider.subtitle_ass_path(job.job_id), ass_content
                )
                logger.done()
                timings["placement"] = round(time.monotonic() - started, 3)

            if controls.run_render and (job.features.render_video or job.features.dubbing):
                started = time.monotonic()
                if not job.features.subtitle:
                    raise ValidationError("Cannot render video without subtitles enabled.")
                ass_path = self.artifact_provider.subtitle_ass_path(job.job_id)
                if not os.path.exists(ass_path):
                    raise ValidationError("Subtitle ASS file does not exist. Cannot render video.")
                self.state_machine.transition(job, JobStatus.RENDERING_VIDEO)
                logger = FeatureLogger("RenderVideo")
                logger.start()
                job.output_video_path = self.render_video_uc.execute(
                    video_path=job.input_video,
                    subtitle_path=ass_path,
                    output_path=job.output_video_path
                    or self.artifact_provider.video_output_path(job.job_id),
                )
                logger.done()
                timings["render"] = round(time.monotonic() - started, 3)

            if job.features.dubbing and controls.run_tts and controls.run_mux:
                if self.generate_dubbed_audio_uc is None or self.mux_dubbed_video_uc is None:
                    raise ValidationError("Dubbing feature is not configured.")
                self.state_machine.transition(job, JobStatus.GENERATING_DUBBED_AUDIO)
                started = time.monotonic()
                job.dubbed_audio_path = self.generate_dubbed_audio_uc.execute(
                    job.job_id,
                    final_timeline.narration,
                    video_duration,
                    job.tts_config,
                    timeline_id=final_timeline.timeline_id,
                )
                timings["tts"] = round(time.monotonic() - started, 3)
                self.state_machine.transition(job, JobStatus.MUXING_DUBBED_VIDEO)
                started = time.monotonic()
                job.dubbed_video_path = self.mux_dubbed_video_uc.execute(
                    job.job_id,
                    job.output_video_path,
                    job.dubbed_audio_path,
                    job.tts_config,
                )
                timings["mux"] = round(time.monotonic() - started, 3)
                final_manifest = self.final_render_manifest_service.build(
                    job.job_id,
                    final_timeline.timeline_id,
                    {
                        "subtitle_srt": job.subtitle_path,
                        "subtitle_ass": self.artifact_provider.subtitle_ass_path(job.job_id),
                        "placement_quality_report": self.artifact_provider.placement_quality_report_path(job.job_id),
                        "rendered_video": job.output_video_path,
                        "tts_manifest": self.artifact_provider.tts_manifest_path(job.job_id),
                        "dubbed_audio": job.dubbed_audio_path,
                        "dubbed_video": job.dubbed_video_path,
                    },
                )
                self.final_render_manifest_service.validate(
                    final_manifest, final_timeline.timeline_id
                )
                job.final_render_manifest_path = (
                    self.artifact_provider.final_render_manifest_path(job.job_id)
                )
                self.artifact_provider.write_json(
                    job.final_render_manifest_path, final_manifest
                )

            if (
                job.features.subtitle_placement_debug
                and self.generate_subtitle_placement_debug_uc is not None
                and frames
            ):
                self.generate_subtitle_placement_debug_uc.execute(
                    frames=frames,
                    regions=regions,
                    placements=placements,
                    before_path=self.artifact_provider.debug_before_path(job.job_id),
                    overlay_path=self.artifact_provider.debug_overlay_path(job.job_id),
                    after_path=self.artifact_provider.debug_after_path(job.job_id),
                    regions_path=self.artifact_provider.debug_regions_json_path(job.job_id),
                    placements_path=self.artifact_provider.debug_placements_json_path(
                        job.job_id
                    ),
                    rendered_video_path=job.output_video_path or None,
                )
            quality_metrics["stage_timings_seconds"] = timings
            for stage, elapsed in timings.items():
                print(
                    f"[Timing] Stage {stage} completed in {elapsed:.3f}s",
                    file=sys.stderr,
                )
            self._write_quality_metrics(job.job_id, quality_metrics)
            self._write_provider_provenance(job.job_id)

            if quality_metrics["quality_gate_status"] == "FAIL":
                raise ValidationError(
                    f"Quality Gate Failed: {', '.join(gate_reasons)}"
                )

            self.state_machine.complete(job)
        except Exception as e:
            self.state_machine.fail(job, str(e))
            raise
        return job

    def _write_quality_metrics(self, job_id: str, quality_metrics: dict) -> None:
        path = self.artifact_provider.debug_metadata_json_path(job_id)
        metadata = {}
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as file:
                try:
                    metadata = json.load(file)
                except Exception:
                    metadata = {}
        metadata["quality_metrics"] = quality_metrics

        # Gather provider provenance
        from typing import Any
        provenance_data = {}
        for category, uc_name in [("ocr", "detect_onscreen_text_uc"), ("speech", "transcribe_speech_uc")]:
            uc = getattr(self, uc_name, None)
            if uc is not None:
                provider = getattr(uc, "ocr_provider" if category == "ocr" else "speech_provider", None)
                if provider is not None:
                    provenance_data[category] = self._extract_provider_provenance(provider)
        metadata["provenance"] = provenance_data

        self.artifact_provider.write_json(path, metadata)

    def _write_gemini_multistage_ocr_artifacts(self, job_id: str) -> None:
        provider = self.detect_onscreen_text_uc.ocr_provider
        if provider.__class__.__name__ != "GeminiMultiStageOCRProvider":
            return
        for name, artifact in provider.debug_artifacts().items():
            self.artifact_provider.write_json(
                os.path.join(self.artifact_provider.debug_dir(job_id), f"g3_{name}.json"),
                artifact,
            )

    def _write_gemini_direct_video_ocr_artifacts(self, job_id: str) -> None:
        provider = self.detect_onscreen_text_uc.ocr_provider
        if provider.__class__.__name__ != "GeminiDirectVideoOCRProvider":
            return
        names = {
            "video_upload": "g5_video_upload.json",
            "fps_smoke_test": "g5_fps_smoke_test.json",
            "candidates": "g5_candidates.json",
            "classifications": "g5_classifications.json",
            "temporal_events": "g5_temporal_events.json",
            "geometry": "g5_geometry.json",
            "transcriptions": "g5_transcriptions.json",
            "translations": "g5_translations.json",
            "checkpoints": "g5_checkpoints.json",
            "quota_failure": "g5_quota_failure.json",
        }
        for key, filename in names.items():
            self.artifact_provider.write_json(
                os.path.join(self.artifact_provider.debug_dir(job_id), filename),
                provider.debug_artifacts().get(key, {}),
            )
        manifest = list(getattr(provider, "provenance", []))
        self.artifact_provider.write_json(
            os.path.join(self.artifact_provider.debug_dir(job_id), "g5_request_manifest.json"),
            manifest,
        )
        self.artifact_provider.write_json(
            os.path.join(self.artifact_provider.debug_dir(job_id), "g5_usage.json"),
            [{"stage": item.get("stage"), "actual_usage": item.get("actual_usage", {})} for item in manifest],
        )
        plan = provider.debug_artifacts().get("checkpoints", {}).get("request_plan")
        if plan:
            self.artifact_provider.write_json(
                os.path.join(
                    self.artifact_provider.debug_dir(job_id), "g5_request_plan_v2.json"
                ),
                plan,
            )
        resume_path = getattr(provider, "_resume_manifest_path", None)
        if resume_path and resume_path.is_file():
            self.artifact_provider.write_json(
                os.path.join(
                    self.artifact_provider.debug_dir(job_id), "g5_resume_manifest.json"
                ),
                json.loads(resume_path.read_text(encoding="utf-8")),
            )

    def _extract_provider_provenance(self, provider: object) -> dict:
        info = {
            "configured_provider": provider.__class__.__name__,
            "effective_provider": provider.__class__.__name__,
            "fallback_used": False,
        }

        is_fallback_wrapper = provider.__class__.__name__ in ("FallbackOCRProvider", "FallbackSpeechProvider")
        if is_fallback_wrapper:
            info.update(getattr(provider, "provenance", {}))
            primary_provider = getattr(provider, "primary", None)
        else:
            primary_provider = provider

        if primary_provider is not None:
            class_name = primary_provider.__class__.__name__
            if class_name in ("GeminiVisionOCRProvider", "GeminiAudioSpeechProvider"):
                gemini_prov = getattr(primary_provider, "provenance", [])
                validation_records = getattr(primary_provider, "validation", [])

                runtime = getattr(primary_provider, "runtime", None)
                totals = {}
                if runtime is not None and getattr(runtime, "usage_recorder", None) is not None:
                    totals = runtime.usage_recorder.totals()

                request_count = len(gemini_prov)
                escalation_count = getattr(primary_provider, "_escalation_requests", 0)

                cache_hits = sum(1 for req in gemini_prov if req.get("cache_hit") is True)
                cache_misses = request_count - cache_hits

                model_ids = list(set(req.get("model") for req in gemini_prov if req.get("model")))

                chunk_count = sum(1 for req in gemini_prov if "chunk_start" in req)

                info.update({
                    "exact_model_id": model_ids[0] if model_ids else getattr(primary_provider, "model", None),
                    "model_role": "primary",
                    "request_count": request_count,
                    "retry_count": len(validation_records),
                    "cache_hit_count": cache_hits,
                    "cache_miss_count": cache_misses,
                    "chunk_count": chunk_count,
                    "escalation_count": escalation_count,
                    "token_usage": totals,
                })
        return info

    def _write_provider_provenance(self, job_id: str) -> None:
        path = self.artifact_provider.debug_metadata_json_path(job_id)
        metadata = self.artifact_provider.read_json(path) if os.path.exists(path) else {}
        metadata["provider_provenance"] = {
            "ocr": self._provider_provenance(self.detect_onscreen_text_uc.ocr_provider),
            "speech": self._provider_provenance(self.transcribe_speech_uc.speech_provider),
        }
        self.artifact_provider.write_json(path, metadata)

    def _ocr_segments_for_confirmed_tracks(
        self,
        regions: list[SubtitleRegion],
        frames: list[Frame],
        video_width: int,
        video_height: int,
    ) -> list[Segment]:
        """Read text only for geometry that already has a positive track decision."""
        if not regions:
            return []
        provider = self.detect_onscreen_text_uc.ocr_provider
        read_track = getattr(provider, "read_confirmed_tracks", None)
        if not callable(read_track):
            audit_logger.warning(
                "LOUD-WARN: configured OCR provider cannot read confirmed tracks; "
                "no OCR subtitle segments will be rendered."
            )
            return []
        return read_track(regions, frames, video_width, video_height)

    @staticmethod
    def _provider_provenance(provider: object) -> dict:
        raw_record = getattr(provider, "provenance", None)
        if raw_record is None:
            record = {}
        elif isinstance(raw_record, dict):
            record = dict(raw_record)
        elif isinstance(raw_record, list):
            if not all(isinstance(item, dict) for item in raw_record):
                raise TypeError("Provider provenance entries must be dictionaries.")
            record = {"request_provenance": [dict(item) for item in raw_record]}
        else:
            raise TypeError("Provider provenance must be a dictionary or list of dictionaries.")
        primary = getattr(provider, "primary", provider)
        requests = getattr(primary, "provenance", [])
        if isinstance(requests, list):
            record.setdefault("configured_provider", primary.__class__.__name__)
            record.setdefault("effective_provider", primary.__class__.__name__)
            record.setdefault("fallback_used", False)
            record["request_count"] = len(requests)
            record["retry_count"] = len(getattr(primary, "validation", []))
            record["cache_hit_count"] = sum(bool(item.get("cache_hit")) for item in requests)
            record["cache_miss_count"] = record["request_count"] - record["cache_hit_count"]
            record["chunk_count"] = sum("chunk_start" in item for item in requests)
            record["escalation_count"] = getattr(primary, "_escalation_requests", 0)
            record["exact_model_id"] = getattr(primary, "model", None)
        return record

    @staticmethod
    def _segment_fields(item: dict) -> dict:
        return {
            key: item[key]
            for key in (
                "id",
                "start",
                "end",
                "text",
                "translated_text",
                "source",
                "confidence",
                "confidence_available",
                "bbox",
                "metadata",
            )
            if key in item
        }
