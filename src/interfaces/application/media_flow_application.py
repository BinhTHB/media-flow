import os
from typing import Any
from src.application.orchestrator.job_orchestrator import JobOrchestrator
from src.domain.entities.feature_config import FeatureConfig
from src.domain.entities.video_job import VideoJob
from src.domain.entities.tts_config import TTSConfig
from src.domain.entities.stage_control_config import StageControlConfig
from src.interfaces.composition.application_context import ApplicationContext

class MediaFlowApplication:
    def __init__(self, orchestrator: JobOrchestrator, context: ApplicationContext):
        self.orchestrator = orchestrator
        self.context = context

    def run(self, job: VideoJob) -> VideoJob:
        if self.context.gemini_runtime is not None:
            self.context.gemini_runtime.set_job_id(job.job_id)
        return self.orchestrator.execute(job)

    def run_video(
        self,
        video_path: str,
        job_id: str = "job_demo_123",
        source_language: str = "zh",
        speech_language: str = "auto",
        target_language: str = "vi",
        speech_to_text: bool = False,
        ocr: bool = False,
        translate: bool = False,
        subtitle: bool = False,
        subtitle_placement: bool = False,
        subtitle_placement_debug: bool = False,
        render_video: bool = False,
        dubbing: bool = False,
        output_video_path: str = "",
    ) -> VideoJob:
        job = VideoJob(
            job_id=job_id,
            input_video=video_path,
            source_language=source_language,
            speech_language=speech_language,
            target_language=target_language,
            features=FeatureConfig(
                speech_to_text=speech_to_text,
                ocr=ocr,
                translate=translate,
                subtitle=subtitle,
                subtitle_placement=subtitle_placement,
                subtitle_placement_debug=subtitle_placement_debug,
                render_video=render_video,
                dubbing=dubbing,
            ),
            output_video_path=output_video_path,
        )
        return self.run(job)

    def run_cli(self, args: Any) -> VideoJob:
        features = FeatureConfig(
            speech_to_text=args.stt,
            ocr=args.ocr,
            translate=args.translate,
            subtitle=args.subtitle,
            subtitle_placement=getattr(args, "subtitle_placement", False),
            subtitle_placement_debug=getattr(args, "subtitle_placement_debug", False),
            render_video=args.render,
            dubbing=getattr(args, "dubbing", False),
        )
        job = VideoJob(
            job_id=args.job_id,
            input_video=args.video_path,
            source_language=args.source_lang,
            speech_language=args.speech_lang,
            target_language=args.target_lang,
            features=features,
            output_video_path=args.render_output or "",
            tts_config=TTSConfig(
                voice=getattr(args, "tts_voice", "vi-VN-HoaiMyNeural"),
                rate=getattr(args, "tts_rate", "+0%"),
                original_audio_mode=(
                    "keep"
                    if getattr(args, "preserve_original_audio", False)
                    else getattr(args, "original_audio_mode", "lower")
                ),
                original_audio_volume=(
                    1.0
                    if getattr(args, "preserve_original_audio", False)
                    else getattr(args, "original_audio_volume", 0.2)
                ),
            ),
            stage_controls=StageControlConfig(
                run_asr=not getattr(args, "rerender_only", False),
                run_ocr=not getattr(args, "skip_ocr", False)
                and not getattr(args, "rerender_only", False),
                run_subtitle_region_detection=not getattr(
                    args, "skip_subtitle_region_detection", False
                )
                and not getattr(args, "rerender_only", False),
                run_translation=not getattr(args, "rerender_only", False),
                run_normalization=not getattr(args, "rerender_only", False),
                run_segmentation=(
                    not getattr(args, "rerender_only", False)
                    or getattr(args, "resegment", False)
                )
                and not getattr(args, "disable_segmentation", False),
                reuse_ocr_cache=getattr(args, "reuse_ocr_cache", False),
                reuse_subtitle_region_cache=getattr(
                    args, "reuse_subtitle_region_cache", False
                ),
                debug_from_artifacts=getattr(args, "debug_from_artifacts", ""),
            ),
        )
        if args.render:
            render_output = job.output_video_path or self.context.artifact_manager.video_path(args.job_id)
            if os.path.exists(render_output) and not args.overwrite:
                raise FileExistsError(f"Render output already exists: {render_output}")
        return self.run(job)
