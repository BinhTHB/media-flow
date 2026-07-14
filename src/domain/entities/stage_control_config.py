from dataclasses import dataclass

from src.shared.exceptions import ValidationError


@dataclass
class StageControlConfig:
    run_asr: bool = True
    run_ocr: bool = True
    run_subtitle_region_detection: bool = True
    run_translation: bool = True
    run_normalization: bool = True
    run_segmentation: bool = True
    run_placement: bool = True
    run_render: bool = True
    run_tts: bool = True
    run_mux: bool = True
    reuse_ocr_cache: bool = False
    reuse_subtitle_region_cache: bool = False
    debug_from_artifacts: str = ""

    def __post_init__(self) -> None:
        for name, value in self.__dict__.items():
            if name != "debug_from_artifacts" and not isinstance(value, bool):
                raise ValidationError(f"StageControlConfig.{name} must be a boolean.")

    def to_dict(self) -> dict:
        return dict(self.__dict__)
