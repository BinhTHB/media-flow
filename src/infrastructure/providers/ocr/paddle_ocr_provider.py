import os
import json
import time
from collections.abc import Sequence
from typing import List
from src.application.ports.ocr_provider import OCRProvider
from src.domain.entities.frame import Frame
from src.domain.entities.segment import Segment
from src.shared.exceptions import ProviderError
from src.infrastructure.runtime.ocr_device_policy import validate_ocr_device

class PaddleOCRProvider(OCRProvider):
    def __init__(self, language: str = "ch", use_gpu: bool = True):
        self.language = language
        self.use_gpu = use_gpu
        self._ocr = None

    def _get_ocr(self):
        if self._ocr is None:
            if self.use_gpu:
                validate_ocr_device("gpu")
            try:
                import paddleocr
            except ImportError as e:
                raise ProviderError(
                    "paddleocr is not installed. Install it with: pip install paddleocr"
                ) from e
            try:
                if int(getattr(paddleocr, "__version__", "2").split(".", 1)[0]) >= 3:
                    self._ocr = paddleocr.PaddleOCR(
                        lang=self.language,
                        device="gpu" if self.use_gpu else "cpu",
                        enable_mkldnn=False,
                        engine_config={"paddle_static": {"enable_new_ir": False}},
                    )
                else:
                    self._ocr = paddleocr.PaddleOCR(
                        use_angle_cls=True,
                        lang=self.language,
                        use_gpu=self.use_gpu,
                        show_log=False,
                    )
            except Exception as e:
                raise ProviderError(f"PaddleOCR initialization failed: {e}") from e
        return self._ocr

    def detect_text(self, frames: List[Frame]) -> List[Segment]:
        ocr = self._get_ocr()
        segments = []
        for idx, frame in enumerate(frames):
            # Check if parent directory is named "frames" to enable checkpointing
            parent_dir_name = os.path.basename(os.path.dirname(frame.image_path))
            use_checkpoint = (parent_dir_name == "frames")

            checkpoint_path = None
            frame_name = None
            if use_checkpoint:
                job_dir = os.path.dirname(os.path.dirname(frame.image_path))
                if not job_dir or job_dir == ".":
                    debug_dir = os.path.join("output", "debug")
                else:
                    debug_dir = os.path.join(job_dir, "debug")
                ocr_frames_dir = os.path.join(debug_dir, "ocr_frames")
                os.makedirs(ocr_frames_dir, exist_ok=True)
                
                frame_name = os.path.splitext(os.path.basename(frame.image_path))[0]
                checkpoint_path = os.path.join(ocr_frames_dir, f"{frame_name}.json")

                # Checkpoint resume/reuse
                if os.path.exists(checkpoint_path):
                    try:
                        with open(checkpoint_path, "r", encoding="utf-8") as f:
                            cp = json.load(f)
                        if "error" not in cp or cp["error"] is None:
                            for seg_dict in cp.get("segments", []):
                                segments.append(
                                    Segment(
                                        id=seg_dict["id"],
                                        start=seg_dict["start"],
                                        end=seg_dict["end"],
                                        text=seg_dict["text"],
                                        translated_text=seg_dict.get("translated_text", ""),
                                        source=seg_dict.get("source", "ocr"),
                                        confidence=seg_dict.get("confidence", 0.0),
                                    )
                                )
                            continue
                    except Exception:
                        pass

            start_time = time.monotonic()
            frame_segments = []
            error_msg = None
            duration = 0.0
            
            try:
                try:
                    result = ocr.ocr(frame.image_path, cls=True)
                except TypeError:
                    result = ocr.ocr(frame.image_path)

                lines = self._normalize_result(result)
                duration = time.monotonic() - start_time
                if lines:
                    for line_idx, line in enumerate(lines):
                        if (
                            not isinstance(line, Sequence)
                            or isinstance(line, (str, bytes))
                            or len(line) != 2
                        ):
                            raise ValueError("PaddleOCR detection returned an invalid line shape.")
                        box, (text, confidence) = line
                        seg = Segment(
                            id=f"ocr_{idx}_{line_idx}",
                            start=frame.timestamp,
                            end=frame.timestamp + 2.0,
                            text=text,
                            translated_text="",
                            source="ocr",
                            confidence=confidence,
                        )
                        frame_segments.append(seg)
                        segments.append(seg)
            except Exception as e:
                error_msg = str(e)
                duration = time.monotonic() - start_time
                if use_checkpoint and checkpoint_path:
                    # Write error checkpoint
                    checkpoint_data = {
                        "frame_id": frame_name,
                        "frame_path": frame.image_path,
                        "frame_timestamp": frame.timestamp,
                        "provider_name": "paddle",
                        "processing_duration": duration,
                        "error": error_msg,
                        "segments": [],
                    }
                    try:
                        with open(checkpoint_path, "w", encoding="utf-8") as f:
                            json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)
                    except Exception:
                        pass
                raise ProviderError(
                    f"PaddleOCR detection failed for frame {frame.image_path}: {e}"
                ) from e

            # Write successful checkpoint
            if use_checkpoint and checkpoint_path:
                checkpoint_data = {
                    "frame_id": frame_name,
                    "frame_path": frame.image_path,
                    "frame_timestamp": frame.timestamp,
                    "provider_name": "paddle",
                    "processing_duration": duration,
                    "error": None,
                    "segments": [s.to_dict() for s in frame_segments],
                }
                try:
                    with open(checkpoint_path, "w", encoding="utf-8") as f:
                        json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)
                except Exception:
                    pass

        return segments

    @staticmethod
    def _normalize_result(result):
        if not result:
            return []

        page = result[0]
        if isinstance(page, dict):
            return [
                [
                    polygon.tolist() if hasattr(polygon, "tolist") else polygon,
                    (text, float(confidence)),
                ]
                for polygon, text, confidence in zip(
                    page.get("rec_polys", []),
                    page.get("rec_texts", []),
                    page.get("rec_scores", []),
                )
            ]

        if (
            not isinstance(page, Sequence)
            or isinstance(page, (str, bytes))
        ):
            raise ValueError("PaddleOCR detection returned an invalid result shape.")
        return page