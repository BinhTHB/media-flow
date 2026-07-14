import math
from typing import List
from src.application.ports.speech_provider import SpeechProvider
from src.domain.entities.segment import Segment
from src.shared.exceptions import ProviderError
from src.infrastructure.runtime.cuda_dll_bootstrap import register_cuda_dll_directories

class FasterWhisperSpeechProvider(SpeechProvider):
    def __init__(
        self,
        model_size: str = "base",
        device: str = "cpu",
        compute_type: str = "int8",
    ):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self._model = None

    def _get_model(self):
        if self._model is None:
            try:
                register_cuda_dll_directories()
                from faster_whisper import WhisperModel
                self._model = WhisperModel(
                    self.model_size,
                    device=self.device,
                    compute_type=self.compute_type,
                )
            except ImportError as e:
                raise ProviderError(
                    "faster-whisper is not installed. Install it with: pip install faster-whisper"
                ) from e
        return self._model

    def transcribe(self, audio_path: str, source_language: str) -> List[Segment]:
        model = self._get_model()
        lang = None if source_language == "auto" else source_language
        try:
            segments, _ = model.transcribe(audio_path, language=lang)
            result = []
            for idx, seg in enumerate(segments):
                probability = getattr(seg, "probability", None)
                if probability is None:
                    avg_logprob = getattr(seg, "avg_logprob", None)
                    probability = math.exp(avg_logprob) if avg_logprob is not None else 0.0
                result.append(Segment(
                    id=f"speech_{idx}",
                    start=seg.start,
                    end=seg.end,
                    text=seg.text,
                    source="speech",
                    confidence=max(0.0, min(float(probability), 1.0)),
                ))
            return result
        except Exception as e:
            raise ProviderError(f"Faster-Whisper transcription failed: {e}") from e
