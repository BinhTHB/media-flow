from abc import ABC, abstractmethod


class TTSProvider(ABC):
    @abstractmethod
    def synthesize(
        self,
        text: str,
        output_path: str,
        voice: str,
        rate: str,
    ) -> str:
        pass
