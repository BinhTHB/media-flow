from dataclasses import dataclass

@dataclass
class SubtitleItem:
    index: int
    start: float
    end: float
    text: str

    def to_srt_format(self) -> str:
        def format_time(seconds: float) -> str:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = int(seconds % 60)
            millis = int((seconds - int(seconds)) * 1000)
            return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

        return f"{self.index}\n{format_time(self.start)} --> {format_time(self.end)}\n{self.text}\n"
