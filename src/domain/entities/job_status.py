from enum import Enum

class JobStatus(str, Enum):
    CREATED = "created"
    RUNNING = "running"
    EXTRACTING_AUDIO = "extracting_audio"
    TRANSCRIBING = "transcribing"
    DETECTING_TEXT = "detecting_text"
    TRANSLATING = "translating"
    GENERATING_SUBTITLE = "generating_subtitle"
    RENDERING_VIDEO = "rendering_video"
    GENERATING_DUBBED_AUDIO = "generating_dubbed_audio"
    MUXING_DUBBED_VIDEO = "muxing_dubbed_video"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
