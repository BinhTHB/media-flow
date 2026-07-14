import pytest
import os
from src.infrastructure.registry.provider_registry import ProviderRegistry
from src.infrastructure.factory.provider_factory import ProviderFactory
from src.infrastructure.artifacts.artifact_manager import ArtifactManager
from src.application.orchestrator.job_state_machine import JobStateMachine
from src.domain.entities.video_job import VideoJob
from src.domain.entities.job_status import JobStatus

def test_provider_registry():
    registry = ProviderRegistry()
    assert "mock" in registry.names("video")
    assert "ffmpeg" in registry.names("video")
    assert "faster-whisper" in registry.names("speech")
    
    # Test registration
    class DummyProvider:
        pass
    registry.register("speech", "dummy", DummyProvider)
    assert registry.get("speech", "dummy") == DummyProvider

def test_provider_factory():
    config = {
        "defaults": {
            "video": "mock"
        },
        "providers": {
            "video": {
                "mock": {}
            }
        }
    }
    factory = ProviderFactory(config=config)
    video_p = factory.create_video_provider()
    assert video_p.__class__.__name__ == "MockVideoProvider"

def test_artifact_manager(tmp_path):
    manager = ArtifactManager(base_dir=str(tmp_path))
    job_id = "job_123"
    
    audio_p = manager.audio_path(job_id)
    assert audio_p.endswith(os.path.normpath("job_123/audio/audio.wav"))
    
    # Write json and read back
    data = {"hello": "world"}
    trans_p = manager.transcript_path(job_id)
    manager.write_json(trans_p, data)
    assert manager.read_json(trans_p) == data

def test_job_state_machine():
    job = VideoJob(job_id="j1", input_video="test.mp4")
    sm = JobStateMachine()
    
    assert job.status == JobStatus.CREATED
    sm.transition(job, JobStatus.RUNNING)
    assert job.status == JobStatus.RUNNING
    assert job.started_at > 0
    
    sm.complete(job)
    assert job.status == JobStatus.COMPLETED
    assert job.completed_at >= job.started_at
