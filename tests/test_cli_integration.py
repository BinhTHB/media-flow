import os
import subprocess
import json
import pytest

def test_cli_mock_flow():
    # Run the demo with mock flow
    cmd = [
        "python", "src/interfaces/cli/demo.py", "dummy_video.mp4",
        "--stt", "--translate", "--subtitle",
        "--video-provider", "mock",
        "--speech-provider", "mock",
        "--translation-provider", "mock",
        "--subtitle-provider", "mock",
        "--output-dir", "test_output",
        "--job-id", "job_test_1"
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, f"CLI execution failed: {res.stderr}"
    
    # Parse json output
    data = json.loads(res.stdout)
    assert data["job_id"] == "job_test_1"
    assert len(data["segments"]) > 0
    assert data["status"] == "completed"
    
    # Verify outputs written in test_output/job_test_1
    assert os.path.exists("test_output/job_test_1/audio/audio.wav")
    assert os.path.exists("test_output/job_test_1/transcripts/transcript.json")
    assert os.path.exists("test_output/job_test_1/transcripts/translated.json")
    assert os.path.exists("test_output/job_test_1/subtitles/subtitle.srt")
    
    # Clean up
    for f in ["audio/audio.wav", "transcripts/transcript.json", "transcripts/translated.json", "subtitles/subtitle.srt"]:
        try:
            os.remove(f"test_output/job_test_1/{f}")
        except OSError:
            pass
    for d in ["audio", "transcripts", "subtitles", "videos", "logs"]:
        try:
            os.rmdir(f"test_output/job_test_1/{d}")
        except OSError:
            pass
    try:
        os.rmdir("test_output/job_test_1")
    except OSError:
        pass
    try:
        os.rmdir("test_output")
    except OSError:
        pass
