import os
import subprocess
import json
import pytest

def test_cli_mock_flow(tmp_path):
    output_dir = tmp_path / "test_output"
    # Run the demo with mock flow
    cmd = [
        "python", "src/interfaces/cli/demo.py", "dummy_video.mp4",
        "--stt", "--translate", "--subtitle",
        "--video-provider", "mock",
        "--speech-provider", "mock",
        "--translation-provider", "mock",
        "--subtitle-provider", "mock",
        "--ocr-provider", "mock",
        "--frame-provider", "mock",
        "--output-dir", str(output_dir),
        "--job-id", "job_test_1"
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, f"CLI execution failed: {res.stderr}"
    
    # Parse json output
    data = json.loads(res.stdout)
    assert data["job_id"] == "job_test_1"
    assert len(data["segments"]) > 0
    assert data["status"] == "completed"
    
    assert (output_dir / "job_test_1" / "audio" / "audio.wav").exists()
    assert (output_dir / "job_test_1" / "transcripts" / "transcript.json").exists()
    assert (output_dir / "job_test_1" / "transcripts" / "translated.json").exists()
    assert (output_dir / "job_test_1" / "subtitles" / "subtitle.srt").exists()
