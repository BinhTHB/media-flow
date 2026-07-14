import time
from src.domain.entities.job_status import JobStatus
from src.domain.entities.video_job import VideoJob

class JobStateMachine:
    def transition(self, job: VideoJob, status: JobStatus | str) -> VideoJob:
        next_status = JobStatus(status)
        if next_status == JobStatus.RUNNING and not job.started_at:
            job.started_at = time.time()
        job.status = next_status
        return job

    def complete(self, job: VideoJob) -> VideoJob:
        job.status = JobStatus.COMPLETED
        job.completed_at = time.time()
        return job

    def fail(self, job: VideoJob, error_message: str) -> VideoJob:
        job.status = JobStatus.FAILED
        job.error_message = error_message
        job.completed_at = time.time()
        return job

    def skip(self, job: VideoJob) -> VideoJob:
        job.status = JobStatus.SKIPPED
        job.completed_at = time.time()
        return job
