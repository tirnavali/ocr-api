"""Unit tests for durable job queue state transitions, retries, and idempotency."""

from __future__ import annotations

from pathlib import Path

import pytest

from ocr_service.config import Settings
from ocr_service.domain.models import JobStatus, JobSubmissionOptions, can_transition
from ocr_service.exceptions import JobStateConflictError, QueueBackpressureError
from ocr_service.infrastructure.db.database import DatabaseSessionManager
from ocr_service.infrastructure.db.repository import JobRepository


def test_valid_and_invalid_state_transitions() -> None:
    assert can_transition(JobStatus.QUEUED, JobStatus.PROCESSING) is True
    assert can_transition(JobStatus.PROCESSING, JobStatus.COMPLETED) is True
    assert can_transition(JobStatus.PROCESSING, JobStatus.QUEUED) is True  # retry / crash recovery
    assert can_transition(JobStatus.COMPLETED, JobStatus.EXPIRED) is True
    assert can_transition(JobStatus.EXPIRED, JobStatus.PROCESSING) is False
    assert can_transition(JobStatus.COMPLETED, JobStatus.CANCELLED) is False


def test_repository_enqueue_claim_retry_and_complete(test_settings: Settings, tmp_path: Path) -> None:
    db = DatabaseSessionManager(test_settings)
    repo = JobRepository(db)
    dummy_file = tmp_path / "doc.jpg"
    dummy_file.write_bytes(b"test")

    job = repo.enqueue_job(
        job_id="job-test-lifecycle-1",
        options=JobSubmissionOptions(),
        input_filenames=["doc.jpg"],
        input_paths=[dummy_file],
        content_sha256="abc123sha",
        idempotency_key="idem-key-1",
        correlation_id="corr-1",
        max_retries=1,
        max_queue_depth=10,
    )
    assert job.status == JobStatus.QUEUED

    # Verify idempotency lookup returns the queued job
    idem_hit = repo.find_idempotent_job("idem-key-1", "abc123sha")
    assert idem_hit is not None
    assert idem_hit.job_id == "job-test-lifecycle-1"

    # Worker claims job (attempt 1)
    claimed = repo.claim_next_job(
        worker_id="w1", lease_seconds=60, max_concurrent_jobs=4
    )
    assert claimed is not None
    assert claimed.status == JobStatus.PROCESSING
    assert claimed.attempt_count == 1

    # Simulate transient failure on attempt 1 (max_retries=1 -> should re-queue!)
    retried = repo.mark_failed_or_retry(
        job_id=job.job_id,
        error_code="OCR_ENGINE_FAILURE",
        error_message="Temporary timeout",
        retryable=True,
        failed_retention_hours=12.0,
        backoff_base_seconds=0.0,
    )
    assert retried.status == JobStatus.QUEUED
    assert retried.retryable is True

    # Worker claims job again (attempt 2)
    claimed2 = repo.claim_next_job(
        worker_id="w1", lease_seconds=60, max_concurrent_jobs=4
    )
    assert claimed2 is not None
    assert claimed2.attempt_count == 2

    # Complete job on attempt 2
    completed = repo.mark_completed(
        job_id=job.job_id,
        page_count=1,
        processing_duration_ms=250,
        retention_hours=24.0,
        pdfa_compliant=True,
        pdfa_validator="structural",
    )
    assert completed.status == JobStatus.COMPLETED
    assert completed.pdfa_compliant is True

    # Cannot cancel an already completed job
    with pytest.raises(JobStateConflictError):
        repo.request_cancel(job.job_id)

    db.dispose()


def test_permanent_failure_does_not_retry(test_settings: Settings, tmp_path: Path) -> None:
    db = DatabaseSessionManager(test_settings)
    repo = JobRepository(db)
    dummy_file = tmp_path / "corrupt.pdf"
    dummy_file.write_bytes(b"bad")

    job = repo.enqueue_job(
        job_id="job-perm-fail-1",
        options=JobSubmissionOptions(),
        input_filenames=["corrupt.pdf"],
        input_paths=[dummy_file],
        content_sha256="badsha",
        idempotency_key=None,
        correlation_id="corr-2",
        max_retries=3,
        max_queue_depth=10,
    )
    repo.claim_next_job(worker_id="w1", lease_seconds=60, max_concurrent_jobs=4)

    failed = repo.mark_failed_or_retry(
        job_id=job.job_id,
        error_code="CORRUPTED_DOCUMENT",
        error_message="Invalid PDF xref table",
        retryable=False,
        failed_retention_hours=12.0,
    )
    assert failed.status == JobStatus.FAILED
    assert failed.attempt_count == 1
    db.dispose()


def test_queue_backpressure_limit(test_settings: Settings, tmp_path: Path) -> None:
    db = DatabaseSessionManager(test_settings)
    repo = JobRepository(db)
    f = tmp_path / "f.jpg"
    f.write_bytes(b"123")

    repo.enqueue_job(
        job_id="job-bp-1",
        options=JobSubmissionOptions(),
        input_filenames=["f.jpg"],
        input_paths=[f],
        content_sha256="h1",
        idempotency_key=None,
        correlation_id="c1",
        max_retries=1,
        max_queue_depth=1,
    )

    with pytest.raises(QueueBackpressureError):
        repo.enqueue_job(
            job_id="job-bp-2",
            options=JobSubmissionOptions(),
            input_filenames=["f.jpg"],
            input_paths=[f],
            content_sha256="h2",
            idempotency_key=None,
            correlation_id="c2",
            max_retries=1,
            max_queue_depth=1,
        )
    db.dispose()
