"""Unit tests for job types: per-type claiming, concurrency, backpressure, retention, schema upgrade."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from ocr_service.config import Settings
from ocr_service.domain.models import JobEntity, JobStatus, JobSubmissionOptions, JobType
from ocr_service.exceptions import QueueBackpressureError
from ocr_service.infrastructure.db.database import DatabaseSessionManager
from ocr_service.infrastructure.db.repository import JobRepository
from ocr_service.infrastructure.storage.local_storage import LocalJobStorage
from ocr_service.worker_main import OcrWorker


def _enqueue(repo: JobRepository, job_id: str, job_type: JobType, max_queue_depth: int = 10) -> JobEntity:
    return repo.enqueue_job(
        job_id=job_id,
        options=JobSubmissionOptions(),
        input_filenames=["x.jpg"],
        input_paths=[],
        content_sha256=f"sha-{job_id}",
        idempotency_key=None,
        correlation_id="corr",
        max_retries=1,
        max_queue_depth=max_queue_depth,
        job_type=job_type,
    )


def test_claim_only_returns_requested_job_types(test_settings: Settings) -> None:
    repo = JobRepository(DatabaseSessionManager(test_settings))
    _enqueue(repo, "job-type-ocr-0001", JobType.OCR)
    _enqueue(repo, "job-type-pkg-0001", JobType.PACKAGE)

    pkg = repo.claim_next_job(
        worker_id="w-pkg", lease_seconds=60, max_concurrent_jobs=4, job_types=["package"]
    )
    assert pkg is not None and pkg.job_id == "job-type-pkg-0001"
    assert pkg.job_type == JobType.PACKAGE

    # Default claim (no job_types) keeps the original OCR-only behaviour.
    ocr = repo.claim_next_job(worker_id="w-ocr", lease_seconds=60, max_concurrent_jobs=4)
    assert ocr is not None and ocr.job_id == "job-type-ocr-0001"
    assert repo.claim_next_job(worker_id="w-ocr", lease_seconds=60, max_concurrent_jobs=4) is None


def test_concurrency_cap_is_independent_per_job_type(test_settings: Settings) -> None:
    repo = JobRepository(DatabaseSessionManager(test_settings))
    _enqueue(repo, "job-cap-pkg-0001", JobType.PACKAGE)
    _enqueue(repo, "job-cap-pkg-0002", JobType.PACKAGE)
    _enqueue(repo, "job-cap-ocr-0001", JobType.OCR)
    caps = {"ocr": 4, "package": 1}
    both = ["ocr", "package"]

    first = repo.claim_next_job(
        worker_id="w1", lease_seconds=60, max_concurrent_jobs=4,
        job_types=both, max_concurrent_by_type=caps,
    )
    assert first is not None and first.job_id == "job-cap-pkg-0001"

    # Package cap (1) is reached, so the second package job is skipped in favour of OCR.
    second = repo.claim_next_job(
        worker_id="w2", lease_seconds=60, max_concurrent_jobs=4,
        job_types=both, max_concurrent_by_type=caps,
    )
    assert second is not None and second.job_id == "job-cap-ocr-0001"

    assert repo.claim_next_job(
        worker_id="w3", lease_seconds=60, max_concurrent_jobs=4,
        job_types=both, max_concurrent_by_type=caps,
    ) is None


def test_backpressure_is_counted_per_job_type(test_settings: Settings) -> None:
    repo = JobRepository(DatabaseSessionManager(test_settings))
    _enqueue(repo, "job-bp-ocr-0001", JobType.OCR, max_queue_depth=1)
    with pytest.raises(QueueBackpressureError):
        _enqueue(repo, "job-bp-ocr-0002", JobType.OCR, max_queue_depth=1)
    # A full OCR queue must not block other job types.
    accepted = _enqueue(repo, "job-bp-pkg-0001", JobType.PACKAGE, max_queue_depth=1)
    assert accepted.status == JobStatus.QUEUED


def test_retention_and_concurrency_settings_per_type(tmp_path: Path) -> None:
    settings = Settings(
        TEMP_STORAGE_PATH=tmp_path,
        RESULT_RETENTION_HOURS=24.0,
        PACKAGE_RETENTION_HOURS=168.0,
        MAX_CONCURRENT_JOBS=8,
        MAX_CONCURRENT_PACKAGE_JOBS=1,
        WORKER_JOB_TYPES="ocr, package",
    )
    assert settings.worker_job_types == ["ocr", "package"]
    assert settings.retention_hours_for("ocr") == 24.0
    assert settings.retention_hours_for("package") == 168.0
    assert settings.max_concurrent_jobs_for("ocr") == 8
    assert settings.max_concurrent_jobs_for("package") == 1


def test_unknown_worker_job_type_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="WORKER_JOB_TYPES"):
        Settings(TEMP_STORAGE_PATH=tmp_path, WORKER_JOB_TYPES="ocr,ftp")


class _RecordingHandler:
    def __init__(self, repo: JobRepository) -> None:
        self.repo = repo
        self.seen: list[str] = []

    def execute_job(self, job: JobEntity, worker_id: str) -> JobEntity:
        self.seen.append(job.job_id)
        return self.repo.mark_completed(
            job_id=job.job_id,
            page_count=0,
            processing_duration_ms=1,
            retention_hours=168.0,
            pdfa_compliant=True,
            pdfa_validator="n/a",
        )


def test_worker_pool_only_processes_its_own_job_types(tmp_path: Path) -> None:
    settings = Settings(TEMP_STORAGE_PATH=tmp_path, WORKER_JOB_TYPES="package")
    repo = JobRepository(DatabaseSessionManager(settings))
    storage = LocalJobStorage(settings)
    _enqueue(repo, "job-pool-ocr-0001", JobType.OCR)
    _enqueue(repo, "job-pool-pkg-0001", JobType.PACKAGE)

    handler = _RecordingHandler(repo)
    worker = OcrWorker(settings, repo, storage, worker_id="pkg-1", handlers={"package": handler})

    done = worker.process_next_available_job()
    assert done is not None and done.status == JobStatus.COMPLETED
    assert handler.seen == ["job-pool-pkg-0001"]
    assert worker.process_next_available_job() is None
    assert repo.get_job("job-pool-ocr-0001").status == JobStatus.QUEUED


def test_existing_database_gets_new_columns(tmp_path: Path) -> None:
    """A state DB created before job types existed is upgraded in place on startup."""
    db_file = tmp_path / "ocr_state.sqlite3"
    conn = sqlite3.connect(db_file)
    conn.execute(
        "CREATE TABLE ocr_jobs (job_id VARCHAR(64) PRIMARY KEY, status VARCHAR(32) NOT NULL)"
    )
    conn.execute("INSERT INTO ocr_jobs (job_id, status) VALUES ('job-legacy-00001', 'completed')")
    conn.commit()
    conn.close()

    settings = Settings(TEMP_STORAGE_PATH=tmp_path)
    DatabaseSessionManager(settings).dispose()
    # Second start must be a no-op (columns already present).
    DatabaseSessionManager(settings).dispose()

    conn = sqlite3.connect(db_file)
    columns = {row[1] for row in conn.execute("PRAGMA table_info(ocr_jobs)")}
    row = conn.execute(
        "SELECT job_type, callback_status, callback_attempts, source_refs_json "
        "FROM ocr_jobs WHERE job_id = 'job-legacy-00001'"
    ).fetchone()
    indexes = {row[1] for row in conn.execute("PRAGMA index_list(ocr_jobs)")}
    conn.close()

    assert {"job_type", "source_refs_json", "callback_url", "callback_status"} <= columns
    assert row == ("ocr", "none", 0, "[]")
    assert "ix_ocr_jobs_type_claim" in indexes
