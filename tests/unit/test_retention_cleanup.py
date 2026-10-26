"""Unit tests for temporary artifact retention, active-job protection, and disk quota checks (FR-09)."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pytest

from ocr_service.config import Settings
from ocr_service.domain.models import JobStatus, JobSubmissionOptions
from ocr_service.exceptions import StorageQuotaExceededError
from ocr_service.infrastructure.db.database import DatabaseSessionManager
from ocr_service.infrastructure.db.models import JobRecord, utcnow
from ocr_service.infrastructure.db.repository import JobRepository
from ocr_service.infrastructure.storage.local_storage import LocalJobStorage
from ocr_service.worker_main import RetentionAndRecoveryJanitor


def test_janitor_purges_expired_jobs_but_never_deletes_active_jobs(
    test_settings: Settings, tmp_path: Path
) -> None:
    db = DatabaseSessionManager(test_settings)
    repo = JobRepository(db)
    storage = LocalJobStorage(test_settings)
    janitor = RetentionAndRecoveryJanitor(test_settings, repo, storage)

    # 1. Create an expired completed job
    expired_id = "job-expired-0001"
    exp_paths = storage.get_job_paths(expired_id, create=True)
    exp_paths.output_pdf_path.write_bytes(b"%PDF-1.7 dummy")
    f1 = exp_paths.input_dir / "in.jpg"
    f1.write_bytes(b"img")

    repo.enqueue_job(
        job_id=expired_id,
        options=JobSubmissionOptions(),
        input_filenames=["in.jpg"],
        input_paths=[f1],
        content_sha256="sha1",
        idempotency_key=None,
        correlation_id="c1",
        max_retries=1,
        max_queue_depth=10,
    )
    repo.claim_next_job(worker_id="w1", lease_seconds=60, max_concurrent_jobs=4)
    repo.mark_completed(
        job_id=expired_id,
        page_count=1,
        processing_duration_ms=100,
        retention_hours=24.0,
        pdfa_compliant=True,
        pdfa_validator="structural",
    )
    # Force expires_at into the past
    with db.session() as sess:
        rec = sess.get(JobRecord, expired_id)
        assert rec is not None
        rec.expires_at = utcnow() - timedelta(minutes=5)

    # 2. Create an active processing job
    active_id = "job-active-0002"
    act_paths = storage.get_job_paths(active_id, create=True)
    f2 = act_paths.input_dir / "active.jpg"
    f2.write_bytes(b"active_img_bytes")
    repo.enqueue_job(
        job_id=active_id,
        options=JobSubmissionOptions(),
        input_filenames=["active.jpg"],
        input_paths=[f2],
        content_sha256="sha2",
        idempotency_key=None,
        correlation_id="c2",
        max_retries=1,
        max_queue_depth=10,
    )
    repo.claim_next_job(worker_id="w2", lease_seconds=300, max_concurrent_jobs=4)

    # Run Janitor
    summary = janitor.run_maintenance_cycle()
    assert summary["expired_jobs"] == 1

    # Verify expired job's disk directory was purged, but DB metadata remains with status='expired'
    assert not exp_paths.job_dir.exists()
    expired_entity = repo.get_job(expired_id)
    assert expired_entity.status == JobStatus.EXPIRED
    assert "expired_at" in expired_entity.metadata_json

    # Verify active job's files and status were NOT touched
    assert act_paths.job_dir.exists()
    assert f2.exists()
    active_entity = repo.get_job(active_id)
    assert active_entity.status == JobStatus.PROCESSING

    db.dispose()


def test_storage_quota_enforcement(tmp_path: Path) -> None:
    # Configure tiny quota
    settings = Settings(
        TEMP_STORAGE_PATH=tmp_path / "quota_test",
        MAX_DISK_USAGE_GB=0.1,  # ~100 MB
    )
    storage = LocalJobStorage(settings)
    with pytest.raises(StorageQuotaExceededError):
        storage.check_storage_capacity(incoming_bytes=200 * 1024 * 1024)
