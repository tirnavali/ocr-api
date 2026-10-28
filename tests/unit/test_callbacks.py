"""Unit tests for the signed completion-callback outbox."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from ocr_service.config import Settings
from ocr_service.domain.models import CallbackStatus, JobSubmissionOptions
from ocr_service.exceptions import ValidationError
from ocr_service.infrastructure.callbacks.notifier import (
    SIGNATURE_HEADER,
    TIMESTAMP_HEADER,
    CallbackNotifier,
    sign_payload,
    validate_callback_url,
)
from ocr_service.infrastructure.db.database import DatabaseSessionManager
from ocr_service.infrastructure.db.repository import JobRepository

CALLBACK = "http://kahb-api.test/api/v1/Ocr/Sonuc"


@pytest.fixture()
def cb_settings(tmp_path: Path) -> Settings:
    return Settings(
        TEMP_STORAGE_PATH=tmp_path,
        CALLBACK_HMAC_SECRET="s3cret",
        CALLBACK_ALLOWED_PREFIXES="http://kahb-api.test/",
        CALLBACK_MAX_ATTEMPTS=3,
    )


def _enqueue(repo: JobRepository, job_id: str, callback_url: str | None) -> None:
    repo.enqueue_job(
        job_id=job_id,
        options=JobSubmissionOptions(),
        input_filenames=["a.jpg"],
        input_paths=[],
        content_sha256="sha",
        idempotency_key="idem-1",
        correlation_id="corr-cb",
        max_retries=1,
        max_queue_depth=10,
        callback_url=callback_url,
    )


def _complete(repo: JobRepository, job_id: str) -> None:
    claimed = repo.claim_next_job(worker_id="w", lease_seconds=60, max_concurrent_jobs=4)
    assert claimed is not None and claimed.job_id == job_id
    repo.mark_completed(
        job_id=job_id,
        page_count=3,
        processing_duration_ms=10,
        retention_hours=24.0,
        pdfa_compliant=True,
        pdfa_validator="structural",
    )


def test_validate_callback_url(cb_settings: Settings, tmp_path: Path) -> None:
    assert validate_callback_url(cb_settings, None) is None
    assert validate_callback_url(cb_settings, CALLBACK) == CALLBACK
    with pytest.raises(ValidationError, match="allowed prefix"):
        validate_callback_url(cb_settings, "http://169.254.169.254/latest/meta-data")
    disabled = Settings(TEMP_STORAGE_PATH=tmp_path)
    with pytest.raises(ValidationError) as exc_info:
        validate_callback_url(disabled, CALLBACK)
    assert exc_info.value.code == "CALLBACKS_DISABLED"


def test_prefix_without_secret_is_a_config_error(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="CALLBACK_HMAC_SECRET"):
        Settings(TEMP_STORAGE_PATH=tmp_path, CALLBACK_ALLOWED_PREFIXES="http://x/")


def test_callback_waits_until_terminal_then_is_delivered_signed(cb_settings: Settings) -> None:
    repo = JobRepository(DatabaseSessionManager(cb_settings))
    received: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        received.append(request)
        return httpx.Response(200)

    notifier = CallbackNotifier(cb_settings, repo, transport=httpx.MockTransport(handler))

    _enqueue(repo, "job-cb-complete-01", CALLBACK)
    assert repo.get_job("job-cb-complete-01").callback_status == CallbackStatus.WAITING
    assert notifier.deliver_due() == 0  # not terminal yet
    assert received == []

    _complete(repo, "job-cb-complete-01")
    assert repo.get_job("job-cb-complete-01").callback_status == CallbackStatus.PENDING

    assert notifier.deliver_due() == 1
    assert len(received) == 1
    req = received[0]
    body = req.content
    expected = sign_payload("s3cret", req.headers[TIMESTAMP_HEADER], body)
    assert req.headers[SIGNATURE_HEADER] == expected
    payload = json.loads(body)
    assert payload["job_id"] == "job-cb-complete-01"
    assert payload["status"] == "completed"
    assert payload["job_type"] == "ocr"
    assert payload["page_count"] == 3
    assert payload["pdf_url"] == "/api/v1/jobs/job-cb-complete-01/pdf"

    job = repo.get_job("job-cb-complete-01")
    assert job.callback_status == CallbackStatus.DELIVERED
    assert job.callback_attempts == 1
    assert notifier.deliver_due() == 0  # never sent twice


def test_failed_delivery_is_retried_with_backoff_then_abandoned(cb_settings: Settings) -> None:
    repo = JobRepository(DatabaseSessionManager(cb_settings))
    notifier = CallbackNotifier(
        cb_settings, repo, transport=httpx.MockTransport(lambda r: httpx.Response(500, text="down"))
    )
    _enqueue(repo, "job-cb-retry-0001", CALLBACK)
    _complete(repo, "job-cb-retry-0001")

    assert notifier.deliver_due() == 0
    job = repo.get_job("job-cb-retry-0001")
    assert job.callback_status == CallbackStatus.PENDING
    assert job.callback_attempts == 1
    assert "HTTP 500" in (job.callback_last_error or "")
    # Backoff: not due again immediately.
    assert notifier.deliver_due() == 0
    assert repo.get_job("job-cb-retry-0001").callback_attempts == 1

    # Simulate the backoff elapsing for the remaining attempts.
    for _ in range(2):
        with repo._db.session() as sess:
            from ocr_service.infrastructure.db.models import JobRecord, utcnow

            sess.get(JobRecord, "job-cb-retry-0001").callback_next_at = utcnow()
        notifier.deliver_due()

    job = repo.get_job("job-cb-retry-0001")
    assert job.callback_attempts == 3
    assert job.callback_status == CallbackStatus.FAILED


def test_jobs_without_callback_url_are_never_sent(cb_settings: Settings) -> None:
    repo = JobRepository(DatabaseSessionManager(cb_settings))
    calls: list[httpx.Request] = []
    notifier = CallbackNotifier(
        cb_settings, repo, transport=httpx.MockTransport(lambda r: calls.append(r) or httpx.Response(200))
    )
    _enqueue(repo, "job-cb-none-00001", None)
    _complete(repo, "job-cb-none-00001")
    assert repo.get_job("job-cb-none-00001").callback_status == CallbackStatus.NONE
    assert notifier.deliver_due() == 0
    assert calls == []
