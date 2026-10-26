"""Integration tests for worker crash recovery, page checkpointing, cancellation, and expiration."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

from ocr_service.domain.models import DpiSource, JobStatus, PageOcrResult
from ocr_service.infrastructure.db.models import JobRecord, utcnow
from ocr_service.worker_main import RetentionAndRecoveryJanitor


def test_worker_crash_recovery_resumes_from_checkpointed_page(
    app_and_worker,
    turkish_multipage_tiff: Path,
) -> None:
    client, worker, app = app_and_worker
    repo = app.state.repo
    storage = app.state.storage
    settings = app.state.settings

    with open(turkish_multipage_tiff, "rb") as f:
        resp = client.post(
            "/api/v1/jobs",
            files=[("files", ("2page.tiff", f, "image/tiff"))],
        )
    job_id = resp.json()["job_id"]

    # Simulate worker-A claiming the job, completing page 1 checkpoint, and then crashing
    claimed = repo.claim_next_job(
        worker_id="crashed-worker-A",
        lease_seconds=60,
        max_concurrent_jobs=4,
    )
    assert claimed is not None
    repo.set_job_page_count(job_id, 2)
    repo.save_page_checkpoint(
        job_id=job_id,
        worker_id="crashed-worker-A",
        page_result=PageOcrResult(
            page_number=1,
            width_px=1600,
            height_px=1200,
            width_pt=384.0,
            height_pt=288.0,
            dpi_x=300.0,
            dpi_y=300.0,
            dpi_source=DpiSource.DECLARED,
            rotation_deg=0,
            deskew_angle_deg=0.0,
            ocr_applied=True,
            mean_confidence=99.0,
            column_count=1,
            text="CHECKPOINTED_PAGE_1_TEXT",
            words=[],
            processing_duration_ms=50,
        ),
        lease_seconds=60,
    )

    # Expire worker-A's lease to simulate an unexpected worker process crash
    with app.state.db.session() as sess:
        rec = sess.get(JobRecord, job_id)
        assert rec is not None
        rec.lease_expires_at = utcnow() - timedelta(seconds=10)

    # Run Janitor to detect and recover the stale lease back to QUEUED
    janitor = RetentionAndRecoveryJanitor(settings, repo, storage)
    maint = janitor.run_maintenance_cycle()
    assert maint["recovered_jobs"] == 1
    assert repo.get_job(job_id).status == JobStatus.QUEUED

    # Healthy worker picks up the recovered job, reuses page 1 checkpoint, and OCRs page 2
    completed = worker.process_next_available_job()
    assert completed is not None
    assert completed.status == JobStatus.COMPLETED

    res = client.get(f"/api/v1/jobs/{job_id}/result").json()
    assert len(res["pages"]) == 2
    assert res["pages"][0]["text"] == "CHECKPOINTED_PAGE_1_TEXT"
    assert "İKİNCİ" in res["pages"][1]["text"]


def test_job_cancellation_and_expiration_http_semantics(
    app_and_worker,
    turkish_sample_jpeg,
) -> None:
    client, worker, app = app_and_worker
    janitor = RetentionAndRecoveryJanitor(
        app.state.settings, app.state.repo, app.state.storage
    )

    # 1. Submit and immediately cancel a queued job
    with open(turkish_sample_jpeg.path, "rb") as f:
        r1 = client.post("/api/v1/jobs", files=[("files", ("cancel_me.jpg", f, "image/jpeg"))])
    cancel_id = r1.json()["job_id"]

    cancel_resp = client.post(f"/api/v1/jobs/{cancel_id}/cancel")
    assert cancel_resp.status_code == 200
    assert cancel_resp.json()["status"] == "cancelled"

    # Worker should not process a cancelled job
    assert worker.process_next_available_job() is None

    # 2. Submit, complete, and expire a job -> verify 410 Gone on /pdf and /result, 200 on status
    with open(turkish_sample_jpeg.path, "rb") as f:
        r2 = client.post("/api/v1/jobs", files=[("files", ("expire_me.jpg", f, "image/jpeg"))])
    exp_id = r2.json()["job_id"]
    worker.process_next_available_job()

    with app.state.db.session() as sess:
        rec = sess.get(JobRecord, exp_id)
        assert rec is not None
        rec.expires_at = utcnow() - timedelta(minutes=1)

    janitor.run_maintenance_cycle()

    st = client.get(f"/api/v1/jobs/{exp_id}")
    assert st.status_code == 200
    assert st.json()["status"] == "expired"

    gone_pdf = client.get(f"/api/v1/jobs/{exp_id}/pdf")
    assert gone_pdf.status_code == 410
    assert gone_pdf.json()["error"]["code"] == "JOB_EXPIRED"

    gone_res = client.get(f"/api/v1/jobs/{exp_id}/result")
    assert gone_res.status_code == 410
