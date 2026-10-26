"""End-to-end integration tests for the REST API contract, polling, PDF download, and auth."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from ocr_service.config import Settings
from ocr_service.main import create_app


def test_complete_vertical_slice_jpeg_to_pdfa_workflow(
    app_and_worker,
    turkish_sample_jpeg,
) -> None:
    client, worker, _app = app_and_worker

    # 1. Submit JPEG via POST /api/v1/jobs (and verify Idempotency-Key replay)
    with open(turkish_sample_jpeg.path, "rb") as f:
        resp = client.post(
            "/api/v1/jobs",
            files=[("files", ("arsiv_belgesi.jpg", f, "image/jpeg"))],
            data={"language": "tur", "working_dpi": "300", "pdfa_profile": "PDF/A-2b"},
            headers={"Idempotency-Key": "idem-slice-001", "X-Correlation-ID": "corr-slice-001"},
        )
    assert resp.status_code == 202
    assert resp.headers["X-Correlation-ID"] == "corr-slice-001"
    body = resp.json()
    job_id = body["job_id"]
    assert body["status"] == "queued"
    assert body["idempotent_replay"] is False

    # Replay same Idempotency-Key -> should return existing job_id without duplicating
    with open(turkish_sample_jpeg.path, "rb") as f:
        replay = client.post(
            "/api/v1/ocr/jobs",
            files=[("files", ("arsiv_belgesi.jpg", f, "image/jpeg"))],
            headers={"Idempotency-Key": "idem-slice-001"},
        )
    assert replay.status_code == 202
    assert replay.json()["job_id"] == job_id
    assert replay.json()["idempotent_replay"] is True

    # 2. Before worker processes job, /pdf and /result return 409 Conflict (JobNotReadyError)
    not_ready = client.get(f"/api/v1/jobs/{job_id}/pdf")
    assert not_ready.status_code == 409
    assert not_ready.json()["error"]["code"] == "JOB_NOT_READY"

    # 3. Worker processes the queued job
    completed_entity = worker.process_next_available_job()
    assert completed_entity is not None
    assert completed_entity.job_id == job_id
    assert completed_entity.status.value == "completed"

    # 4. Poll job status
    status_resp = client.get(f"/api/v1/jobs/{job_id}")
    assert status_resp.status_code == 200
    status_data = status_resp.json()
    assert status_data["status"] == "completed"
    assert status_data["page_count"] == 1
    assert status_data["progress_percent"] == 100.0
    assert status_data["pdfa_compliant"] is True

    # 5. Retrieve structured OCR result (FR-07)
    res_resp = client.get(f"/api/v1/jobs/{job_id}/result?include_boxes=true")
    assert res_resp.status_code == 200
    res_data = res_resp.json()
    assert res_data["job_id"] == job_id
    assert res_data["page_count"] == 1
    assert res_data["pdf_url"] == f"/api/v1/jobs/{job_id}/pdf"
    page1 = res_data["pages"][0]
    assert page1["page_number"] == 1
    assert "MECLİSİ" in page1["text"] or "Cumhuriyet" in page1["text"]
    assert len(page1["words"]) >= 10
    assert page1["nlp"] is not None
    assert len(page1["nlp"]["topics"]) >= 1
    assert len(page1["nlp"]["keywords"]) >= 1
    assert res_data["document_nlp_summary"] is not None
    assert res_data["document_nlp_summary"]["analyzed_page_count"] == 1

    # 6. Retrieve paginated /pages endpoint
    pages_resp = client.get(f"/api/v1/jobs/{job_id}/pages?page_start=1&page_end=1")
    assert pages_resp.status_code == 200
    assert pages_resp.json()["returned_pages"] == 1

    # 7. Download generated searchable PDF/A binary
    pdf_resp = client.get(f"/api/v1/jobs/{job_id}/pdf")
    assert pdf_resp.status_code == 200
    assert pdf_resp.headers["content-type"] == "application/pdf"
    assert pdf_resp.content.startswith(b"%PDF-1.")

    # 8. Retrieve job metadata & telemetry
    meta_resp = client.get(f"/api/v1/jobs/{job_id}/metadata")
    assert meta_resp.status_code == 200
    meta_data = meta_resp.json()
    assert meta_data["pdfa_compliant"] is True
    assert len(meta_data["page_telemetry"]) == 1
    assert meta_data["page_telemetry"][0]["image_preservation_mode"] == "jpeg_dct_passthrough"


def test_health_and_metrics_endpoints(app_and_worker) -> None:
    client, _worker, _app = app_and_worker

    ui_resp = client.get("/")
    assert ui_resp.status_code == 200
    assert "Archival Turkish OCR" in ui_resp.text

    live = client.get("/api/v1/health/live")
    assert live.status_code == 200
    assert live.json()["status"] == "ok"

    ready = client.get("/api/v1/health/ready")
    assert ready.status_code == 200
    assert ready.json()["ocr_engine_ok"] is True

    metrics = client.get("/api/v1/metrics")
    assert metrics.status_code == 200
    mdata = metrics.json()
    assert "queue_depth" in mdata
    assert "system" in mdata
    assert "cpu_percent" in mdata["system"]


def test_api_key_authentication_enforcement(tmp_path: Path) -> None:
    settings = Settings(
        APP_ENV="testing",
        TEMP_STORAGE_PATH=tmp_path / "auth_data",
        AUTH_ENABLED=True,
        API_KEYS=["secret-prod-key-123"],
    )
    app = create_app(settings)
    with TestClient(app) as client:
        # Unauthenticated request -> 401
        r1 = client.get("/api/v1/metrics")
        assert r1.status_code == 401
        assert r1.json()["error"]["code"] == "UNAUTHORIZED"

        # Wrong key -> 401
        r2 = client.get("/api/v1/metrics", headers={"X-API-Key": "wrong-key"})
        assert r2.status_code == 401

        # Valid X-API-Key -> 200
        r3 = client.get("/api/v1/metrics", headers={"X-API-Key": "secret-prod-key-123"})
        assert r3.status_code == 200

        # Valid Bearer token -> 200
        r4 = client.get(
            "/api/v1/metrics",
            headers={"Authorization": "Bearer secret-prod-key-123"},
        )
        assert r4.status_code == 200
