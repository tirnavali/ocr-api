"""End-to-end: submit by CDN path, worker downloads + OCRs, signed callback is delivered."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from ocr_service.config import Settings
from ocr_service.infrastructure.callbacks.notifier import (
    SIGNATURE_HEADER,
    TIMESTAMP_HEADER,
    CallbackNotifier,
    sign_payload,
)
from ocr_service.infrastructure.cdn.cdn_client import InternalCdnClient
from ocr_service.main import create_app
from ocr_service.pipeline.orchestrator import OcrPipelineOrchestrator
from ocr_service.worker_main import OcrWorker

CDN_BASE = "http://cdn.test/File/"
CALLBACK_URL = "http://kahb-api.test/api/v1/Ocr/Sonuc"
SECRET = "integration-secret"


@pytest.fixture()
def cdn_settings(tmp_path: Path) -> Settings:
    storage_dir = tmp_path / "ocr_data"
    storage_dir.mkdir()
    return Settings(
        APP_ENV="testing",
        LOG_JSON=False,
        LOG_LEVEL="WARNING",
        TEMP_STORAGE_PATH=storage_dir,
        OCR_LANGUAGE="tur",
        JOB_MAX_RETRIES=2,
        JOB_TIMEOUT_SECONDS=120,
        MAX_UPLOAD_MB=50,
        CDN_BASE_URL=CDN_BASE,
        CDN_VERIFICATION_ID="verif-xyz",
        CALLBACK_HMAC_SECRET=SECRET,
        CALLBACK_ALLOWED_PREFIXES="http://kahb-api.test/",
    )


def _build_worker(settings: Settings, app, cdn_files: dict[str, bytes], callbacks: list[httpx.Request]):
    def cdn_handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["UygulamaId"] == "verif-xyz"
        rel = str(request.url).removeprefix(CDN_BASE)
        if rel in cdn_files:
            return httpx.Response(200, content=cdn_files[rel])
        return httpx.Response(404)

    def callback_handler(request: httpx.Request) -> httpx.Response:
        callbacks.append(request)
        return httpx.Response(204)

    repo, storage = app.state.repo, app.state.storage
    orchestrator = OcrPipelineOrchestrator(
        settings,
        repo,
        storage,
        cdn_client=InternalCdnClient(settings, transport=httpx.MockTransport(cdn_handler)),
    )
    notifier = CallbackNotifier(settings, repo, transport=httpx.MockTransport(callback_handler))
    return OcrWorker(
        settings, repo, storage, orchestrator=orchestrator, worker_id="cdn-w1", notifier=notifier
    )


def _verify_signed(request: httpx.Request) -> dict:
    body = request.content
    assert request.headers[SIGNATURE_HEADER] == sign_payload(
        SECRET, request.headers[TIMESTAMP_HEADER], body
    )
    return json.loads(body)


def test_cdn_referenced_job_is_downloaded_ocred_and_reported(
    cdn_settings: Settings, turkish_sample_jpeg
) -> None:
    app = create_app(cdn_settings)
    callbacks: list[httpx.Request] = []
    cdn_files = {"AOS/ab/cd/arsiv_belgesi.jpg": Path(turkish_sample_jpeg.path).read_bytes()}

    with TestClient(app) as client:
        worker = _build_worker(cdn_settings, app, cdn_files, callbacks)
        resp = client.post(
            "/api/v1/jobs/from-cdn",
            json={
                "sources": [{"path": "AOS\\ab\\cd\\arsiv_belgesi.jpg"}],
                "callback_url": CALLBACK_URL,
            },
            headers={"Idempotency-Key": "ek-123"},
        )
        assert resp.status_code == 202, resp.text
        job_id = resp.json()["job_id"]

        replay = client.post(
            "/api/v1/jobs/from-cdn",
            json={"sources": [{"path": "AOS\\ab\\cd\\arsiv_belgesi.jpg"}], "callback_url": CALLBACK_URL},
            headers={"Idempotency-Key": "ek-123"},
        )
        assert replay.json() == {**resp.json(), "idempotent_replay": True}

        status_before = client.get(f"/api/v1/jobs/{job_id}").json()
        assert status_before["callback_status"] == "waiting"

        done = worker.process_next_available_job()
        assert done is not None and done.status.value == "completed", done and done.error_message

        status = client.get(f"/api/v1/jobs/{job_id}").json()
        assert status["status"] == "completed"
        assert status["callback_status"] == "delivered"

        pdf = client.get(f"/api/v1/jobs/{job_id}/pdf")
        assert pdf.status_code == 200 and pdf.content.startswith(b"%PDF-")

        result = client.get(f"/api/v1/jobs/{job_id}/result").json()
        text = " ".join(p["text"] for p in result["pages"])
        assert "MECLİSİ" in text or "MECLISI" in text.upper()

    assert len(callbacks) == 1
    payload = _verify_signed(callbacks[0])
    assert payload["job_id"] == job_id
    assert payload["status"] == "completed"
    assert payload["idempotency_key"] == "ek-123"
    assert payload["pdf_url"] == f"/api/v1/jobs/{job_id}/pdf"


def test_missing_cdn_file_fails_permanently_and_reports_failure(cdn_settings: Settings) -> None:
    app = create_app(cdn_settings)
    callbacks: list[httpx.Request] = []

    with TestClient(app) as client:
        worker = _build_worker(cdn_settings, app, {}, callbacks)
        resp = client.post(
            "/api/v1/jobs/from-cdn",
            json={"sources": [{"path": "AOS\\yok\\olmayan.pdf"}], "callback_url": CALLBACK_URL},
        )
        job_id = resp.json()["job_id"]

        done = worker.process_next_available_job()
        # Permanent: no retries burned on a file that does not exist.
        assert done is not None and done.status.value == "failed"
        assert done.attempt_count == 1
        assert done.error_code == "CDN_SOURCE_NOT_FOUND"
        assert worker.process_next_available_job() is None

        status = client.get(f"/api/v1/jobs/{job_id}").json()
        assert status["callback_status"] == "delivered"

    payload = _verify_signed(callbacks[0])
    assert payload["status"] == "failed"
    assert payload["error_code"] == "CDN_SOURCE_NOT_FOUND"
    assert payload["pdf_url"] is None


@pytest.mark.parametrize(
    ("body", "code"),
    [
        ({"sources": [{"path": "..\\..\\etc\\passwd"}]}, "INVALID_CDN_PATH"),
        ({"sources": [{"path": "AOS\\a.pdf"}], "callback_url": "http://evil.test/x"}, "CALLBACK_URL_NOT_ALLOWED"),
    ],
)
def test_cdn_submission_rejects_unsafe_input(cdn_settings: Settings, body: dict, code: str) -> None:
    with TestClient(create_app(cdn_settings)) as client:
        resp = client.post("/api/v1/jobs/from-cdn", json=body)
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == code


def test_cdn_submission_disabled_without_cdn_config(test_settings: Settings) -> None:
    with TestClient(create_app(test_settings)) as client:
        resp = client.post("/api/v1/jobs/from-cdn", json={"sources": [{"path": "AOS\\a.pdf"}]})
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "CDN_NOT_CONFIGURED"
