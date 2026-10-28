"""Package jobs: CDN files streamed into size-limited zip parts, skipped files reported."""

from __future__ import annotations

import io
import json
import os
import zipfile
from datetime import UTC, datetime, timedelta
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
from ocr_service.infrastructure.db.models import JobRecord
from ocr_service.main import create_app
from ocr_service.pipeline import packager
from ocr_service.pipeline.packager import PackageJobHandler, sanitize_archive_name
from ocr_service.worker_main import OcrWorker

CDN_BASE = "http://cdn.test/File/"
CALLBACK_URL = "http://kahb-api.test/api/v1/DisaAktar/PaketlemeSonucu"
SECRET = "package-secret"


@pytest.fixture()
def package_settings(tmp_path: Path) -> Settings:
    storage_dir = tmp_path / "ocr_data"
    storage_dir.mkdir()
    return Settings(
        APP_ENV="testing",
        LOG_JSON=False,
        LOG_LEVEL="WARNING",
        TEMP_STORAGE_PATH=storage_dir,
        WORKER_JOB_TYPES="package",
        JOB_MAX_RETRIES=2,
        CDN_BASE_URL=CDN_BASE,
        CDN_VERIFICATION_ID="verif-xyz",
        CALLBACK_HMAC_SECRET=SECRET,
        CALLBACK_ALLOWED_PREFIXES="http://kahb-api.test/",
        MAX_PACKAGE_ENTRIES=10,
    )


@pytest.fixture(autouse=True)
def _no_retry_delay(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(packager, "FILE_RETRY_DELAY_SECONDS", 0.0)


class FakeCdn:
    """CDN double: known paths return bytes, unknown ones answer like the TBMM CDN (500)."""

    def __init__(self, files: dict[str, bytes]) -> None:
        self.files = files
        self.requests: list[str] = []
        self.network_down = False

    def handler(self, request: httpx.Request) -> httpx.Response:
        assert request.headers["UygulamaId"] == "verif-xyz"
        rel = str(request.url).removeprefix(CDN_BASE)
        self.requests.append(rel)
        if self.network_down:
            raise httpx.ConnectError("connection refused", request=request)
        if rel in self.files:
            return httpx.Response(200, content=self.files[rel])
        return httpx.Response(500)


def _build_worker(settings: Settings, app, cdn: FakeCdn, callbacks: list[httpx.Request]) -> OcrWorker:
    def callback_handler(request: httpx.Request) -> httpx.Response:
        callbacks.append(request)
        return httpx.Response(204)

    repo, storage = app.state.repo, app.state.storage
    handler = PackageJobHandler(
        settings,
        repo,
        storage,
        cdn_client=InternalCdnClient(settings, transport=httpx.MockTransport(cdn.handler)),
    )
    notifier = CallbackNotifier(settings, repo, transport=httpx.MockTransport(callback_handler))
    return OcrWorker(
        settings, repo, storage, worker_id="pkg-w1", handlers={"package": handler}, notifier=notifier
    )


def _verify_signed(request: httpx.Request) -> dict:
    body = request.content
    assert request.headers[SIGNATURE_HEADER] == sign_payload(
        SECRET, request.headers[TIMESTAMP_HEADER], body
    )
    return json.loads(body)


def _blob(size: int, seed: int) -> bytes:
    # Incompressible content so the zip part sizes are predictable.
    return os.urandom(size - 4) + seed.to_bytes(4, "big")


def test_package_job_splits_parts_and_reports_skipped_files(package_settings: Settings) -> None:
    app = create_app(package_settings)
    callbacks: list[httpx.Request] = []
    files = {
        "AOS/aa/01/a.pdf": _blob(700_000, 1),
        "AOS/aa/02/b.pdf": _blob(700_000, 2),
        "AOS/aa/03/c.pdf": _blob(300_000, 3),
        "AOS/aa/04/d.pdf": _blob(10_000, 4),
    }
    cdn = FakeCdn(files)
    entries = [
        {"path": "AOS\\aa\\01\\a.pdf", "name": "Gazete Fonu/K00001_a.pdf", "ref": "ek-1"},
        {"path": "AOS\\aa\\99\\yok.pdf", "name": "Gazete Fonu/K00002_yok.pdf", "ref": "ek-2"},
        {"path": "AOS\\aa\\02\\b.pdf", "name": "Gazete Fonu/K00001_a.pdf", "ref": "ek-3"},
        {"path": "AOS\\aa\\03\\c.pdf", "name": "../../etc/Özel Fon/c.pdf", "ref": "ek-4"},
        {"path": "AOS\\aa\\04\\d.pdf", "name": "MANIFEST.txt", "ref": "ek-5"},
    ]

    with TestClient(app) as client:
        worker = _build_worker(package_settings, app, cdn, callbacks)
        resp = client.post(
            "/api/v1/jobs/package",
            json={
                "entries": entries,
                "max_part_mb": 1,
                "part_name_prefix": "disa-aktarma_abc",
                "manifest_title": "KAHB Toplu Dışa Aktarma — Talep: abc",
                "callback_url": CALLBACK_URL,
            },
            headers={"Idempotency-Key": "talep-abc"},
        )
        assert resp.status_code == 202, resp.text
        job_id = resp.json()["job_id"]

        done = worker.process_next_available_job()
        assert done is not None and done.status.value == "completed", done and done.error_message

        status = client.get(f"/api/v1/jobs/{job_id}").json()
        assert status["job_type"] == "package"
        assert status["progress_percent"] == 100.0
        assert status["pdf_url"] is None
        assert status["callback_status"] == "delivered"
        pkg = status["package"]
        assert (pkg["entry_count"], pkg["ok_count"], pkg["error_count"]) == (5, 4, 1)
        # Part 1 is closed once it passes 1 MB (a + b), the rest goes to part 2.
        assert [p["file_count"] for p in pkg["parts"]] == [2, 2]
        assert [p["filename"] for p in pkg["parts"]] == [
            "disa-aktarma_abc_paket-01.zip",
            "disa-aktarma_abc_paket-02.zip",
        ]
        assert pkg["errors"][0]["ref"] == "ek-2"
        assert pkg["errors"][0]["error_code"] == "CDN_FETCH_FAILED"

        part1 = client.get(pkg["parts"][0]["download_url"])
        assert part1.status_code == 200
        assert part1.headers["content-type"] == "application/zip"
        with zipfile.ZipFile(io.BytesIO(part1.content)) as zf:
            names = zf.namelist()
            assert names == [
                "Gazete Fonu/K00001_a.pdf",
                "Gazete Fonu/K00001_a_2.pdf",
                "MANIFEST.txt",
            ]
            assert zf.read("Gazete Fonu/K00001_a.pdf") == files["AOS/aa/01/a.pdf"]
            assert zf.getinfo("Gazete Fonu/K00001_a.pdf").compress_type == zipfile.ZIP_STORED
            manifest = zf.read("MANIFEST.txt").decode("utf-8")
            assert manifest.startswith("# KAHB Toplu Dışa Aktarma — Talep: abc")
            assert "HATA;ek-2;Gazete Fonu/K00002_yok.pdf;;CDN_FETCH_FAILED" in manifest
            assert "OK;ek-3;Gazete Fonu/K00001_a_2.pdf;700000" in manifest

        part2 = client.get(pkg["parts"][1]["download_url"])
        with zipfile.ZipFile(io.BytesIO(part2.content)) as zf:
            # Parent segments are dropped; the user's own MANIFEST.txt does not clash.
            assert zf.namelist() == ["etc/Özel Fon/c.pdf", "MANIFEST_2.txt", "MANIFEST.txt"]

        ranged = client.get(pkg["parts"][0]["download_url"], headers={"Range": "bytes=0-3"})
        assert ranged.status_code == 206
        assert ranged.content == b"PK\x03\x04"

        assert client.get(f"/api/v1/jobs/{job_id}/packages/9").status_code == 404

    # The missing file was tried twice (one retry), the others once.
    assert cdn.requests.count("AOS/aa/99/yok.pdf") == 2
    payload = _verify_signed(callbacks[0])
    assert payload["status"] == "completed"
    assert payload["job_type"] == "package"
    assert payload["idempotency_key"] == "talep-abc"
    assert payload["package"]["ok_count"] == 4
    assert payload["packages_url"] == f"/api/v1/jobs/{job_id}/packages"
    assert payload["pdf_url"] is None


def test_package_with_no_fetchable_file_fails_with_details(package_settings: Settings) -> None:
    app = create_app(package_settings)
    callbacks: list[httpx.Request] = []

    with TestClient(app) as client:
        worker = _build_worker(package_settings, app, FakeCdn({}), callbacks)
        resp = client.post(
            "/api/v1/jobs/package",
            json={
                "entries": [{"path": "AOS\\x\\1.pdf", "name": "1.pdf", "ref": "ek-9"}],
                "callback_url": CALLBACK_URL,
            },
        )
        job_id = resp.json()["job_id"]

        done = worker.process_next_available_job()
        assert done is not None and done.status.value == "failed"
        assert done.error_code == "PACKAGE_EMPTY"
        assert worker.process_next_available_job() is None

        status = client.get(f"/api/v1/jobs/{job_id}").json()
        assert status["package"]["error_count"] == 1
        assert status["package"]["parts"] == []

    payload = _verify_signed(callbacks[0])
    assert payload["status"] == "failed"
    assert payload["package"]["errors"][0]["ref"] == "ek-9"


def test_cdn_outage_retries_the_whole_job_instead_of_skipping_everything(
    package_settings: Settings,
) -> None:
    app = create_app(package_settings)
    cdn = FakeCdn({f"AOS/n/{i}.pdf": b"x" * 10 for i in range(8)})
    cdn.network_down = True

    with TestClient(app) as client:
        worker = _build_worker(package_settings, app, cdn, [])
        resp = client.post(
            "/api/v1/jobs/package",
            json={"entries": [{"path": f"AOS\\n\\{i}.pdf", "name": f"{i}.pdf"} for i in range(8)]},
        )
        job_id = resp.json()["job_id"]

        first = worker.process_next_available_job()
        assert first is not None and first.status.value == "queued"
        assert first.error_code == "CDN_FETCH_FAILED"

        cdn.network_down = False
        # Skip the retry backoff instead of sleeping in the test.
        with app.state.repo._db.session() as sess:
            sess.get(JobRecord, job_id).next_retry_at = datetime.now(UTC) - timedelta(seconds=1)
        second = worker.process_next_available_job()
        assert second is not None and second.status.value == "completed"
        assert second.page_count == 8

        pkg = client.get(f"/api/v1/jobs/{job_id}").json()["package"]
        assert pkg["ok_count"] == 8 and pkg["error_count"] == 0
        assert len(pkg["parts"]) == 1


def test_delete_purges_parts_immediately(package_settings: Settings) -> None:
    app = create_app(package_settings)
    cdn = FakeCdn({"AOS/d/1.pdf": b"%PDF-1.4 test"})

    with TestClient(app) as client:
        worker = _build_worker(package_settings, app, cdn, [])
        job_id = client.post(
            "/api/v1/jobs/package",
            json={"entries": [{"path": "AOS\\d\\1.pdf", "name": "1.pdf"}]},
        ).json()["job_id"]
        worker.process_next_available_job()
        url = client.get(f"/api/v1/jobs/{job_id}").json()["package"]["parts"][0]["download_url"]
        assert client.get(url).status_code == 200

        deleted = client.delete(f"/api/v1/jobs/{job_id}")
        assert deleted.status_code == 200
        assert deleted.json()["status"] == "expired"
        assert not app.state.storage.get_job_paths(job_id).job_dir.exists()
        assert client.get(url).status_code == 410


@pytest.mark.parametrize(
    ("body", "code"),
    [
        ({"entries": [{"path": "..\\..\\etc\\passwd", "name": "x"}]}, "INVALID_CDN_PATH"),
        ({"entries": [{"path": "http://evil.test/a", "name": "x"}]}, "INVALID_CDN_PATH"),
        (
            {"entries": [{"path": f"AOS\\{i}.pdf", "name": "x"} for i in range(11)]},
            "PACKAGE_TOO_MANY_ENTRIES",
        ),
        (
            {"entries": [{"path": "AOS\\a.pdf", "name": "a"}], "callback_url": "http://evil.test/"},
            "CALLBACK_URL_NOT_ALLOWED",
        ),
        ({"entries": [{"path": "AOS\\a.pdf", "name": "a"}], "max_part_mb": 999_999}, "PACKAGE_PART_TOO_LARGE"),
    ],
)
def test_package_submission_rejects_unsafe_input(
    package_settings: Settings, body: dict, code: str
) -> None:
    with TestClient(create_app(package_settings)) as client:
        resp = client.post("/api/v1/jobs/package", json=body)
    assert resp.status_code == 400, resp.text
    assert resp.json()["error"]["code"] == code


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Fon/K00001_a.pdf", "Fon/K00001_a.pdf"),
        ("..\\..\\x.pdf", "x.pdf"),
        ("/abs/path.pdf", "abs/path.pdf"),
        ("C:\\Windows\\a.pdf", "C_/Windows/a.pdf"),
        ('a<b>:"c|?*.pdf', "a_b___c___.pdf"),
        ("  ./ / ", "fallback.bin"),
        ("Fon. /dosya.", "Fon/dosya"),
    ],
)
def test_sanitize_archive_name(raw: str, expected: str) -> None:
    assert sanitize_archive_name(raw, fallback="fallback.bin") == expected
