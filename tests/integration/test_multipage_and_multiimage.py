"""Integration tests for multipage TIFF, multi-image array, and scanned PDF processing (FR-03)."""

from __future__ import annotations

from pathlib import Path

import pikepdf


def test_multipage_tiff_to_pdfa_preserves_page_order_and_aspect_ratio(
    app_and_worker,
    turkish_multipage_tiff: Path,
) -> None:
    client, worker, _app = app_and_worker

    with open(turkish_multipage_tiff, "rb") as f:
        resp = client.post(
            "/api/v1/jobs",
            files=[("files", ("multipage.tiff", f, "image/tiff"))],
        )
    assert resp.status_code == 202
    job_id = resp.json()["job_id"]

    completed = worker.process_next_available_job()
    assert completed is not None
    assert completed.status.value == "completed"
    assert completed.page_count == 2

    res = client.get(f"/api/v1/jobs/{job_id}/result").json()
    assert len(res["pages"]) == 2
    assert "BİRİNCİ" in res["pages"][0]["text"]
    assert "İKİNCİ" in res["pages"][1]["text"]

    # Frame 1 was 1600x1200 (4:3), Frame 2 was 1800x1400 (9:7) -> verify mixed page sizes preserved
    assert res["pages"][0]["width_px"] == 1600
    assert res["pages"][0]["height_px"] == 1200
    assert res["pages"][1]["width_px"] == 1800
    assert res["pages"][1]["height_px"] == 1400


def test_multiple_images_converted_into_single_pdfa_in_client_order(
    app_and_worker,
    turkish_sample_jpeg,
    turkish_sample_png,
) -> None:
    client, worker, _app = app_and_worker

    with open(turkish_sample_jpeg.path, "rb") as f1, open(turkish_sample_png.path, "rb") as f2:
        resp = client.post(
            "/api/v1/jobs",
            files=[
                ("files", ("01_first.jpg", f1, "image/jpeg")),
                ("files", ("02_second.png", f2, "image/png")),
            ],
        )
    assert resp.status_code == 202
    job_id = resp.json()["job_id"]

    completed = worker.process_next_available_job()
    assert completed is not None
    assert completed.status.value == "completed"
    assert completed.page_count == 2

    res = client.get(f"/api/v1/jobs/{job_id}/result").json()
    assert len(res["pages"]) == 2
    assert "MECLİSİ" in res["pages"][0]["text"]
    assert "KÜTÜPHANE" in res["pages"][1]["text"]


def test_scanned_pdf_to_searchable_pdfa(
    app_and_worker,
    scanned_image_pdf: Path,
) -> None:
    client, worker, _app = app_and_worker

    with open(scanned_image_pdf, "rb") as f:
        resp = client.post(
            "/api/v1/jobs",
            files=[("files", ("scanned.pdf", f, "application/pdf"))],
        )
    assert resp.status_code == 202
    job_id = resp.json()["job_id"]

    completed = worker.process_next_available_job()
    assert completed is not None
    assert completed.status.value == "completed"

    res = client.get(f"/api/v1/jobs/{job_id}/result").json()
    assert res["pages"][0]["ocr_applied"] is True
    assert "Cumhuriyet" in res["pages"][0]["text"]


def test_password_protected_pdf_fails_permanently_without_retries(
    app_and_worker,
    tmp_path: Path,
    scanned_image_pdf: Path,
) -> None:
    client, worker, _app = app_and_worker

    enc_pdf_path = tmp_path / "encrypted.pdf"
    with pikepdf.open(scanned_image_pdf) as pdf:
        pdf.save(
            enc_pdf_path,
            encryption=pikepdf.Encryption(owner="owner123", user="user123", R=4),
        )

    with open(enc_pdf_path, "rb") as f:
        resp = client.post(
            "/api/v1/jobs",
            files=[("files", ("encrypted.pdf", f, "application/pdf"))],
        )
    assert resp.status_code == 202
    resp.json()["job_id"]

    failed_job = worker.process_next_available_job()
    assert failed_job is not None
    assert failed_job.status.value == "failed"
    assert failed_job.error_code == "PASSWORD_PROTECTED_PDF"
    assert failed_job.retryable is False
    assert failed_job.attempt_count == 1
