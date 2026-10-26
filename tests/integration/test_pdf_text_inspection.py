"""Integration tests verifying existing PDF text layer inspection and force_ocr behavior (FR-03)."""

from __future__ import annotations

from pathlib import Path


def test_pdf_with_existing_usable_text_skips_ocr_unless_forced(
    app_and_worker,
    scanned_image_pdf: Path,
    tmp_path: Path,
) -> None:
    client, worker, _app = app_and_worker

    # Step 1: Produce a searchable PDF/A from the scanned PDF
    with open(scanned_image_pdf, "rb") as f:
        r1 = client.post("/api/v1/jobs", files=[("files", ("scan.pdf", f, "application/pdf"))])
    job1_id = r1.json()["job_id"]
    worker.process_next_available_job()

    searchable_pdf_bytes = client.get(f"/api/v1/jobs/{job1_id}/pdf").content
    searchable_pdf_path = tmp_path / "already_searchable.pdf"
    searchable_pdf_path.write_bytes(searchable_pdf_bytes)

    # Step 2: Submit the already-searchable PDF with force_ocr=false (default)
    with open(searchable_pdf_path, "rb") as f:
        r2 = client.post(
            "/api/v1/jobs",
            files=[("files", ("already_searchable.pdf", f, "application/pdf"))],
            data={"force_ocr": "false"},
        )
    job2_id = r2.json()["job_id"]
    worker.process_next_available_job()

    res2 = client.get(f"/api/v1/jobs/{job2_id}/result").json()
    page_no_ocr = res2["pages"][0]
    assert page_no_ocr["ocr_applied"] is False
    assert page_no_ocr["image_preservation_mode"] == "pdf_native_preserved"
    assert "Cumhuriyet" in page_no_ocr["text"]

    # Step 3: Submit the same PDF with force_ocr=true -> must re-OCR!
    with open(searchable_pdf_path, "rb") as f:
        r3 = client.post(
            "/api/v1/jobs",
            files=[("files", ("already_searchable.pdf", f, "application/pdf"))],
            data={"force_ocr": "true"},
        )
    job3_id = r3.json()["job_id"]
    worker.process_next_available_job()

    res3 = client.get(f"/api/v1/jobs/{job3_id}/result").json()
    assert res3["pages"][0]["ocr_applied"] is True
