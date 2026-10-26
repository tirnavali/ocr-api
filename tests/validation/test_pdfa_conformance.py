"""PDF/A-2b and PDF/A-1b conformance and lossless source image preservation tests (FR-05, FR-14)."""

from __future__ import annotations

from pathlib import Path

import pikepdf

from ocr_service.domain.models import PdfAProfile
from ocr_service.pipeline.pdfa_validator import DualPdfAValidator


def test_generated_pdfa_2b_and_1b_conformance_and_jpeg_passthrough_identity(
    app_and_worker,
    turkish_sample_jpeg,
    tmp_path: Path,
) -> None:
    """Verify PDF/A structural conformance and 100% byte-identical JPEG stream preservation."""
    client, worker, app = app_and_worker
    validator = DualPdfAValidator(app.state.settings)

    original_jpeg_bytes = turkish_sample_jpeg.path.read_bytes()

    for profile in (PdfAProfile.PDFA_2B, PdfAProfile.PDFA_1B):
        with open(turkish_sample_jpeg.path, "rb") as f:
            resp = client.post(
                "/api/v1/jobs",
                files=[("files", ("archival.jpg", f, "image/jpeg"))],
                data={"pdfa_profile": profile.value},
            )
        job_id = resp.json()["job_id"]
        worker.process_next_available_job()

        pdf_bytes = client.get(f"/api/v1/jobs/{job_id}/pdf").content
        out_pdf_path = tmp_path / f"out_{profile.name}.pdf"
        out_pdf_path.write_bytes(pdf_bytes)

        report = validator.validate(out_pdf_path, profile=profile)
        assert report.compliant is True, f"PDF/A validation errors: {report.errors}"
        assert report.details["output_intent_components"] == 3

        # Inspect embedded image XObject inside the generated PDF/A and assert byte-for-byte identity
        with pikepdf.open(out_pdf_path) as pdf:
            page = pdf.pages[0]
            im0 = page.obj["/Resources"]["/XObject"]["/Im0"]
            assert im0.get("/Filter") == pikepdf.Name("/DCTDecode")
            embedded_raw_jpeg = im0.read_raw_bytes()
            assert embedded_raw_jpeg == original_jpeg_bytes, (
                "Embedded JPEG stream in PDF/A was re-encoded instead of preserved losslessly!"
            )


def test_pdf_with_unembedded_type1_fonts_remediated_in_both_preserve_and_force_ocr_modes(
    app_and_worker,
    scanned_image_pdf: Path,
    tmp_path: Path,
) -> None:
    """Verify PDFs with legacy unembedded /Type1 fonts (e.g. ABBYY FineReader) pass PDF/A-2b validation."""
    client, worker, _app = app_and_worker

    legacy_pdf_path = tmp_path / "legacy_unembedded_type1.pdf"
    with pikepdf.open(scanned_image_pdf) as pdf:
        page = pdf.pages[0]
        unembedded_font = pdf.make_indirect(
            pikepdf.Dictionary(
                {
                    "/Type": pikepdf.Name("/Font"),
                    "/Subtype": pikepdf.Name("/Type1"),
                    "/BaseFont": pikepdf.Name("/Times-Roman"),
                    "/Encoding": pikepdf.Name("/WinAnsiEncoding"),
                    "/FontDescriptor": pikepdf.Dictionary(
                        {
                            "/Type": pikepdf.Name("/FontDescriptor"),
                            "/FontName": pikepdf.Name("/Times-Roman"),
                            "/Flags": 32,
                            "/FontBBox": pikepdf.Array([-568, -307, 2000, 1007]),
                            "/ItalicAngle": 0,
                            "/Ascent": 891,
                            "/Descent": -216,
                            "/CapHeight": 662,
                            "/StemV": 80,
                        }
                    ),
                }
            )
        )
        page.obj["/Resources"]["/Font"] = pikepdf.Dictionary({"/F48": unembedded_font})
        legacy_text = (
            b"\nBT 3 Tr /F48 12 Tf 1 0 0 1 72 700 Tm "
            b"(Turkiye Buyuk Millet Meclisi Arsiv Belgesi Metni) Tj ET\n"
        )
        page.contents_add(pdf.make_indirect(pikepdf.Stream(pdf, legacy_text)))
        pdf.save(legacy_pdf_path)

    for force_flag in ("false", "true"):
        with open(legacy_pdf_path, "rb") as f:
            resp = client.post(
                "/api/v1/jobs",
                files=[("files", ("legacy.pdf", f, "application/pdf"))],
                data={"force_ocr": force_flag},
            )
        job_id = resp.json()["job_id"]
        done = worker.process_next_available_job()
        assert done is not None
        assert done.job_id == job_id
        assert done.status.value == "completed", f"Failed with force_ocr={force_flag}: {done.error_message}"
        assert done.pdfa_compliant is True
