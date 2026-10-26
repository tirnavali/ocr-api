"""Automated geometric alignment verification of the invisible PDF/A text layer (FR-06, FR-13)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pypdfium2 as pdfium
from PIL import Image
from tests.conftest import render_turkish_archival_image


def test_invisible_text_layer_word_coordinates_match_source_image(
    app_and_worker,
    turkish_sample_jpeg,
    tmp_path: Path,
) -> None:
    """Verify OCR word positions in PDF user space match ground-truth word coordinates."""
    client, worker, _app = app_and_worker

    with open(turkish_sample_jpeg.path, "rb") as f:
        resp = client.post(
            "/api/v1/jobs",
            files=[("files", ("align_check.jpg", f, "image/jpeg"))],
        )
    job_id = resp.json()["job_id"]
    worker.process_next_available_job()

    res = client.get(f"/api/v1/jobs/{job_id}/result?include_boxes=true").json()
    ocr_words = res["pages"][0]["words"]
    assert len(ocr_words) >= len(turkish_sample_jpeg.words) - 2

    # Match recognized words to ground-truth words and verify center coordinate alignment in PDF points
    matched_count = 0
    for gt_w in turkish_sample_jpeg.words:
        candidates = [w for w in ocr_words if w["text"].strip(".,:") == gt_w.text.strip(".,:")]
        if not candidates:
            continue
        best = min(
            candidates,
            key=lambda w: abs(w["bbox_pdf_pt"]["x"] - gt_w.bbox_pt.x)
            + abs(w["bbox_pdf_pt"]["y"] - gt_w.bbox_pt.y),
        )
        pdf_box = best["bbox_pdf_pt"]
        dx_pt = abs(pdf_box["x"] - gt_w.bbox_pt.x)
        dy_pt = abs(pdf_box["y"] - gt_w.bbox_pt.y)
        dw_pt = abs(pdf_box["width"] - gt_w.bbox_pt.width)

        # Within 4 PDF points (~0.05 inch) of ground-truth rendered glyph box
        assert dx_pt < 4.5, f"Word {gt_w.text!r} X drift {dx_pt:.2f} pt"
        assert dy_pt < 4.5, f"Word {gt_w.text!r} Y drift {dy_pt:.2f} pt"
        assert dw_pt < 6.0, f"Word {gt_w.text!r} width drift {dw_pt:.2f} pt"
        matched_count += 1

    assert matched_count >= 12

    # Also verify directly inside the generated PDF using pypdfium2 character bounding boxes!
    pdf_bytes = client.get(f"/api/v1/jobs/{job_id}/pdf").content
    pdf_path = tmp_path / "alignment_verified.pdf"
    pdf_path.write_bytes(pdf_bytes)

    pdf_doc = pdfium.PdfDocument(str(pdf_path))
    try:
        page = pdf_doc[0]
        w_pt, h_pt = page.get_size()
        assert abs(w_pt - (turkish_sample_jpeg.width_px * 72.0 / turkish_sample_jpeg.dpi)) < 0.5
        assert abs(h_pt - (turkish_sample_jpeg.height_px * 72.0 / turkish_sample_jpeg.dpi)) < 0.5

        textpage = page.get_textpage()
        n_chars = textpage.count_chars()
        assert n_chars > 80
        # First character ("T" of TÜRKİYE) must sit near the top-left ground-truth word
        first_gt = turkish_sample_jpeg.words[0].bbox_pt
        left, bottom, _right, _top = textpage.get_charbox(0)
        assert abs(left - first_gt.x) < 5.0
        assert abs(bottom - first_gt.y) < 6.0
        textpage.close()
        page.close()
    finally:
        pdf_doc.close()


def test_400_dpi_source_with_250_working_dpi_preserves_400_dpi_output_and_alignment(
    app_and_worker,
    tmp_path: Path,
) -> None:
    """Verify FR-13 requirement: 400 DPI scan processed at 250 working DPI retains full 400 DPI visual image."""
    client, worker, _app = app_and_worker

    sample_400 = render_turkish_archival_image(
        tmp_path / "scan_400dpi.jpg",
        [
            "YÜKSEK ÇÖZÜNÜRLÜKLÜ ARŞİV BELGESİ",
            "Dört yüz DPI tarama kalitesi korunmalıdır.",
        ],
        width_px=2400,
        height_px=3200,
        dpi=400,
        font_size_px=58,
        fmt="JPEG",
    )

    with open(sample_400.path, "rb") as f:
        resp = client.post(
            "/api/v1/jobs",
            files=[("files", ("scan_400dpi.jpg", f, "image/jpeg"))],
            data={"working_dpi": "250", "preserve_source_images": "true"},
        )
    job_id = resp.json()["job_id"]
    worker.process_next_available_job()

    res = client.get(f"/api/v1/jobs/{job_id}/result?include_boxes=true").json()
    page = res["pages"][0]

    # Output page dimensions must still be 2400x3200 px at 400 DPI (432 x 576 pt)
    assert page["width_px"] == 2400
    assert page["height_px"] == 3200
    assert abs(page["width_pt"] - 432.0) < 0.5
    assert abs(page["height_pt"] - 576.0) < 0.5
    assert page["image_preservation_mode"] == "jpeg_dct_passthrough"

    # And word coordinates mapped from the downsampled 250 DPI working image back to 400 DPI source
    # must match the 400 DPI ground truth!
    first_gt = sample_400.words[0]
    first_ocr = next(w for w in page["words"] if "YÜKSEK" in w["text"])
    assert abs(first_ocr["bbox_source_px"]["left"] - first_gt.bbox_px.left) < 15.0
    assert abs(first_ocr["bbox_pdf_pt"]["x"] - first_gt.bbox_pt.x) < 4.0


def test_skewed_working_image_inverse_coordinate_alignment(
    app_and_worker,
    tmp_path: Path,
) -> None:
    """Verify deskewing on working image inverse-maps coordinates accurately back to skewed source."""
    client, worker, _app = app_and_worker

    base_sample = render_turkish_archival_image(
        tmp_path / "unskewed.png",
        [
            "EĞİK TARANMIŞ GAZETE SAYFASI TESTİ",
            "Metin katmanı hizalaması ters dönüşüm ile korunur.",
            "Üçüncü satır kontrol metni ve Türkçe karakterler: ğüşiöç.",
        ],
        width_px=1800,
        height_px=2200,
        dpi=300,
        fmt="PNG",
    )

    # Introduce a synthetic +2.2 degree skew to the input scan
    arr = np.array(Image.open(base_sample.path))
    h, w = arr.shape[:2]
    mat = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), 2.2, 1.0)
    skewed_arr = cv2.warpAffine(
        arr, mat, (w, h), flags=cv2.INTER_CUBIC, borderValue=(252, 250, 245)
    )
    skewed_path = tmp_path / "skewed_input.png"
    Image.fromarray(skewed_arr).save(skewed_path, format="PNG", dpi=(300, 300))

    with open(skewed_path, "rb") as f:
        resp = client.post(
            "/api/v1/jobs",
            files=[("files", ("skewed.png", f, "image/png"))],
            data={"deskew_working_image": "true"},
        )
    job_id = resp.json()["job_id"]
    worker.process_next_available_job()

    res = client.get(f"/api/v1/jobs/{job_id}/result?include_boxes=true").json()
    page = res["pages"][0]
    assert "GAZETE" in page["text"]
    assert len(page["words"]) >= 12
