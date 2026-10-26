"""Validation tests for Turkish OCR accuracy (CER/WER), Turkish Unicode round-trip, and multi-column newspapers."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pypdfium2 as pdfium
from tests.conftest import render_two_column_newspaper_image


def _levenshtein_distance(seq1: list[str] | str, seq2: list[str] | str) -> int:
    """Compute exact Levenshtein edit distance between two character or token sequences."""
    n, m = len(seq1), len(seq2)
    if n == 0:
        return m
    if m == 0:
        return n
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, m + 1):
            temp = dp[j]
            cost = 0 if seq1[i - 1] == seq2[j - 1] else 1
            dp[j] = min(
                dp[j] + 1,      # deletion
                dp[j - 1] + 1,  # insertion
                prev + cost,    # substitution
            )
            prev = temp
    return dp[m]


def compute_cer(reference: str, hypothesis: str) -> float:
    """Compute Character Error Rate (CER) normalized by reference length."""
    ref_norm = " ".join(reference.split())
    hyp_norm = " ".join(hypothesis.split())
    if not ref_norm:
        return 0.0 if not hyp_norm else 1.0
    dist = _levenshtein_distance(ref_norm, hyp_norm)
    return dist / float(len(ref_norm))


def compute_wer(reference: str, hypothesis: str) -> float:
    """Compute Word Error Rate (WER) normalized by reference word count."""
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    if not ref_words:
        return 0.0 if not hyp_words else 1.0
    dist = _levenshtein_distance(ref_words, hyp_words)
    return dist / float(len(ref_words))


def test_turkish_unicode_preservation_and_cer_wer_in_api_and_pdfa(
    app_and_worker,
    turkish_sample_jpeg,
    tmp_path: Path,
) -> None:
    """Verify Turkish characters (ç, Ç, ğ, Ğ, ı, I, i, İ, ö, Ö, ş, Ş, ü, Ü) survive OCR and PDF/A extraction."""
    client, worker, _app = app_and_worker

    with open(turkish_sample_jpeg.path, "rb") as f:
        resp = client.post(
            "/api/v1/jobs",
            files=[("files", ("turkish.jpg", f, "image/jpeg"))],
            data={"language": "tur"},
        )
    job_id = resp.json()["job_id"]
    worker.process_next_available_job()

    res = client.get(f"/api/v1/jobs/{job_id}/result").json()
    api_text = res["pages"][0]["text"]

    # Measure CER and WER against ground-truth transcription
    cer = compute_cer(turkish_sample_jpeg.ground_truth_text, api_text)
    wer = compute_wer(turkish_sample_jpeg.ground_truth_text, api_text)
    assert cer < 0.05, f"CER too high ({cer:.3f}):\nGT: {turkish_sample_jpeg.ground_truth_text}\nOCR: {api_text}"
    assert wer < 0.12, f"WER too high ({wer:.3f}):\nGT: {turkish_sample_jpeg.ground_truth_text}\nOCR: {api_text}"

    # Verify all special Turkish characters are present in the API text
    for tr_char in ["ç", "ğ", "ı", "ö", "ş", "ü", "Ç", "Ğ", "İ", "Ö", "Ş", "Ü"]:
        assert tr_char in api_text, f"Missing Turkish character {tr_char!r} in API text: {api_text}"

    # Download generated PDF/A and extract text independently via pypdfium2 AND pdftotext CLI
    pdf_bytes = client.get(f"/api/v1/jobs/{job_id}/pdf").content
    out_pdf = tmp_path / "verified_turkish.pdf"
    out_pdf.write_bytes(pdf_bytes)

    pdf_doc = pdfium.PdfDocument(str(out_pdf))
    try:
        page = pdf_doc[0]
        textpage = page.get_textpage()
        pdfium_extracted = textpage.get_text_range()
        textpage.close()
        page.close()
    finally:
        pdf_doc.close()

    for tr_char in ["ç", "ğ", "ı", "ö", "ş", "ü", "Ç", "Ğ", "İ", "Ö", "Ş", "Ü"]:
        assert tr_char in pdfium_extracted, (
            f"Missing Turkish character {tr_char!r} in PDF/A extracted text: {pdfium_extracted}"
        )

    if shutil.which("pdftotext"):
        proc = subprocess.run(
            ["pdftotext", "-enc", "UTF-8", str(out_pdf), "-"],
            capture_output=True,
            text=True,
            check=True,
        )
        pdftotext_out = proc.stdout
        for tr_char in ["ç", "ğ", "ı", "ö", "ş", "ü", "Ç", "Ğ", "İ", "Ö", "Ş", "Ü"]:
            assert tr_char in pdftotext_out


def test_multicolumn_turkish_newspaper_reading_order_end_to_end(
    app_and_worker,
    tmp_path: Path,
) -> None:
    client, worker, _app = app_and_worker

    newspaper_img = render_two_column_newspaper_image(
        tmp_path / "newspaper_2col.png",
        headline="CUMHURİYET ARŞİV GAZETESİ",
        left_column_lines=[
            "Birinci sütun ilk satır metni.",
            "Birinci sütun ikinci satır devamı.",
            "Birinci sütun üçüncü satır sonu.",
        ],
        right_column_lines=[
            "İkinci sütun ilk satır başlangıcı.",
            "İkinci sütun ikinci satır haberi.",
            "İkinci sütun üçüncü satır bitişi.",
        ],
    )

    with open(newspaper_img, "rb") as f:
        resp = client.post(
            "/api/v1/jobs",
            files=[("files", ("newspaper.png", f, "image/png"))],
            data={"language": "tur", "layout_mode": "auto"},
        )
    job_id = resp.json()["job_id"]
    worker.process_next_available_job()

    res = client.get(f"/api/v1/jobs/{job_id}/result").json()
    page = res["pages"][0]
    assert page["column_count"] == 2
    text = page["text"]

    # Spanning headline comes first, then entire First Column, then Second Column
    pos_headline = text.find("CUMHURİYET")
    pos_col1_end = text.find("üçüncü")
    pos_col2_start = text.find("başlangıcı")

    assert pos_headline != -1 and pos_col1_end != -1 and pos_col2_start != -1
    assert pos_headline < pos_col1_end < pos_col2_start
