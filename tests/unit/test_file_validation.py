"""Unit tests for magic-byte file validation, filename sanitization, and DPI resolution."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from ocr_service.config import Settings
from ocr_service.domain.models import DpiSource
from ocr_service.exceptions import (
    CorruptedDocumentError,
    ImageDimensionsExceededError,
    UnsupportedMediaTypeError,
    ValidationError,
)
from ocr_service.infrastructure.storage.local_storage import LocalJobStorage, sanitize_filename
from ocr_service.pipeline.inspector import (
    DocumentInspector,
    detect_file_format_by_magic,
    has_usable_pdf_text,
    resolve_image_dpi,
)


def test_sanitize_filename_prevents_path_traversal() -> None:
    assert sanitize_filename("../../etc/passwd") == "passwd"
    assert sanitize_filename("..\\..\\Windows\\System32\\cmd.exe") == "cmd.exe"
    assert sanitize_filename("...hidden") == "hidden"
    assert sanitize_filename("") == "upload.bin"
    assert sanitize_filename("gazete arşiv 1923!.pdf") == "gazete_ar_iv_1923_.pdf"


def test_magic_byte_validation_rejects_spoofed_extension(tmp_path: Path) -> None:
    spoofed_pdf = tmp_path / "malicious.pdf"
    spoofed_pdf.write_bytes(b"MZ\x90\x00\x03\x00\x00\x00\x04\x00\x00\x00\xff\xff")

    with pytest.raises(UnsupportedMediaTypeError) as exc_info:
        detect_file_format_by_magic(spoofed_pdf)
    assert exc_info.value.code == "UNSUPPORTED_MEDIA_TYPE"
    assert exc_info.value.retryable is False


def test_magic_byte_validation_detects_supported_formats(
    turkish_sample_jpeg,
    turkish_sample_png,
    turkish_multipage_tiff: Path,
    scanned_image_pdf: Path,
) -> None:
    assert detect_file_format_by_magic(turkish_sample_jpeg.path) == "JPEG"
    assert detect_file_format_by_magic(turkish_sample_png.path) == "PNG"
    assert detect_file_format_by_magic(turkish_multipage_tiff) == "TIFF"
    assert detect_file_format_by_magic(scanned_image_pdf) == "PDF"


def test_dpi_resolution_distinguishes_declared_estimated_and_fallback(tmp_path: Path) -> None:
    # 1. Explicit 400 DPI declared
    p1 = tmp_path / "declared_400.png"
    Image.new("RGB", (800, 1000)).save(p1, format="PNG", dpi=(400, 400))
    with Image.open(p1) as img1:
        dx, dy, src = resolve_image_dpi(img1, default_dpi=300)
        assert round(dx) == 400
        assert round(dy) == 400
        assert src == DpiSource.DECLARED

    # 2. Bogus 72 DPI header on a high-res 2500x3500 archival image -> ESTIMATED
    p2 = tmp_path / "bogus_72.jpg"
    Image.new("RGB", (2500, 3500)).save(p2, format="JPEG", dpi=(72, 72))
    with Image.open(p2) as img2:
        dx, dy, src = resolve_image_dpi(img2, default_dpi=300)
        assert dx == 300.0
        assert src == DpiSource.ESTIMATED

    # 3. Missing DPI metadata -> FALLBACK
    p3 = tmp_path / "no_dpi.png"
    Image.new("RGB", (800, 1000)).save(p3, format="PNG")
    with Image.open(p3) as img3:
        img3.info.pop("dpi", None)
        dx, dy, src = resolve_image_dpi(img3, default_dpi=300)
        assert dx == 300.0
        assert src == DpiSource.FALLBACK


def test_inspector_rejects_oversized_image_dimensions(tmp_path: Path) -> None:
    settings = Settings(TEMP_STORAGE_PATH=tmp_path, MAX_IMAGE_PIXELS=1_000_000)
    inspector = DocumentInspector(settings)
    large_img = tmp_path / "too_many_pixels.png"
    Image.new("L", (1200, 1200)).save(large_img, format="PNG")

    with pytest.raises(ImageDimensionsExceededError):
        inspector.inspect_inputs([large_img], work_dir=tmp_path / "work")


def test_inspector_rejects_corrupted_image(tmp_path: Path, test_settings: Settings) -> None:
    inspector = DocumentInspector(test_settings)
    bad_jpg = tmp_path / "corrupted.jpg"
    bad_jpg.write_bytes(b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01corrupted_stream_data")

    with pytest.raises(CorruptedDocumentError):
        inspector.inspect_inputs([bad_jpg], work_dir=tmp_path / "work")


def test_storage_validates_job_id_against_traversal(test_settings: Settings) -> None:
    storage = LocalJobStorage(test_settings)
    with pytest.raises(ValidationError):
        storage.get_job_paths("../escape")


def test_usable_pdf_text_detector() -> None:
    assert has_usable_pdf_text("Türkiye Büyük Millet Meclisi zabıt ceridesi arşiv metni.") is True
    assert has_usable_pdf_text("   short   ") is False
    assert has_usable_pdf_text("\ufffd" * 40) is False
