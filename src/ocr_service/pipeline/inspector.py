"""Magic-byte file validation, DPI extraction, multipage TIFF & PDF text-layer inspection."""

from __future__ import annotations

import logging
import re
from pathlib import Path

import pikepdf
import pypdfium2 as pdfium
from PIL import Image, UnidentifiedImageError

from ocr_service.config import Settings
from ocr_service.domain.models import DpiSource, PageSourceDescriptor
from ocr_service.exceptions import (
    CorruptedDocumentError,
    ImageDimensionsExceededError,
    PageLimitExceededError,
    PasswordProtectedPdfError,
    UnsupportedMediaTypeError,
    ValidationError,
)

logger = logging.getLogger(__name__)

# Standard archival A4 height in inches (297mm = 11.69 inches) used only for plausible DPI estimation
ARCHIVAL_PRINTABLE_CHAR_RE = re.compile(r"[\wçğıöşüÇĞİÖŞÜ.,:;!?()-]", re.UNICODE)


def detect_file_format_by_magic(file_path: Path) -> str:
    """Inspect magic bytes to determine real format (JPEG, PNG, TIFF, PDF) (FR-03, FR-17)."""
    try:
        with open(file_path, "rb") as f:
            header = f.read(16)
    except OSError as exc:
        raise CorruptedDocumentError(f"Unable to read uploaded file: {exc}") from exc

    if len(header) < 4:
        raise UnsupportedMediaTypeError(
            f"File {file_path.name!r} is too small to be a valid image or PDF.",
            details={"filename": file_path.name},
        )

    if header.startswith(b"\xff\xd8\xff"):
        return "JPEG"
    if header.startswith(b"\x89PNG\r\n\x1a\n"):
        return "PNG"
    if header.startswith((b"II*\x00", b"MM\x00*")):
        return "TIFF"
    if header.startswith(b"%PDF-"):
        return "PDF"

    raise UnsupportedMediaTypeError(
        f"Unsupported file format for {file_path.name!r}. "
        "Supported formats: JPEG, PNG, TIFF (single/multipage), and PDF.",
        details={"filename": file_path.name, "header_hex": header[:8].hex()},
    )


def resolve_image_dpi(
    img: Image.Image,
    default_dpi: int = 300,
) -> tuple[float, float, DpiSource]:
    """Read declared DPI from image metadata and distinguish missing/implausible values (FR-13)."""
    raw_dpi = img.info.get("dpi")
    if isinstance(raw_dpi, (tuple, list)) and len(raw_dpi) >= 2:
        try:
            dx = float(raw_dpi[0])
            dy = float(raw_dpi[1])
            if 72.0 <= dx <= 2400.0 and 72.0 <= dy <= 2400.0:
                # Special check: many scanners/tools write default 72 DPI on high-res >2000px images
                if dx == 72.0 and dy == 72.0 and max(img.width, img.height) >= 2200:
                    return float(default_dpi), float(default_dpi), DpiSource.ESTIMATED
                return dx, dy, DpiSource.DECLARED
        except (TypeError, ValueError):
            pass

    # Check TIFF XResolution / YResolution tags (282, 283) if available
    tag_v2 = getattr(img, "tag_v2", None)
    if tag_v2 is not None:
        x_res = tag_v2.get(282)
        y_res = tag_v2.get(283)
        res_unit = tag_v2.get(296, 2)  # 2 = inch, 3 = cm
        if x_res and y_res:
            try:
                dx = float(x_res)
                dy = float(y_res)
                if res_unit == 3:
                    dx *= 2.54
                    dy *= 2.54
                if 72.0 <= dx <= 2400.0 and 72.0 <= dy <= 2400.0:
                    return dx, dy, DpiSource.DECLARED
            except (TypeError, ValueError, ZeroDivisionError):
                pass

    return float(default_dpi), float(default_dpi), DpiSource.FALLBACK


def has_usable_pdf_text(text: str, min_chars: int = 25) -> bool:
    """Determine if extracted PDF page text is genuine usable text vs. empty/mojibake (FR-03)."""
    stripped = text.strip()
    if len(stripped) < min_chars:
        return False
    # Reject replacement character or Private Use Area dominance
    if "\ufffd" in stripped and (stripped.count("\ufffd") / len(stripped)) > 0.05:
        return False
    valid_matches = ARCHIVAL_PRINTABLE_CHAR_RE.findall(stripped)
    ratio = len(valid_matches) / len(stripped)
    return ratio >= 0.65


class DocumentInspector:
    """Validate uploaded files and expand them into normalized PageSourceDescriptors."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        Image.MAX_IMAGE_PIXELS = settings.max_image_pixels

    def inspect_inputs(
        self,
        input_paths: list[Path],
        work_dir: Path,
        *,
        force_ocr: bool = False,
        working_dpi: int = 300,
    ) -> list[PageSourceDescriptor]:
        """Validate all input files and return ordered page descriptors."""
        if not input_paths:
            raise ValidationError("No input files provided.", code="NO_INPUT_FILES")

        detected_formats = [detect_file_format_by_magic(p) for p in input_paths]
        if "PDF" in detected_formats and len(input_paths) > 1:
            raise ValidationError(
                "Please submit either a single PDF document or one/more image files (JPEG/PNG/TIFF).",
                code="MIXED_PDF_AND_IMAGE_INPUTS",
            )

        descriptors: list[PageSourceDescriptor] = []
        for path, fmt in zip(input_paths, detected_formats, strict=True):
            if fmt == "PDF":
                descriptors.extend(
                    self._inspect_pdf(
                        path,
                        start_page_number=len(descriptors) + 1,
                        force_ocr=force_ocr,
                        working_dpi=working_dpi,
                    )
                )
            elif fmt == "TIFF":
                descriptors.extend(
                    self._inspect_tiff(
                        path,
                        work_dir=work_dir,
                        start_page_number=len(descriptors) + 1,
                    )
                )
            else:
                descriptors.append(
                    self._inspect_single_image(
                        path,
                        fmt=fmt,
                        page_number=len(descriptors) + 1,
                    )
                )

            if len(descriptors) > self._settings.max_pages_per_job:
                raise PageLimitExceededError(
                    f"Document exceeds maximum allowed pages ({self._settings.max_pages_per_job}).",
                    details={"max_pages": self._settings.max_pages_per_job},
                )

        if not descriptors:
            raise CorruptedDocumentError("Input document contains 0 pages.")

        return descriptors

    def _inspect_single_image(
        self,
        image_path: Path,
        fmt: str,
        page_number: int,
    ) -> PageSourceDescriptor:
        try:
            with Image.open(image_path) as img:
                img.verify()
            with Image.open(image_path) as img:
                width_px, height_px = img.size
                if width_px <= 0 or height_px <= 0:
                    raise CorruptedDocumentError(f"Image {image_path.name!r} has invalid dimensions.")
                if width_px * height_px > self._settings.max_image_pixels:
                    raise ImageDimensionsExceededError(
                        f"Image {image_path.name!r} ({width_px}x{height_px}) exceeds maximum pixel limit "
                        f"({self._settings.max_image_pixels} pixels).",
                        details={"width": width_px, "height": height_px},
                    )
                dpi_x, dpi_y, dpi_source = resolve_image_dpi(
                    img, default_dpi=self._settings.default_assumed_dpi
                )
                color_mode = img.mode
        except Image.DecompressionBombError as exc:
            raise ImageDimensionsExceededError(
                f"Image {image_path.name!r} exceeds safe decompression pixel limits."
            ) from exc
        except (UnidentifiedImageError, OSError, SyntaxError) as exc:
            raise CorruptedDocumentError(
                f"Corrupted or unreadable image file {image_path.name!r}: {exc}"
            ) from exc

        width_pt = width_px * 72.0 / dpi_x
        height_pt = height_px * 72.0 / dpi_y

        return PageSourceDescriptor(
            page_number=page_number,
            source_path=image_path,
            source_format=fmt,
            width_px=width_px,
            height_px=height_px,
            width_pt=width_pt,
            height_pt=height_pt,
            dpi_x=dpi_x,
            dpi_y=dpi_y,
            dpi_source=dpi_source,
            rotation_deg=0,
            has_usable_text=False,
            color_mode=color_mode,
        )

    def _inspect_tiff(
        self,
        tiff_path: Path,
        work_dir: Path,
        start_page_number: int,
    ) -> list[PageSourceDescriptor]:
        """Inspect single or multipage TIFF and extract individual lossless frames when multipage."""
        descriptors: list[PageSourceDescriptor] = []
        frames_dir = work_dir / "tiff_frames"
        frames_dir.mkdir(parents=True, exist_ok=True)

        try:
            with Image.open(tiff_path) as img:
                n_frames = getattr(img, "n_frames", 1)
                if n_frames <= 0:
                    raise CorruptedDocumentError(f"TIFF {tiff_path.name!r} contains 0 frames.")
                if start_page_number + n_frames - 1 > self._settings.max_pages_per_job:
                    raise PageLimitExceededError(
                        f"Multipage TIFF has {n_frames} frames, exceeding limit of {self._settings.max_pages_per_job}."
                    )

                for frame_idx in range(n_frames):
                    img.seek(frame_idx)
                    width_px, height_px = img.size
                    if width_px * height_px > self._settings.max_image_pixels:
                        raise ImageDimensionsExceededError(
                            f"TIFF frame {frame_idx + 1} ({width_px}x{height_px}) exceeds pixel limit."
                        )
                    dpi_x, dpi_y, dpi_source = resolve_image_dpi(
                        img, default_dpi=self._settings.default_assumed_dpi
                    )
                    color_mode = img.mode
                    page_num = start_page_number + frame_idx

                    # If single-frame TIFF, reference original file; if multipage, save lossless frame PNG
                    if n_frames == 1:
                        frame_source_path = tiff_path
                        frame_fmt = "TIFF"
                    else:
                        frame_source_path = frames_dir / f"page_{page_num:04d}.png"
                        frame_copy = img.copy()
                        frame_copy.save(
                            frame_source_path,
                            format="PNG",
                            compress_level=1,
                            dpi=(dpi_x, dpi_y),
                        )
                        frame_fmt = "PNG"

                    width_pt = width_px * 72.0 / dpi_x
                    height_pt = height_px * 72.0 / dpi_y

                    descriptors.append(
                        PageSourceDescriptor(
                            page_number=page_num,
                            source_path=frame_source_path,
                            source_format=frame_fmt,
                            width_px=width_px,
                            height_px=height_px,
                            width_pt=width_pt,
                            height_pt=height_pt,
                            dpi_x=dpi_x,
                            dpi_y=dpi_y,
                            dpi_source=dpi_source,
                            rotation_deg=0,
                            has_usable_text=False,
                            tiff_frame_index=frame_idx,
                            color_mode=color_mode,
                        )
                    )
        except (ImageDimensionsExceededError, PageLimitExceededError):
            raise
        except Image.DecompressionBombError as exc:
            raise ImageDimensionsExceededError(
                f"TIFF {tiff_path.name!r} exceeds safe decompression pixel limits."
            ) from exc
        except (UnidentifiedImageError, OSError, SyntaxError, EOFError) as exc:
            raise CorruptedDocumentError(
                f"Corrupted or unreadable TIFF file {tiff_path.name!r}: {exc}"
            ) from exc

        return descriptors

    def _inspect_pdf(
        self,
        pdf_path: Path,
        start_page_number: int,
        force_ocr: bool,
        working_dpi: int,
    ) -> list[PageSourceDescriptor]:
        """Inspect PDF pages, check encryption, page geometry, and existing text layers."""
        try:
            with pikepdf.open(pdf_path) as pdf:
                if pdf.is_encrypted:
                    raise PasswordProtectedPdfError(
                        f"PDF {pdf_path.name!r} is encrypted / password-protected."
                    )
                num_pages = len(pdf.pages)
                if num_pages == 0:
                    raise CorruptedDocumentError(f"PDF {pdf_path.name!r} contains 0 pages.")
                if start_page_number + num_pages - 1 > self._settings.max_pages_per_job:
                    raise PageLimitExceededError(
                        f"PDF has {num_pages} pages, exceeding limit of {self._settings.max_pages_per_job}."
                    )

                page_boxes: list[tuple[float, float, int]] = []
                for page in pdf.pages:
                    mbox = page.mediabox
                    w_pt = float(mbox[2]) - float(mbox[0])
                    h_pt = float(mbox[3]) - float(mbox[1])
                    rot = int(page.get("/Rotate", 0)) % 360
                    if rot in (90, 270):
                        w_pt, h_pt = h_pt, w_pt
                    if w_pt <= 0 or h_pt <= 0:
                        raise CorruptedDocumentError("PDF page has non-positive MediaBox dimensions.")
                    page_boxes.append((w_pt, h_pt, rot))
        except pikepdf.PasswordError as exc:
            raise PasswordProtectedPdfError(
                f"PDF {pdf_path.name!r} requires a password to open."
            ) from exc
        except (PasswordProtectedPdfError, PageLimitExceededError, CorruptedDocumentError):
            raise
        except pikepdf.PdfError as exc:
            raise CorruptedDocumentError(
                f"Invalid or corrupted PDF file {pdf_path.name!r}: {exc}"
            ) from exc

        descriptors: list[PageSourceDescriptor] = []
        try:
            pdfium_doc = pdfium.PdfDocument(str(pdf_path))
            try:
                for idx, (w_pt, h_pt, rot) in enumerate(page_boxes):
                    page_num = start_page_number + idx
                    existing_text = ""
                    usable = False
                    if not force_ocr:
                        pdfium_page = pdfium_doc[idx]
                        textpage = pdfium_page.get_textpage()
                        existing_text = textpage.get_text_bounded() or ""
                        textpage.close()
                        pdfium_page.close()
                        usable = has_usable_pdf_text(existing_text)

                    # Effective pixel dimensions at target working_dpi
                    w_px = max(1, round(w_pt * working_dpi / 72.0))
                    h_px = max(1, round(h_pt * working_dpi / 72.0))
                    if w_px * h_px > self._settings.max_image_pixels:
                        raise ImageDimensionsExceededError(
                            f"PDF page {idx + 1} at {working_dpi} DPI ({w_px}x{h_px}) exceeds pixel limit."
                        )

                    descriptors.append(
                        PageSourceDescriptor(
                            page_number=page_num,
                            source_path=pdf_path,
                            source_format="PDF_PAGE",
                            width_px=w_px,
                            height_px=h_px,
                            width_pt=w_pt,
                            height_pt=h_pt,
                            dpi_x=float(working_dpi),
                            dpi_y=float(working_dpi),
                            dpi_source=DpiSource.PDF_USER_SPACE,
                            rotation_deg=rot,
                            has_usable_text=usable,
                            existing_text=existing_text.replace("\r\n", "\n").strip(),
                            source_pdf_page_index=idx,
                            color_mode="RGB",
                        )
                    )
            finally:
                pdfium_doc.close()
        except ImageDimensionsExceededError:
            raise
        except Exception as exc:
            raise CorruptedDocumentError(
                f"Failed to inspect PDF pages in {pdf_path.name!r}: {exc}"
            ) from exc

        return descriptors
