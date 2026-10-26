"""Working image preparation (Track B) decoupled from archival image preservation (Track A)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pypdfium2 as pdfium
from PIL import Image

from ocr_service.domain.models import PageSourceDescriptor
from ocr_service.pipeline.geometry import WorkingToSourceTransform

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class PreparedWorkingPage:
    """Result of preparing an ephemeral working image for OCR recognition."""

    working_image_path: Path
    working_dpi: int
    transform: WorkingToSourceTransform


def estimate_skew_angle_deg(gray: np.ndarray, max_skew_deg: float = 6.0) -> float:
    """Estimate document skew angle in degrees using morphological text line orientation."""
    h, w = gray.shape[:2]
    if h < 100 or w < 100:
        return 0.0

    # Downsample for fast skew estimation (<15 ms on CPU)
    max_dim = 1200
    scale = min(1.0, max_dim / float(max(h, w)))
    if scale < 1.0:
        small = cv2.resize(
            gray,
            (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_AREA,
        )
    else:
        small = gray

    # Invert binary Otsu so text pixels are foreground (255)
    _, thresh = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Connect characters horizontally into text lines
    kernel_w = max(15, small.shape[1] // 40)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_w, 3))
    connected = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    angles: list[float] = []
    weights: list[float] = []

    min_line_w = max(30, small.shape[1] // 20)
    for cnt in contours:
        _x, _y, cw, ch = cv2.boundingRect(cnt)
        if cw < min_line_w or ch < 4 or cw < ch * 2.5:
            continue
        rect = cv2.minAreaRect(cnt)
        (_, _), (rw, rh), angle = rect
        if rw < rh:
            angle = angle + 90.0
        if angle > 45.0:
            angle -= 90.0
        elif angle < -45.0:
            angle += 90.0

        if -max_skew_deg <= angle <= max_skew_deg:
            angles.append(float(angle))
            weights.append(float(cw))

    if len(angles) < 3:
        return 0.0

    # Weighted median skew angle
    order = np.argsort(angles)
    sorted_angles = np.array(angles)[order]
    sorted_weights = np.array(weights)[order]
    cum_weights = np.cumsum(sorted_weights)
    median_idx = int(np.searchsorted(cum_weights, cum_weights[-1] / 2.0))
    estimated = float(sorted_angles[min(median_idx, len(sorted_angles) - 1)])

    if abs(estimated) < 0.15:
        return 0.0
    return estimated


def rotate_working_image_for_deskew(gray: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate working image by angle_deg around center with white background fill."""
    h, w = gray.shape[:2]
    center = (w / 2.0, h / 2.0)
    mat = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    return cv2.warpAffine(
        gray,
        mat,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255,
    )


def rotate_cardinal_cw(gray: np.ndarray, degrees_cw: int) -> np.ndarray:
    """Rotate image clockwise by 0, 90, 180, or 270 degrees."""
    rot = degrees_cw % 360
    if rot == 90:
        return cv2.rotate(gray, cv2.ROTATE_90_CLOCKWISE)
    if rot == 180:
        return cv2.rotate(gray, cv2.ROTATE_180)
    if rot == 270:
        return cv2.rotate(gray, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return gray


class WorkingImagePreprocessor:
    """Create OCR-optimized working images without modifying archival source images (FR-13, FR-14)."""

    def prepare_working_page(
        self,
        descriptor: PageSourceDescriptor,
        work_dir: Path,
        *,
        target_working_dpi: int = 300,
        enable_deskew: bool = True,
        cardinal_rotation_cw: int = 0,
    ) -> PreparedWorkingPage:
        """Render or load source page into a grayscale contrast-normalized working PNG."""
        work_dir.mkdir(parents=True, exist_ok=True)
        gray_src, src_w, src_h, effective_dpi = self._load_grayscale_source(
            descriptor, target_working_dpi=target_working_dpi
        )

        # Downsample ONLY if source resolution is significantly above target working DPI (>1.15x)
        # Never upscale low-resolution images with naive interpolation (FR-13).
        scale_x = 1.0
        scale_y = 1.0
        if effective_dpi > target_working_dpi * 1.15 and max(src_w, src_h) > 1600:
            ratio = float(target_working_dpi) / float(effective_dpi)
            new_w = max(1, round(src_w * ratio))
            new_h = max(1, round(src_h * ratio))
            gray_work = cv2.resize(gray_src, (new_w, new_h), interpolation=cv2.INTER_AREA)
            scale_x = new_w / float(src_w)
            scale_y = new_h / float(src_h)
            working_dpi = target_working_dpi
        else:
            gray_work = gray_src
            working_dpi = round(effective_dpi)

        # Apply optional cardinal rotation to working image if page orientation requires it
        if cardinal_rotation_cw % 360 != 0:
            gray_work = rotate_cardinal_cw(gray_work, cardinal_rotation_cw)

        # Estimate and correct slight skew on working copy only
        deskew_angle = 0.0
        if enable_deskew:
            deskew_angle = estimate_skew_angle_deg(gray_work)
            if abs(deskew_angle) >= 0.15:
                gray_work = rotate_working_image_for_deskew(gray_work, deskew_angle)

        # Gentle illumination normalization (CLAHE) to enhance degraded archival newspaper contrast
        # while preserving fine Turkish diacritics (dots on İ/i, cedilla on ç/ş, breve on ğ).
        clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(16, 16))
        enhanced = clahe.apply(gray_work)

        work_h, work_w = enhanced.shape[:2]
        out_path = work_dir / f"working_page_{descriptor.page_number:04d}.png"

        pil_work = Image.fromarray(enhanced)
        pil_work.save(out_path, format="PNG", compress_level=1, dpi=(working_dpi, working_dpi))

        transform = WorkingToSourceTransform(
            source_width_px=src_w,
            source_height_px=src_h,
            working_width_px=work_w,
            working_height_px=work_h,
            scale_x=scale_x,
            scale_y=scale_y,
            deskew_angle_deg=deskew_angle,
            cardinal_rotation_deg=cardinal_rotation_cw % 360,
        )

        return PreparedWorkingPage(
            working_image_path=out_path,
            working_dpi=working_dpi,
            transform=transform,
        )

    def _load_grayscale_source(
        self,
        descriptor: PageSourceDescriptor,
        target_working_dpi: int,
    ) -> tuple[np.ndarray, int, int, float]:
        if descriptor.source_format == "PDF_PAGE":
            assert descriptor.source_pdf_page_index is not None
            pdf_doc = pdfium.PdfDocument(str(descriptor.source_path))
            try:
                page = pdf_doc[descriptor.source_pdf_page_index]
                scale = target_working_dpi / 72.0
                bitmap = page.render(scale=scale, rotation=0)
                pil_img = bitmap.to_pil()
                bitmap.close()
                page.close()
            finally:
                pdf_doc.close()

            gray = np.array(pil_img.convert("L"), dtype=np.uint8)
            h, w = gray.shape[:2]
            return gray, w, h, float(target_working_dpi)

        with Image.open(descriptor.source_path) as img:
            gray_pil = img.convert("L")
            gray = np.array(gray_pil, dtype=np.uint8)
            h, w = gray.shape[:2]
            effective_dpi = float(max(descriptor.dpi_x, descriptor.dpi_y))
            return gray, w, h, effective_dpi
