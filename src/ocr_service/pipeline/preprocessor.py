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


# Tesseract misses or garbles glyphs much smaller than ~12 px. Low-resolution newspaper
# scans (a full broadsheet in ~1200 px, median glyph ~8 px) went from 56% to 91% word recall
# when the working copy was upscaled to ~18 px glyphs, while a crisp 9 px magazine page read
# 100% both before and after. Normal 300 DPI documents (glyphs 25-35 px) are never touched.
MIN_GLYPH_HEIGHT_PX = 12.0
TARGET_GLYPH_HEIGHT_PX = 18.0
MAX_UPSCALE_FACTOR = 3.0
MAX_WORKING_PIXELS = 60_000_000


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


def estimate_glyph_height_px(gray: np.ndarray) -> float:
    """Median height of letter-like connected components (0.0 when the page has none)."""
    _, inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    _, _, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=8)
    heights = stats[1:, cv2.CC_STAT_HEIGHT]
    widths = stats[1:, cv2.CC_STAT_WIDTH]
    # Drop speckle/halftone dots, rules, photos and long merged strokes.
    letter_like = (heights >= 4) & (heights <= 200) & (widths <= heights * 3) & (widths * 4 >= heights)
    return float(np.median(heights[letter_like])) if letter_like.any() else 0.0


def small_text_upscale_factor(gray: np.ndarray) -> float:
    """Factor to enlarge the working copy so small print reaches a Tesseract-friendly size."""
    glyph_h = estimate_glyph_height_px(gray)
    if glyph_h <= 0.0 or glyph_h >= MIN_GLYPH_HEIGHT_PX:
        return 1.0
    factor = min(MAX_UPSCALE_FACTOR, TARGET_GLYPH_HEIGHT_PX / glyph_h)
    h, w = gray.shape[:2]
    max_by_pixels = (MAX_WORKING_PIXELS / float(max(1, h * w))) ** 0.5
    factor = min(factor, max_by_pixels)
    return factor if factor >= 1.2 else 1.0


def suppress_light_color_marks(rgb: np.ndarray) -> np.ndarray:
    """Replace light, clearly coloured pixels (red/pink stamps, yellow highlights) with paper.

    Dark ink is never light-and-saturated, so text under a mark survives. Opt-in only:
    light coloured print (e.g. a red masthead) would be removed as well.
    """
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    red = ((h <= 12) | (h >= 160)) & (s >= 60) & (v >= 120)
    yellow = (h >= 18) & (h <= 40) & (s >= 90) & (v >= 150)
    mask = red | yellow
    if not mask.any():
        return rgb
    # Paper colour from light unmarked pixels only, so dense ink does not darken the fill.
    light = ~mask & (v >= 150)
    paper = (
        np.median(rgb[light].reshape(-1, 3), axis=0).astype(np.uint8)
        if light.any()
        else np.array([255, 255, 255], dtype=np.uint8)
    )
    out = rgb.copy()
    out[mask] = paper
    return out


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
        remove_color_marks: bool = False,
    ) -> PreparedWorkingPage:
        """Render or load source page into a grayscale contrast-normalized working PNG."""
        work_dir.mkdir(parents=True, exist_ok=True)
        gray_src, src_w, src_h, effective_dpi = self._load_grayscale_source(
            descriptor,
            target_working_dpi=target_working_dpi,
            remove_color_marks=remove_color_marks,
        )

        # Downsample if source resolution is significantly above target working DPI (>1.15x).
        # DPI metadata alone never triggers an upscale (FR-13); small print is enlarged below
        # based on the measured glyph height, on this working copy only.
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

        upscale = small_text_upscale_factor(gray_work)
        if upscale > 1.0:
            gray_work = cv2.resize(gray_work, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_CUBIC)
            scale_x = gray_work.shape[1] / float(src_w)
            scale_y = gray_work.shape[0] / float(src_h)
            # working_dpi is deliberately NOT multiplied: telling Tesseract 675 DPI for the
            # enlarged 1959 newspaper page dropped word recall from 86% to ~64%.
            logger.info(
                "Upscaled small-print working image",
                extra={"page_number": descriptor.page_number, "factor": round(upscale, 2)},
            )

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
        # Skipped for upscaled small print: on the yellowed 1959 newspaper scan it amplified the
        # paper texture and halved word recall (86% -> 43%); it changed nothing on the magazine
        # page or on 300 DPI pages, which keep it as before.
        if upscale > 1.0:
            enhanced = gray_work
        else:
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
        remove_color_marks: bool = False,
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

            gray = self._to_gray(pil_img, remove_color_marks)
            h, w = gray.shape[:2]
            return gray, w, h, float(target_working_dpi)

        with Image.open(descriptor.source_path) as img:
            gray = self._to_gray(img, remove_color_marks)
            h, w = gray.shape[:2]
            effective_dpi = float(max(descriptor.dpi_x, descriptor.dpi_y))
            return gray, w, h, effective_dpi

    @staticmethod
    def _to_gray(img: Image.Image, remove_color_marks: bool) -> np.ndarray:
        if remove_color_marks and img.mode not in ("1", "L", "LA", "I", "I;16", "F"):
            rgb = suppress_light_color_marks(np.array(img.convert("RGB"), dtype=np.uint8))
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        return np.array(img.convert("L"), dtype=np.uint8)
