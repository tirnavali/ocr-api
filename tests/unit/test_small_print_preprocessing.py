"""Small-print upscaling and opt-in colour-mark removal on the OCR working copy."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ocr_service.domain.models import BBoxPx, DpiSource, PageSourceDescriptor
from ocr_service.pipeline.geometry import working_bbox_to_source_bbox
from ocr_service.pipeline.preprocessor import (
    MAX_UPSCALE_FACTOR,
    WorkingImagePreprocessor,
    estimate_glyph_height_px,
    small_text_upscale_factor,
    suppress_light_color_marks,
)

FONT_PATH = Path(__file__).resolve().parents[2] / "resources" / "fonts" / "ArchiveUnicode-Regular.ttf"


def _text_page(font_size: int, size: tuple[int, int] = (900, 600)) -> np.ndarray:
    img = Image.new("L", size, color=255)
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(str(FONT_PATH), size=font_size)
    y = 20
    while y < size[1] - font_size * 2:
        draw.text((20, y), "Bandırmalının tek isteği var vilâyet olmak", fill=0, font=font)
        y += int(font_size * 1.6)
    return np.array(img)


def _descriptor(path: Path, w: int, h: int) -> PageSourceDescriptor:
    return PageSourceDescriptor(
        page_number=1,
        source_path=path,
        source_format="PNG",
        width_px=w,
        height_px=h,
        width_pt=w * 72 / 300,
        height_pt=h * 72 / 300,
        dpi_x=300,
        dpi_y=300,
        dpi_source=DpiSource.FALLBACK,
    )


def test_small_print_is_detected_and_upscaled_to_target() -> None:
    small = _text_page(11)
    glyph = estimate_glyph_height_px(small)
    assert 4 <= glyph < 12
    factor = small_text_upscale_factor(small)
    assert 1.2 <= factor <= MAX_UPSCALE_FACTOR
    assert 15 <= glyph * factor <= 19


def test_normal_print_is_left_alone() -> None:
    # 10 pt at 300 DPI is ~42 px font size: the common case must not change at all.
    assert small_text_upscale_factor(_text_page(42)) == 1.0
    assert small_text_upscale_factor(np.full((400, 400), 255, dtype=np.uint8)) == 1.0


def test_upscaled_working_copy_maps_words_back_to_source(tmp_path: Path) -> None:
    src = _text_page(11)
    path = tmp_path / "small.png"
    Image.fromarray(src).save(path)
    h, w = src.shape
    prepared = WorkingImagePreprocessor().prepare_working_page(
        _descriptor(path, w, h), tmp_path / "work", enable_deskew=False
    )

    t = prepared.transform
    assert t.scale_x > 1.2 and abs(t.scale_x - t.scale_y) < 0.01
    with Image.open(prepared.working_image_path) as work:
        assert work.size == (t.working_width_px, t.working_height_px)
    # A box in working pixels lands at the same spot of the untouched source image.
    back = working_bbox_to_source_bbox(
        BBoxPx(left=40 * t.scale_x, top=20 * t.scale_y, width=100 * t.scale_x, height=12 * t.scale_y), t
    )
    assert abs(back.left - 40) < 1 and abs(back.top - 20) < 1
    assert abs(back.width - 100) < 1 and abs(back.height - 12) < 1


def test_color_mark_removal_keeps_dark_ink_and_whitens_stamp_and_highlight() -> None:
    paper = (236, 226, 200)
    rgb = np.zeros((10, 50, 3), dtype=np.uint8)
    rgb[:] = paper  # columns 40..50 stay plain paper
    rgb[:, 0:10] = (230, 140, 140)  # light red stamp
    rgb[:, 10:20] = (250, 240, 60)  # yellow highlight
    rgb[:, 20:30] = (25, 20, 20)  # black ink
    rgb[:, 30:40] = (90, 20, 20)  # dark red ink under a stamp is still ink

    out = suppress_light_color_marks(rgb)

    assert (out[:, 0:20] == np.array(paper)).all()
    assert (out[:, 20:40] == rgb[:, 20:40]).all()


def test_color_mark_removal_is_a_noop_on_grayscale_scans() -> None:
    gray_rgb = np.repeat(_text_page(20)[..., None], 3, axis=2)
    assert (suppress_light_color_marks(gray_rgb) == gray_rgb).all()
