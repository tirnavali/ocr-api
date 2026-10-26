"""Shared pytest fixtures generating realistic Turkish archival images, TIFFs, and PDFs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pikepdf
import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw, ImageFont

from ocr_service.config import PROJECT_ROOT, Settings
from ocr_service.domain.models import BBoxPt, BBoxPx
from ocr_service.main import create_app
from ocr_service.worker_main import OcrWorker

FONT_PATH = PROJECT_ROOT / "resources" / "fonts" / "ArchiveUnicode-Regular.ttf"


@dataclass(slots=True)
class GroundTruthWord:
    text: str
    bbox_px: BBoxPx
    bbox_pt: BBoxPt


@dataclass(slots=True)
class SyntheticSampleDocument:
    path: Path
    dpi: int
    width_px: int
    height_px: int
    ground_truth_text: str
    words: list[GroundTruthWord]


def render_turkish_archival_image(
    output_path: Path,
    lines: list[str],
    *,
    width_px: int = 1800,
    height_px: int = 2400,
    dpi: int = 300,
    font_size_px: int = 44,
    fmt: str = "JPEG",
) -> SyntheticSampleDocument:
    """Render crisp Turkish archival text onto a page image and record exact word bounding boxes."""
    img = Image.new("RGB", (width_px, height_px), color=(252, 250, 245))
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(str(FONT_PATH), size=font_size_px)

    margin_x = 140
    cursor_y = 160
    line_spacing = int(font_size_px * 1.75)
    gt_words: list[GroundTruthWord] = []

    space_bbox = draw.textbbox((0, 0), " ", font=font)
    space_w = max(12, space_bbox[2] - space_bbox[0])

    for line in lines:
        cursor_x = margin_x
        for word in line.split():
            draw.text((cursor_x, cursor_y), word, fill=(15, 15, 15), font=font)
            wb = draw.textbbox((cursor_x, cursor_y), word, font=font)
            left, top, right, bottom = float(wb[0]), float(wb[1]), float(wb[2]), float(wb[3])
            w_px = right - left
            h_px = bottom - top

            bbox_px = BBoxPx(left=left, top=top, width=w_px, height=h_px)
            x_pt = left * 72.0 / dpi
            w_pt = w_px * 72.0 / dpi
            h_pt = h_px * 72.0 / dpi
            y_pt = (height_px - bottom) * 72.0 / dpi
            bbox_pt = BBoxPt(x=x_pt, y=y_pt, width=w_pt, height=h_pt)

            gt_words.append(GroundTruthWord(text=word, bbox_px=bbox_px, bbox_pt=bbox_pt))
            cursor_x = int(right + space_w)
        cursor_y += line_spacing

    save_kwargs: dict[str, object] = {"dpi": (dpi, dpi)}
    if fmt.upper() == "JPEG":
        save_kwargs["quality"] = 95
    img.save(output_path, format=fmt, **save_kwargs)

    return SyntheticSampleDocument(
        path=output_path,
        dpi=dpi,
        width_px=width_px,
        height_px=height_px,
        ground_truth_text="\n".join(lines),
        words=gt_words,
    )


def render_two_column_newspaper_image(
    output_path: Path,
    *,
    headline: str,
    left_column_lines: list[str],
    right_column_lines: list[str],
    width_px: int = 2200,
    height_px: int = 2600,
    dpi: int = 300,
) -> Path:
    """Render a 2-column Turkish newspaper front page with a spanning banner headline."""
    img = Image.new("RGB", (width_px, height_px), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    headline_font = ImageFont.truetype(str(FONT_PATH), size=64)
    body_font = ImageFont.truetype(str(FONT_PATH), size=38)

    # Draw spanning banner headline across both columns
    draw.text((160, 140), headline, fill=(10, 10, 10), font=headline_font)

    # Left column: x = 160..1000
    y_left = 340
    for line in left_column_lines:
        draw.text((160, y_left), line, fill=(15, 15, 15), font=body_font)
        y_left += 72

    # Right column: x = 1220..2040 (clear 220px gutter)
    y_right = 340
    for line in right_column_lines:
        draw.text((1220, y_right), line, fill=(15, 15, 15), font=body_font)
        y_right += 72

    img.save(output_path, format="PNG", dpi=(dpi, dpi))
    return output_path


@pytest.fixture()
def test_settings(tmp_path: Path) -> Settings:
    storage_dir = tmp_path / "ocr_data"
    storage_dir.mkdir(parents=True, exist_ok=True)
    return Settings(
        APP_ENV="testing",
        LOG_JSON=False,
        LOG_LEVEL="WARNING",
        TEMP_STORAGE_PATH=storage_dir,
        OCR_LANGUAGE="tur",
        OCR_WORKER_COUNT=1,
        OCR_THREADS_PER_WORKER=1,
        OCR_WORKING_DPI=300,
        JOB_MAX_RETRIES=2,
        JOB_TIMEOUT_SECONDS=120,
        RESULT_RETENTION_HOURS=24.0,
        MAX_UPLOAD_MB=50,
    )


@pytest.fixture()
def app_and_worker(test_settings: Settings):
    app = create_app(test_settings)
    with TestClient(app) as client:
        worker = OcrWorker(
            settings=test_settings,
            repo=app.state.repo,
            storage=app.state.storage,
            worker_id="test-worker-01",
        )
        yield client, worker, app


@pytest.fixture()
def turkish_sample_jpeg(tmp_path: Path) -> SyntheticSampleDocument:
    lines = [
        "TÜRKİYE BÜYÜK MİLLET MECLİSİ ARŞİVİ",
        "Cumhuriyet dönemi gazeteleri ve tarihi belgeler.",
        "Özel Türkçe karakterler: çığöşü ÇİĞÖŞÜ",
        "İstanbul, Diyarbakır, Eskişehir, Gümüşhane, Şanlıurfa.",
    ]
    out = tmp_path / "turkish_sample.jpg"
    return render_turkish_archival_image(out, lines, fmt="JPEG", dpi=300)


@pytest.fixture()
def turkish_sample_png(tmp_path: Path) -> SyntheticSampleDocument:
    lines = [
        "MİLLİ KÜTÜPHANE SÜRELİ YAYINLAR KOLEKSİYONU",
        "Doğruluk, güvenlik ve yüksek çözünürlük korunmaktadır.",
    ]
    out = tmp_path / "turkish_sample.png"
    return render_turkish_archival_image(out, lines, fmt="PNG", dpi=300)


@pytest.fixture()
def turkish_multipage_tiff(tmp_path: Path) -> Path:
    page1 = render_turkish_archival_image(
        tmp_path / "t1.png",
        ["BİRİNCİ SAYFA: Tarihî Arşiv Vesikası", "Çağdaş arşivleme sistemleri."],
        width_px=1600,
        height_px=1200,
        dpi=300,
        fmt="PNG",
    )
    page2 = render_turkish_archival_image(
        tmp_path / "t2.png",
        ["İKİNCİ SAYFA: Meclis Zabıt Ceridesi", "Görüşmeler ve kararlar özeti."],
        width_px=1800,
        height_px=1400,
        dpi=300,
        fmt="PNG",
    )
    tiff_path = tmp_path / "multipage_archival.tiff"
    with Image.open(page1.path) as im1, Image.open(page2.path) as im2:
        im1.save(
            tiff_path,
            format="TIFF",
            save_all=True,
            append_images=[im2],
            compression="tiff_deflate",
            dpi=(300, 300),
        )
    return tiff_path


@pytest.fixture()
def scanned_image_pdf(tmp_path: Path, turkish_sample_jpeg: SyntheticSampleDocument) -> Path:
    """Create a 1-page image-only PDF (no text layer) containing the Turkish sample JPEG."""
    pdf_path = tmp_path / "scanned_input.pdf"
    pdf = pikepdf.Pdf.new()
    raw_jpeg = turkish_sample_jpeg.path.read_bytes()
    w_px = turkish_sample_jpeg.width_px
    h_px = turkish_sample_jpeg.height_px
    w_pt = w_px * 72.0 / turkish_sample_jpeg.dpi
    h_pt = h_px * 72.0 / turkish_sample_jpeg.dpi

    im_stream = pdf.make_indirect(
        pikepdf.Stream(
            pdf,
            raw_jpeg,
            {
                "/Type": pikepdf.Name("/XObject"),
                "/Subtype": pikepdf.Name("/Image"),
                "/Width": w_px,
                "/Height": h_px,
                "/ColorSpace": pikepdf.Name("/DeviceRGB"),
                "/BitsPerComponent": 8,
                "/Filter": pikepdf.Name("/DCTDecode"),
            },
        )
    )
    content = f"q\n{w_pt:.2f} 0 0 {h_pt:.2f} 0 0 cm\n/Im0 Do\nQ\n".encode("ascii")
    page_dict = pikepdf.Dictionary(
        {
            "/Type": pikepdf.Name("/Page"),
            "/MediaBox": pikepdf.Array([0, 0, w_pt, h_pt]),
            "/Resources": pikepdf.Dictionary(
                {"/XObject": pikepdf.Dictionary({"/Im0": im_stream})}
            ),
            "/Contents": pdf.make_indirect(pikepdf.Stream(pdf, content)),
        }
    )
    pdf.pages.append(pikepdf.Page(page_dict))
    pdf.save(pdf_path)
    pdf.close()
    return pdf_path
