"""Unit tests for multi-column newspaper layout segmentation and reading order (FR-11)."""

from __future__ import annotations

from ocr_service.domain.models import BBoxPx, LayoutMode, WordBox
from ocr_service.pipeline.layout import analyze_layout_and_reading_order


def _make_word(text: str, left: float, top: float, width: float = 120.0, height: float = 32.0) -> WordBox:
    box = BBoxPx(left=left, top=top, width=width, height=height)
    return WordBox(
        text=text,
        confidence=95.0,
        bbox_working_px=box,
        bbox_source_px=box,
    )


def test_two_column_newspaper_with_spanning_headline_reading_order() -> None:
    """Verify spanning banner headline is read first, followed by Column 1 top-to-bottom, then Column 2."""
    page_w, page_h = 2000, 2800

    # Spanning headline across x=200..1750 at y=150
    headline_words = [
        _make_word("CUMHURİYET", 200, 150, width=450, height=60),
        _make_word("GAZETESİ", 700, 150, width=420, height=60),
        _make_word("ARŞİVİ", 1180, 150, width=400, height=60),
    ]

    # Left column (x=200..800) at y=350, 420, 490
    col1_words = [
        _make_word("Sol_Satır1_Kelime1", 200, 350, width=250),
        _make_word("Sol_Satır1_Kelime2", 480, 350, width=250),
        _make_word("Sol_Satır2_Kelime1", 200, 420, width=250),
        _make_word("Sol_Satır2_Kelime2", 480, 420, width=250),
        _make_word("Sol_Satır3_Kelime1", 200, 490, width=250),
    ]

    # Right column (x=1150..1750) at y=350, 420, 490
    col2_words = [
        _make_word("Sağ_Satır1_Kelime1", 1150, 350, width=250),
        _make_word("Sağ_Satır1_Kelime2", 1430, 350, width=250),
        _make_word("Sağ_Satır2_Kelime1", 1150, 420, width=250),
        _make_word("Sağ_Satır2_Kelime2", 1430, 420, width=250),
        _make_word("Sağ_Satır3_Kelime1", 1150, 490, width=250),
    ]

    # Interleave input words in raw top-to-bottom Y scan order (simulating naive row-wise OCR)
    scrambled = (
        [*headline_words, col1_words[0], col1_words[1], col2_words[0], col2_words[1], col1_words[2], col1_words[3], col2_words[2], col2_words[3], col1_words[4], col2_words[4]]
    )

    result = analyze_layout_and_reading_order(
        scrambled,
        page_width_px=page_w,
        page_height_px=page_h,
        layout_mode=LayoutMode.AUTO,
    )

    assert result.column_count == 2
    text = result.logical_text

    # Headline must come first, then all of Left Column, then all of Right Column!
    idx_headline = text.index("CUMHURİYET GAZETESİ ARŞİVİ")
    idx_left_end = text.index("Sol_Satır3_Kelime1")
    idx_right_start = text.index("Sağ_Satır1_Kelime1")

    assert idx_headline < idx_left_end < idx_right_start

    # Ensure word geometric coordinates were never modified by reading-order sorting
    for w in result.ordered_words:
        if w.text == "Sağ_Satır1_Kelime1":
            assert w.bbox_source_px.left == 1150.0
            assert w.bbox_source_px.top == 350.0


def _tess_word(
    text: str, left: float, top: float, block: int, line: int, width: float = 60.0, height: float = 14.0
) -> WordBox:
    box = BBoxPx(left=left, top=top, width=width, height=height)
    return WordBox(
        text=text,
        confidence=90.0,
        bbox_working_px=box,
        bbox_source_px=box,
        block_num=block,
        par_num=1,
        line_num=line,
    )


def _magazine_page_words() -> list[WordBox]:
    """800 px wide magazine scan: kicker crossing the gutter, banner, two columns 20 px apart."""
    words = [
        # Kicker split by Tesseract into two blocks; the second crosses the gutter (x 373..578).
        _tess_word("Türkiye", 43, 18, block=2, line=1, width=150),
        _tess_word("Birliği", 213, 18, block=2, line=1, width=150),
        _tess_word("Başkanı", 373, 18, block=3, line=1, width=100),
        _tess_word("Doğan:", 478, 18, block=3, line=1, width=100),
        # Banner across both columns.
        _tess_word("RESMİ", 81, 72, block=4, line=1, width=300, height=40),
        _tess_word("TARİH", 400, 72, block=4, line=1, width=319, height=40),
    ]
    # Column bodies: left x 40..390, right x 411..766; same baselines on both sides.
    for i in range(12):
        top = 300 + i * 20
        words.append(_tess_word(f"Sol{i:02d}a", 40, top, block=7, line=i + 1, width=170))
        words.append(_tess_word(f"Sol{i:02d}b", 215, top, block=7, line=i + 1, width=175))
        words.append(_tess_word(f"Sag{i:02d}a", 411, top, block=10, line=i + 1, width=170))
        words.append(_tess_word(f"Sag{i:02d}b", 586, top, block=10, line=i + 1, width=180))
    return words


def test_narrow_gutter_columns_are_not_merged_when_tesseract_lines_are_known() -> None:
    """A 20 px gutter is far below the gap-split threshold; Tesseract's lines must keep columns apart."""
    res = analyze_layout_and_reading_order(
        _magazine_page_words(), page_width_px=795, page_height_px=744, layout_mode=LayoutMode.AUTO
    )

    assert res.column_count == 2
    lines = res.logical_text.splitlines()
    body = [ln for ln in lines if ln.startswith(("Sol", "Sag"))]
    assert body == [f"Sol{i:02d}a Sol{i:02d}b" for i in range(12)] + [
        f"Sag{i:02d}a Sag{i:02d}b" for i in range(12)
    ]
    # Kicker and banner come before the columns.
    assert lines.index("Türkiye Birliği") < lines.index("RESMİ TARİH") < lines.index(body[0])


def test_words_without_tesseract_structure_still_group_by_position() -> None:
    """Restored/synthetic words (line_num == 0) keep the geometric fallback."""
    words = [
        _make_word("Sol", 200, 350, width=250),
        _make_word("Satır", 480, 350, width=250),
    ]
    res = analyze_layout_and_reading_order(
        words, page_width_px=2000, page_height_px=2800, layout_mode=LayoutMode.AUTO
    )
    assert res.logical_text == "Sol Satır"
