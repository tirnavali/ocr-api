"""Multi-column archival newspaper and document layout & reading-order analysis (FR-11)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ocr_service.domain.models import LayoutMode, WordBox


@dataclass(slots=True)
class ReconstructedLine:
    """A geometric line of words on a page."""

    words: list[WordBox] = field(default_factory=list)
    left: float = 0.0
    top: float = 0.0
    right: float = 0.0
    bottom: float = 0.0
    column_index: int = 0
    is_spanning: bool = False

    @property
    def width(self) -> float:
        return max(0.0, self.right - self.left)

    @property
    def height(self) -> float:
        return max(1.0, self.bottom - self.top)

    @property
    def center_x(self) -> float:
        return (self.left + self.right) / 2.0

    @property
    def center_y(self) -> float:
        return (self.top + self.bottom) / 2.0

    @property
    def text(self) -> str:
        return " ".join(w.text for w in self.words)


@dataclass(slots=True)
class LayoutAnalysisResult:
    """Logical reading-order text and updated WordBox ordering."""

    ordered_words: list[WordBox]
    ordered_lines: list[ReconstructedLine]
    logical_text: str
    column_count: int


def _group_words_into_lines(words: list[WordBox], page_width_px: int) -> list[ReconstructedLine]:
    """Group WordBoxes into horizontal lines, splitting lines across wide column gutters."""
    if not words:
        return []

    # As a safety net against a Tesseract line that still runs across a gutter, any line
    # whose gap between consecutive words exceeds ~2.8x the median word height is split.
    heights = [w.bbox_source_px.height for w in words]
    median_h = float(np.median(heights)) if heights else 20.0
    max_intra_line_gap = max(median_h * 2.8, page_width_px * 0.045)

    # Tesseract's own page segmentation (block, par, line) already separates columns, even
    # when the gutter is much narrower than the gap threshold above (e.g. ~20 px on a
    # 800 px magazine scan). Grouping only by Y merged same-height lines of adjacent
    # columns into one "spanning" line and interleaved the columns, so Tesseract's lines
    # are used whenever the words carry them.
    if any(w.line_num > 0 for w in words):
        raw_clusters = _group_by_tesseract_line(words)
    else:
        raw_clusters = _group_by_vertical_position(words)

    lines: list[ReconstructedLine] = []
    for cluster in raw_clusters:
        cluster.sort(key=lambda w: w.bbox_source_px.left)
        current_segment: list[WordBox] = [cluster[0]]
        for w in cluster[1:]:
            prev = current_segment[-1]
            gap = w.bbox_source_px.left - prev.bbox_source_px.right
            if gap > max_intra_line_gap:
                lines.append(_build_line(current_segment))
                current_segment = [w]
            else:
                current_segment.append(w)
        if current_segment:
            lines.append(_build_line(current_segment))

    return lines


def _group_by_tesseract_line(words: list[WordBox]) -> list[list[WordBox]]:
    clusters: dict[tuple[int, int, int], list[WordBox]] = {}
    for w in words:
        clusters.setdefault((w.block_num, w.par_num, w.line_num), []).append(w)
    return list(clusters.values())


def _group_by_vertical_position(words: list[WordBox]) -> list[list[WordBox]]:
    """Fallback for words without Tesseract structure (e.g. restored or synthetic boxes)."""
    sorted_by_y = sorted(words, key=lambda w: (w.bbox_source_px.center_y, w.bbox_source_px.left))
    raw_clusters: list[list[WordBox]] = []
    for w in sorted_by_y:
        placed = False
        wy = w.bbox_source_px.center_y
        wh = w.bbox_source_px.height
        for cluster in reversed(raw_clusters[-15:]):
            cy = sum(item.bbox_source_px.center_y for item in cluster) / len(cluster)
            ch = sum(item.bbox_source_px.height for item in cluster) / len(cluster)
            if abs(wy - cy) <= 0.55 * max(wh, ch):
                cluster.append(w)
                placed = True
                break
        if not placed:
            raw_clusters.append([w])
    return raw_clusters


def _build_line(words: list[WordBox]) -> ReconstructedLine:
    left = min(w.bbox_source_px.left for w in words)
    top = min(w.bbox_source_px.top for w in words)
    right = max(w.bbox_source_px.right for w in words)
    bottom = max(w.bbox_source_px.bottom for w in words)
    return ReconstructedLine(words=words, left=left, top=top, right=right, bottom=bottom)


def _detect_column_gutters(
    lines: list[ReconstructedLine],
    page_width_px: int,
) -> list[tuple[float, float]]:
    """Detect column X-intervals [(col_left, col_right), ...] using 1D horizontal occupancy histogram."""
    if len(lines) < 2 or page_width_px <= 0:
        return [(0.0, float(page_width_px))]

    min_x = min(line.left for line in lines)
    max_x = max(line.right for line in lines)
    text_span = max_x - min_x
    if text_span <= page_width_px * 0.25:
        return [(0.0, float(page_width_px))]

    # Filter out wide spanning headlines (>58% of total text span) when finding vertical gutters
    column_candidates = [line for line in lines if line.width < 0.58 * text_span]
    if len(column_candidates) < 2:
        return [(0.0, float(page_width_px))]

    bins = 400
    hist = np.zeros(bins, dtype=np.int32)
    for line in column_candidates:
        b0 = max(0, min(bins - 1, int((line.left - min_x) / text_span * bins)))
        b1 = max(0, min(bins, int((line.right - min_x) / text_span * bins) + 1))
        hist[b0:b1] += 1

    # A column interval is a contiguous region where hist > 0 separated by empty gutter bins
    min_gutter_bins = max(3, int(bins * 0.015))
    min_col_bins = max(12, int(bins * 0.08))

    # A few short headline/byline lines that happen to cross the gutter (e.g. a kicker
    # line above the columns) must not close it, so bins covered by only a small share
    # of the lines count as empty. For short pages the floor stays 0 (any coverage counts).
    noise_floor = int(0.08 * int(hist.max()))
    occupied = hist > noise_floor
    intervals: list[tuple[int, int]] = []
    start: int | None = None
    for idx, val in enumerate(occupied):
        if val and start is None:
            start = idx
        elif not val and start is not None:
            intervals.append((start, idx))
            start = None
    if start is not None:
        intervals.append((start, bins))

    if not intervals:
        return [(0.0, float(page_width_px))]

    # Merge intervals separated by narrow gaps (< min_gutter_bins)
    merged: list[tuple[int, int]] = [intervals[0]]
    for cur_start, cur_end in intervals[1:]:
        prev_start, prev_end = merged[-1]
        if (cur_start - prev_end) < min_gutter_bins:
            merged[-1] = (prev_start, cur_end)
        else:
            merged.append((cur_start, cur_end))

    valid_cols = [(s, e) for (s, e) in merged if (e - s) >= min_col_bins]
    if len(valid_cols) <= 1:
        return [(0.0, float(page_width_px))]

    col_bounds: list[tuple[float, float]] = []
    for s, e in valid_cols:
        col_l = min_x + (s / bins) * text_span
        col_r = min_x + (e / bins) * text_span
        col_bounds.append((col_l, col_r))

    return col_bounds


def analyze_layout_and_reading_order(
    words: list[WordBox],
    *,
    page_width_px: int,
    page_height_px: int,
    layout_mode: LayoutMode = LayoutMode.AUTO,
) -> LayoutAnalysisResult:
    """Order recognized words and lines into logical archival/newspaper reading order (FR-11).

    Note: Word geometric coordinates (bbox_source_px, bbox_pdf_pt) are NEVER altered by
    reading-order sorting, preserving 100% geometric alignment in the PDF/A text layer.
    """
    if not words:
        return LayoutAnalysisResult(
            ordered_words=[],
            ordered_lines=[],
            logical_text="",
            column_count=1,
        )

    lines = _group_words_into_lines(words, page_width_px=page_width_px)
    if layout_mode == LayoutMode.SINGLE_COLUMN:
        lines.sort(key=lambda ln: (ln.top, ln.left))
        ordered_words: list[WordBox] = []
        for ln in lines:
            ln.column_index = 0
            for w in ln.words:
                w.column_index = 0
                w.reading_order_index = len(ordered_words)
                ordered_words.append(w)
        return LayoutAnalysisResult(
            ordered_words=ordered_words,
            ordered_lines=lines,
            logical_text="\n".join(ln.text for ln in lines if ln.text),
            column_count=1,
        )

    columns = _detect_column_gutters(lines, page_width_px=page_width_px)
    num_cols = len(columns)
    _assign_columns(lines, columns)

    if any(w.line_num > 0 for w in words) and not _blocks_mix_columns(lines):
        # Tesseract's block order follows the page's articles (headline, its columns, the
        # next article, captions) far better than one page-wide column grid on busy
        # newspaper pages. It is only trusted when no block mixes lines of different
        # columns; otherwise Tesseract cut across the columns and the grid order is used.
        groups = _order_by_tesseract_blocks(lines)
    elif num_cols <= 1:
        groups = [sorted(lines, key=lambda ln: (ln.top, ln.left))]
    else:
        groups = [_order_lines_by_columns(lines, columns)]

    ordered_lines: list[ReconstructedLine] = []
    ordered_words: list[WordBox] = []
    text_blocks: list[str] = []
    for group in groups:
        prev_col: int | None = None
        prev_spanning = False
        if text_blocks:
            text_blocks.append("")  # blank line between Tesseract blocks
        for ln in group:
            if prev_col is not None and (ln.column_index != prev_col or ln.is_spanning != prev_spanning):
                text_blocks.append("")  # blank line separating columns or headline blocks
            text_blocks.append(ln.text)
            prev_col = ln.column_index
            prev_spanning = ln.is_spanning
            ordered_lines.append(ln)
            for w in ln.words:
                w.column_index = ln.column_index
                w.reading_order_index = len(ordered_words)
                ordered_words.append(w)

    return LayoutAnalysisResult(
        ordered_words=ordered_words,
        ordered_lines=ordered_lines,
        logical_text="\n".join(text_blocks).strip(),
        column_count=num_cols,
    )


def _assign_columns(lines: list[ReconstructedLine], columns: list[tuple[float, float]]) -> None:
    """Set column_index / is_spanning of every line against the page's column grid."""
    num_cols = len(columns)
    for ln in lines:
        if num_cols <= 1:
            ln.column_index = 0
            ln.is_spanning = False
            continue
        overlapped_cols: list[int] = []
        for col_idx, (col_l, col_r) in enumerate(columns):
            overlap = min(ln.right, col_r) - max(ln.left, col_l)
            if overlap > 0.25 * min(ln.width, col_r - col_l):
                overlapped_cols.append(col_idx)

        if len(overlapped_cols) >= 2:
            ln.is_spanning = True
            ln.column_index = overlapped_cols[0]
        elif len(overlapped_cols) == 1:
            ln.is_spanning = False
            ln.column_index = overlapped_cols[0]
        else:
            # Assign to nearest column center
            ln.is_spanning = False
            ln.column_index = min(
                range(num_cols),
                key=lambda ci: abs(ln.center_x - ((columns[ci][0] + columns[ci][1]) / 2.0)),
            )


def _order_lines_by_columns(
    lines: list[ReconstructedLine], columns: list[tuple[float, float]]
) -> list[ReconstructedLine]:
    """Spanning headlines first, then each column top-to-bottom, band by band."""
    num_cols = len(columns)
    lines_by_top = sorted(lines, key=lambda ln: (ln.top, ln.left))
    ordered_lines: list[ReconstructedLine] = []
    current_band: list[ReconstructedLine] = []

    def _flush_band(band: list[ReconstructedLine]) -> list[ReconstructedLine]:
        result: list[ReconstructedLine] = []
        for col_idx in range(num_cols):
            col_lines = [b for b in band if b.column_index == col_idx]
            col_lines.sort(key=lambda b: (b.top, b.left))
            result.extend(col_lines)
        return result

    # Partition vertically at spanning headlines so headlines precede the columns below
    for ln in lines_by_top:
        if ln.is_spanning:
            ordered_lines.extend(_flush_band(current_band))
            current_band = []
            ordered_lines.append(ln)
        else:
            current_band.append(ln)
    ordered_lines.extend(_flush_band(current_band))
    return ordered_lines


def _blocks_mix_columns(lines: list[ReconstructedLine]) -> bool:
    """True when a Tesseract block holds non-spanning lines of more than one column."""
    cols_by_block: dict[int, set[int]] = {}
    for ln in lines:
        if not ln.is_spanning:
            cols_by_block.setdefault(ln.words[0].block_num, set()).add(ln.column_index)
    return any(len(cols) > 1 for cols in cols_by_block.values())


def _order_by_tesseract_blocks(lines: list[ReconstructedLine]) -> list[list[ReconstructedLine]]:
    blocks: dict[int, list[ReconstructedLine]] = {}
    for ln in lines:
        blocks.setdefault(ln.words[0].block_num, []).append(ln)
    return [
        sorted(blocks[b], key=lambda ln: (ln.words[0].par_num, ln.words[0].line_num, ln.left))
        for b in sorted(blocks)
    ]
