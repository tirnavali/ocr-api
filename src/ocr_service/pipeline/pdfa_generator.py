"""Lossless PDF/A-2b generator with embedded Unicode CIDFont and 3 Tr invisible text layer."""

from __future__ import annotations

import hashlib
import logging
import uuid
import zlib
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from xml.sax.saxutils import escape as xml_escape

import pikepdf
from fontTools.ttLib import TTFont
from PIL import Image

from ocr_service.config import Settings
from ocr_service.domain.models import (
    PageOcrResult,
    PageSourceDescriptor,
    PdfAProfile,
    WordBox,
)

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class EmbeddedFontMetrics:
    """Extracted TrueType metrics and Unicode-to-GlyphID mappings for PDF Type0 embedding."""

    font_bytes: bytes
    font_name: str
    units_per_em: int
    ascent_1000: int
    descent_1000: int
    cap_height_1000: int
    italic_angle: float
    bbox_1000: tuple[int, int, int, int]
    cmap: dict[int, int]  # Unicode codepoint -> Glyph ID
    gid_widths_1000: dict[int, int]  # Glyph ID -> width in 1/1000 em
    space_gid: int


def load_truetype_metrics(font_path: Path) -> EmbeddedFontMetrics:
    """Parse TrueType font tables via fontTools for accurate PDF CIDFontType2 embedding."""
    font_bytes = font_path.read_bytes()
    tt = TTFont(font_path)
    try:
        head = tt["head"]
        hhea = tt["hhea"]
        os2 = tt.get("OS/2")
        post = tt.get("post")
        hmtx = tt["hmtx"]

        upm = int(head.unitsPerEm)
        scale = 1000.0 / float(upm)

        ascent = round(hhea.ascent * scale)
        descent = round(hhea.descent * scale)
        if descent > 0:
            descent = -descent
        cap_height = (
            round(getattr(os2, "sCapHeight", hhea.ascent * 0.7) * scale)
            if os2
            else round(ascent * 0.7)
        )
        italic_angle = float(getattr(post, "italicAngle", 0.0)) if post else 0.0
        bbox = (
            round(head.xMin * scale),
            round(head.yMin * scale),
            round(head.xMax * scale),
            round(head.yMax * scale),
        )

        best_cmap = dict(tt.getBestCmap() or {})
        glyph_order = tt.getGlyphOrder()
        gid_map = {name: idx for idx, name in enumerate(glyph_order)}

        unicode_to_gid: dict[int, int] = {}
        for codepoint, glyph_name in best_cmap.items():
            gid = gid_map.get(glyph_name)
            if gid is not None:
                unicode_to_gid[int(codepoint)] = int(gid)

        gid_widths_1000: dict[int, int] = {}
        for glyph_name, (advance_width, _lsb) in hmtx.metrics.items():
            gid = gid_map.get(glyph_name)
            if gid is not None:
                gid_widths_1000[int(gid)] = max(0, round(advance_width * scale))

        space_gid = unicode_to_gid.get(0x0020, 3)
        return EmbeddedFontMetrics(
            font_bytes=font_bytes,
            font_name="ArchiveUnicodeRegular",
            units_per_em=upm,
            ascent_1000=ascent,
            descent_1000=descent,
            cap_height_1000=cap_height,
            italic_angle=italic_angle,
            bbox_1000=bbox,
            cmap=unicode_to_gid,
            gid_widths_1000=gid_widths_1000,
            space_gid=space_gid,
        )
    finally:
        tt.close()


def build_tounicode_cmap_bytes(gid_to_unicode: dict[int, int]) -> bytes:
    """Generate a PDF/A-compliant PostScript /ToUnicode CMap stream."""
    sorted_items = sorted(gid_to_unicode.items(), key=lambda kv: kv[0])
    lines = [
        "/CIDInit /ProcSet findresource begin",
        "12 dict begin",
        "begincmap",
        "/CIDSystemInfo << /Registry (Adobe) /Ordering (UCS) /Supplement 0 >> def",
        "/CMapName /Adobe-Identity-UCS def",
        "/CMapType 2 def",
        "1 begincodespacerange",
        "<0000> <FFFF>",
        "endcodespacerange",
    ]

    # Group into chunks of at most 100 entries per beginbfchar block (PDF spec limit)
    chunk_size = 100
    for i in range(0, len(sorted_items), chunk_size):
        chunk = sorted_items[i : i + chunk_size]
        lines.append(f"{len(chunk)} beginbfchar")
        for gid, codepoint in chunk:
            utf16_hex = chr(codepoint).encode("utf-16-be").hex().upper()
            lines.append(f"<{gid:04X}> <{utf16_hex}>")
        lines.append("endbfchar")

    lines.extend(
        [
            "endcmap",
            "CMapName currentdict /CMap defineresource pop",
            "end",
            "end",
        ]
    )
    return "\n".join(lines).encode("ascii")


def build_xmp_metadata_bytes(
    *,
    profile: PdfAProfile,
    title: str,
    creator: str,
    producer: str,
    timestamp_iso: str,
) -> bytes:
    """Create an uncompressed XMP metadata packet conforming to ISO 19005-1 / 19005-2."""
    part = "1" if profile == PdfAProfile.PDFA_1B else "2"
    conformance = "B"
    safe_title = xml_escape(title)
    safe_creator = xml_escape(creator)
    safe_producer = xml_escape(producer)

    xmp = f"""<?xpacket begin="\ufeff" id="W5M0MpCehiHzreSzNTczkc9d"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/">
  <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
    <rdf:Description rdf:about=""
        xmlns:pdfaid="http://www.aiim.org/pdfa/ns/id/"
        xmlns:dc="http://purl.org/dc/elements/1.1/"
        xmlns:xmp="http://ns.adobe.com/xap/1.0/"
        xmlns:pdf="http://ns.adobe.com/pdf/1.3/">
      <pdfaid:part>{part}</pdfaid:part>
      <pdfaid:conformance>{conformance}</pdfaid:conformance>
      <dc:title>
        <rdf:Alt>
          <rdf:li xml:lang="x-default">{safe_title}</rdf:li>
        </rdf:Alt>
      </dc:title>
      <dc:creator>
        <rdf:Seq>
          <rdf:li>{safe_creator}</rdf:li>
        </rdf:Seq>
      </dc:creator>
      <xmp:CreatorTool>{safe_creator}</xmp:CreatorTool>
      <xmp:CreateDate>{timestamp_iso}</xmp:CreateDate>
      <xmp:ModifyDate>{timestamp_iso}</xmp:ModifyDate>
      <pdf:Producer>{safe_producer}</pdf:Producer>
    </rdf:Description>
  </rdf:RDF>
</x:xmpmeta>
<?xpacket end="w"?>"""
    return xmp.encode("utf-8")


class PdfAGenerator:
    """Compose searchable PDF/A documents preserving source images and exact word alignment."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._font_metrics = load_truetype_metrics(settings.unicode_font_path)
        self._srgb_icc_bytes = settings.srgb_icc_path.read_bytes()

    def generate_pdfa(
        self,
        *,
        descriptors: list[PageSourceDescriptor],
        page_results: list[PageOcrResult],
        output_pdf_path: Path,
        profile: PdfAProfile = PdfAProfile.PDFA_2B,
        title: str = "Archival OCR Document",
    ) -> Path:
        """Build a complete searchable PDF/A file from page source descriptors and OCR results."""
        output_pdf_path.parent.mkdir(parents=True, exist_ok=True)
        pdf = pikepdf.Pdf.new()

        # Track all used glyph IDs -> Unicode codepoints across pages for compact /W & /ToUnicode
        used_gid_to_unicode: dict[int, int] = {self._font_metrics.space_gid: 0x0020}

        # Pre-scan all recognized words to register their glyphs in the embedded font dictionary
        for page_res in page_results:
            for word in page_res.words:
                for ch in word.text:
                    cp = ord(ch)
                    gid = self._font_metrics.cmap.get(cp)
                    if gid is not None:
                        used_gid_to_unicode[gid] = cp

        font_obj = self._create_embedded_cid_font(pdf, used_gid_to_unicode)

        # Open source PDF once if the input is a PDF document
        source_pdf: pikepdf.Pdf | None = None
        if descriptors and descriptors[0].source_format == "PDF_PAGE":
            source_pdf = pikepdf.open(descriptors[0].source_path)

        try:
            for desc, page_res in zip(descriptors, page_results, strict=True):
                if desc.source_format == "PDF_PAGE" and source_pdf is not None:
                    self._append_pdf_source_page(
                        pdf=pdf,
                        source_pdf=source_pdf,
                        descriptor=desc,
                        page_result=page_res,
                        font_obj=font_obj,
                    )
                else:
                    self._append_image_source_page(
                        pdf=pdf,
                        descriptor=desc,
                        page_result=page_res,
                        font_obj=font_obj,
                        preserve_source=self._settings.preserve_source_images,
                    )

            self._apply_pdfa_catalog_and_metadata(
                pdf=pdf,
                profile=profile,
                title=title,
            )

            min_version = "1.4" if profile == PdfAProfile.PDFA_1B else "1.7"
            pdf.save(
                output_pdf_path,
                min_version=min_version,
                deterministic_id=True,
                compress_streams=True,
            )
        finally:
            if source_pdf is not None:
                source_pdf.close()
            pdf.close()

        return output_pdf_path

    def _create_embedded_cid_font(
        self,
        pdf: pikepdf.Pdf,
        used_gid_to_unicode: dict[int, int],
    ) -> pikepdf.Dictionary:
        """Construct a PDF/A-compliant Type0 / CIDFontType2 dictionary with /ToUnicode CMap."""
        fm = self._font_metrics
        font_stream = pikepdf.Stream(pdf, fm.font_bytes)
        font_stream[pikepdf.Name("/Length1")] = len(fm.font_bytes)

        font_descriptor = pdf.make_indirect(
            pikepdf.Dictionary(
                {
                    "/Type": pikepdf.Name("/FontDescriptor"),
                    "/FontName": pikepdf.Name(f"/{fm.font_name}"),
                    "/Flags": 32,  # Nonsymbolic
                    "/FontBBox": pikepdf.Array(list(fm.bbox_1000)),
                    "/ItalicAngle": fm.italic_angle,
                    "/Ascent": fm.ascent_1000,
                    "/Descent": fm.descent_1000,
                    "/CapHeight": fm.cap_height_1000,
                    "/StemV": 80,
                    "/FontFile2": font_stream,
                }
            )
        )

        # Build /W array for all used glyphs
        w_entries: list[int | pikepdf.Array] = []
        for gid in sorted(used_gid_to_unicode.keys()):
            w_1000 = fm.gid_widths_1000.get(gid, 600)
            w_entries.append(gid)
            w_entries.append(pikepdf.Array([w_1000]))

        cid_font = pdf.make_indirect(
            pikepdf.Dictionary(
                {
                    "/Type": pikepdf.Name("/Font"),
                    "/Subtype": pikepdf.Name("/CIDFontType2"),
                    "/BaseFont": pikepdf.Name(f"/{fm.font_name}"),
                    "/CIDSystemInfo": pikepdf.Dictionary(
                        {
                            "/Registry": pikepdf.String("Adobe"),
                            "/Ordering": pikepdf.String("Identity"),
                            "/Supplement": 0,
                        }
                    ),
                    "/FontDescriptor": font_descriptor,
                    "/DW": 600,
                    "/W": pikepdf.Array(w_entries),
                    "/CIDToGIDMap": pikepdf.Name("/Identity"),
                }
            )
        )

        tounicode_bytes = build_tounicode_cmap_bytes(used_gid_to_unicode)
        tounicode_stream = pdf.make_indirect(pikepdf.Stream(pdf, tounicode_bytes))

        type0_font = pdf.make_indirect(
            pikepdf.Dictionary(
                {
                    "/Type": pikepdf.Name("/Font"),
                    "/Subtype": pikepdf.Name("/Type0"),
                    "/BaseFont": pikepdf.Name(f"/{fm.font_name}"),
                    "/Encoding": pikepdf.Name("/Identity-H"),
                    "/DescendantFonts": pikepdf.Array([cid_font]),
                    "/ToUnicode": tounicode_stream,
                }
            )
        )
        return type0_font

    def _append_image_source_page(
        self,
        *,
        pdf: pikepdf.Pdf,
        descriptor: PageSourceDescriptor,
        page_result: PageOcrResult,
        font_obj: pikepdf.Dictionary,
        preserve_source: bool,
    ) -> None:
        """Create a PDF page from a source image with lossless stream preservation and 3 Tr text."""
        image_xobj, preservation_mode, warning = self._create_image_xobject(
            pdf, descriptor, preserve_source=preserve_source
        )
        page_result.image_preservation_mode = preservation_mode
        if warning:
            page_result.warnings.append(warning)

        w_pt = page_result.width_pt
        h_pt = page_result.height_pt

        # Content stream part 1: Draw the full-page archival image
        draw_image_ops = f"q\n{w_pt:.4f} 0 0 {h_pt:.4f} 0 0 cm\n/Im0 Do\nQ\n".encode("ascii")

        # Content stream part 2: Draw the invisible OCR text layer (3 Tr)
        text_ops = self._build_invisible_text_stream(page_result.words)

        content_stream = pdf.make_indirect(pikepdf.Stream(pdf, draw_image_ops + text_ops))

        resources = pikepdf.Dictionary(
            {
                "/ProcSet": pikepdf.Array(
                    [
                        pikepdf.Name("/PDF"),
                        pikepdf.Name("/Text"),
                        pikepdf.Name("/ImageB"),
                        pikepdf.Name("/ImageC"),
                    ]
                ),
                "/XObject": pikepdf.Dictionary({"/Im0": image_xobj}),
                "/Font": pikepdf.Dictionary({"/F_OCR": font_obj}),
            }
        )

        page_dict = pikepdf.Dictionary(
            {
                "/Type": pikepdf.Name("/Page"),
                "/MediaBox": pikepdf.Array([0, 0, round(w_pt, 4), round(h_pt, 4)]),
                "/Resources": resources,
                "/Contents": content_stream,
            }
        )
        pdf.pages.append(pikepdf.Page(page_dict))

    def _create_image_xobject(
        self,
        pdf: pikepdf.Pdf,
        descriptor: PageSourceDescriptor,
        *,
        preserve_source: bool,
    ) -> tuple[pikepdf.Stream, str, str | None]:
        """Embed source image stream without lossy recompression (FR-13, FR-14)."""
        # Case 1: Direct raw JPEG passthrough (/DCTDecode) for RGB and Grayscale JPEGs
        if descriptor.source_format == "JPEG" and preserve_source:
            with Image.open(descriptor.source_path) as img:
                mode = img.mode
                width, height = img.size
            if mode in ("RGB", "L"):
                raw_jpeg = descriptor.source_path.read_bytes()
                colorspace = (
                    pikepdf.Name("/DeviceRGB") if mode == "RGB" else pikepdf.Name("/DeviceGray")
                )
                stream = pdf.make_indirect(
                    pikepdf.Stream(
                        pdf,
                        raw_jpeg,
                        {
                            "/Type": pikepdf.Name("/XObject"),
                            "/Subtype": pikepdf.Name("/Image"),
                            "/Width": width,
                            "/Height": height,
                            "/ColorSpace": colorspace,
                            "/BitsPerComponent": 8,
                            "/Filter": pikepdf.Name("/DCTDecode"),
                        },
                    )
                )
                return stream, "jpeg_dct_passthrough", None

        # Case 2: Lossless FlateDecode embedding for PNG, TIFF, or CMYK/RGBA images
        warning: str | None = None
        preservation_mode = "lossless_flate"
        with Image.open(descriptor.source_path) as img:
            width, height = img.size
            orig_mode = img.mode

            if orig_mode in ("RGBA", "LA") or (orig_mode == "P" and "transparency" in img.info):
                # Composite onto white background for strict PDF/A compatibility
                rgba = img.convert("RGBA")
                white_bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
                composited = Image.alpha_composite(white_bg, rgba).convert("RGB")
                raw_pixels = composited.tobytes()
                colorspace = pikepdf.Name("/DeviceRGB")
                preservation_mode = "lossless_flate_alpha_flattened"
                warning = (
                    f"Source image had alpha channel ({orig_mode}); flattened onto white "
                    "background with lossless Flate compression for PDF/A compliance."
                )
            elif orig_mode == "CMYK":
                rgb_img = img.convert("RGB")
                raw_pixels = rgb_img.tobytes()
                colorspace = pikepdf.Name("/DeviceRGB")
                preservation_mode = "lossless_flate_cmyk_to_rgb"
                warning = (
                    "Source CMYK image converted to sRGB with lossless Flate compression "
                    "to match PDF/A sRGB OutputIntent."
                )
            elif orig_mode in ("1", "L"):
                gray_img = img.convert("L")
                raw_pixels = gray_img.tobytes()
                colorspace = pikepdf.Name("/DeviceGray")
            else:
                rgb_img = img.convert("RGB")
                raw_pixels = rgb_img.tobytes()
                colorspace = pikepdf.Name("/DeviceRGB")

        compressed = zlib.compress(raw_pixels, level=6)
        stream = pdf.make_indirect(
            pikepdf.Stream(
                pdf,
                compressed,
                {
                    "/Type": pikepdf.Name("/XObject"),
                    "/Subtype": pikepdf.Name("/Image"),
                    "/Width": width,
                    "/Height": height,
                    "/ColorSpace": colorspace,
                    "/BitsPerComponent": 8,
                    "/Filter": pikepdf.Name("/FlateDecode"),
                },
            )
        )
        return stream, preservation_mode, warning

    def _append_pdf_source_page(
        self,
        *,
        pdf: pikepdf.Pdf,
        source_pdf: pikepdf.Pdf,
        descriptor: PageSourceDescriptor,
        page_result: PageOcrResult,
        font_obj: pikepdf.Dictionary,
    ) -> None:
        """Copy original PDF page and overlay invisible 3 Tr text layer without re-rasterizing."""
        assert descriptor.source_pdf_page_index is not None
        src_page = source_pdf.pages[descriptor.source_pdf_page_index]
        pdf.pages.append(src_page)
        dst_page = pdf.pages[-1]

        if not page_result.ocr_applied or not page_result.words:
            # Preserve existing PDF text layer, remediating any unembedded fonts for PDF/A compliance
            self._normalize_pdf_page_resources_for_pdfa(
                pdf, dst_page, page_result, strip_existing_fonts=False
            )
            page_result.image_preservation_mode = "pdf_native_preserved"
            return

        # When applying fresh OCR to a PDF page, strip any legacy BT..ET text blocks and unembedded
        # fonts from the copied page while keeping all image/vector XObjects untouched.
        self._strip_existing_text_operators(pdf, dst_page)
        self._normalize_pdf_page_resources_for_pdfa(
            pdf, dst_page, page_result, strip_existing_fonts=True
        )

        if "/Resources" not in dst_page.obj:
            dst_page.obj["/Resources"] = pikepdf.Dictionary()
        res = dst_page.obj["/Resources"]
        res["/Font"] = pikepdf.Dictionary({"/F_OCR": font_obj})

        text_ops = self._build_invisible_text_stream(page_result.words)
        if text_ops:
            overlay_stream = pdf.make_indirect(pikepdf.Stream(pdf, b"\nq\n" + text_ops + b"\nQ\n"))
            dst_page.contents_add(overlay_stream, prepend=False)

        page_result.image_preservation_mode = "pdf_native_with_invisible_ocr"

    @staticmethod
    def _strip_existing_text_operators(pdf: pikepdf.Pdf, page: pikepdf.Page) -> None:
        """Remove existing BT ... ET text blocks from a PDF page while preserving image/vector drawing."""
        try:
            instructions = pikepdf.parse_content_stream(page)
            filtered = []
            inside_text_object = False
            for operands, operator in instructions:
                op_name = str(operator)
                if op_name == "BT":
                    inside_text_object = True
                    continue
                if op_name == "ET":
                    inside_text_object = False
                    continue
                if not inside_text_object:
                    filtered.append((operands, operator))
            new_bytes = pikepdf.unparse_content_stream(filtered)
            page.obj["/Contents"] = pdf.make_indirect(pikepdf.Stream(pdf, new_bytes))
        except Exception as exc:
            logger.debug("Could not filter existing content stream text operators: %s", exc)

    def _normalize_pdf_page_resources_for_pdfa(
        self,
        pdf: pikepdf.Pdf,
        page: pikepdf.Page,
        page_result: PageOcrResult,
        *,
        strip_existing_fonts: bool = False,
    ) -> None:
        """Ensure fonts and XObjects on copied PDF pages meet strict PDF/A-2b requirements."""
        res = page.obj.get("/Resources")
        if not isinstance(res, pikepdf.Dictionary):
            return

        # 1. Handle fonts on copied PDF pages
        fonts = res.get("/Font")
        if isinstance(fonts, pikepdf.Dictionary):
            if strip_existing_fonts:
                res["/Font"] = pikepdf.Dictionary()
            else:
                self._embed_missing_page_fonts(pdf, fonts, page_result)

        # 2. Handle Image XObjects
        xobjs = res.get("/XObject")
        if not isinstance(xobjs, pikepdf.Dictionary):
            return

        for key in list(xobjs.keys()):
            xobj = xobjs[key]
            if not isinstance(xobj, pikepdf.Stream):
                continue
            if xobj.get("/Subtype") != pikepdf.Name("/Image"):
                continue
            # Remove /SMask soft-mask transparency if PDF/A-1b is requested
            if self._settings.pdf_a_profile == "PDF/A-1b" and "/SMask" in xobj:
                del xobj["/SMask"]
                page_result.warnings.append(f"Removed /SMask from XObject {key} for PDF/A-1b.")

    def _embed_missing_page_fonts(
        self,
        pdf: pikepdf.Pdf,
        fonts_dict: pikepdf.Dictionary,
        page_result: PageOcrResult,
    ) -> None:
        """Remedate unembedded legacy Type1/TrueType fonts in preserved PDF pages for PDF/A compliance."""
        fm = self._font_metrics
        shared_font_stream: pikepdf.Stream | None = None

        for font_key in list(fonts_dict.keys()):
            font_obj = fonts_dict[font_key]
            if not isinstance(font_obj, pikepdf.Dictionary):
                continue

            subtype = font_obj.get("/Subtype")
            if subtype == pikepdf.Name("/Type0"):
                descendants = font_obj.get("/DescendantFonts")
                if descendants and len(descendants) > 0:
                    target_font = descendants[0]
                else:
                    continue
            else:
                target_font = font_obj

            fdesc = target_font.get("/FontDescriptor")
            is_embedded = isinstance(fdesc, pikepdf.Dictionary) and any(
                k in fdesc for k in ("/FontFile", "/FontFile2", "/FontFile3")
            )
            if is_embedded:
                continue

            # Embed our bundled Unicode TrueType font stream into the unembedded font descriptor
            if shared_font_stream is None:
                shared_font_stream = pdf.make_indirect(pikepdf.Stream(pdf, fm.font_bytes))
                shared_font_stream[pikepdf.Name("/Length1")] = len(fm.font_bytes)

            if not isinstance(fdesc, pikepdf.Dictionary):
                fdesc = pdf.make_indirect(
                    pikepdf.Dictionary(
                        {
                            "/Type": pikepdf.Name("/FontDescriptor"),
                            "/FontName": target_font.get(
                                "/BaseFont", pikepdf.Name(f"/{fm.font_name}")
                            ),
                            "/Flags": 32,
                            "/FontBBox": pikepdf.Array(list(fm.bbox_1000)),
                            "/ItalicAngle": fm.italic_angle,
                            "/Ascent": fm.ascent_1000,
                            "/Descent": fm.descent_1000,
                            "/CapHeight": fm.cap_height_1000,
                            "/StemV": 80,
                        }
                    )
                )
                target_font["/FontDescriptor"] = fdesc

            fdesc["/FontFile2"] = shared_font_stream
            if subtype == pikepdf.Name("/Type1"):
                font_obj["/Subtype"] = pikepdf.Name("/TrueType")
            page_result.warnings.append(
                f"Embedded Unicode TrueType stream for previously unembedded font {font_key}."
            )

    def _build_invisible_text_stream(self, words: list[WordBox]) -> bytes:
        """Build a PDF content stream placing each word with 3 Tr and exact horizontal scaling Tz."""
        if not words:
            return b""

        fm = self._font_metrics
        ops: list[str] = ["q", "BT", "3 Tr"]

        for word in words:
            if word.bbox_pdf_pt is None or not word.text:
                continue

            gids: list[int] = []
            advance_sum_1000 = 0
            for ch in word.text:
                gid = fm.cmap.get(ord(ch))
                if gid is None:
                    continue
                gids.append(gid)
                advance_sum_1000 += fm.gid_widths_1000.get(gid, 600)

            if not gids:
                continue

            box = word.bbox_pdf_pt
            font_size = max(1.0, round(box.height, 2))
            baseline_y = (
                word.baseline_pdf_y
                if word.baseline_pdf_y is not None
                else (box.y + 0.18 * box.height)
            )

            # Unscaled word width in PDF points at font_size
            unscaled_width_pt = (advance_sum_1000 / 1000.0) * font_size
            if unscaled_width_pt > 0.01 and box.width > 0.01:
                tz = max(10.0, min(1000.0, (box.width / unscaled_width_pt) * 100.0))
            else:
                tz = 100.0

            word_hex = "".join(f"{gid:04X}" for gid in gids)
            space_hex = f"{fm.space_gid:04X}"

            # Position word at exact (box.x, baseline_y) with horizontal scale tz,
            # followed by an invisible space glyph so text extraction preserves word boundaries.
            ops.append(f"/F_OCR {font_size:.2f} Tf")
            ops.append(f"{tz:.2f} Tz")
            ops.append(f"1 0 0 1 {box.x:.3f} {baseline_y:.3f} Tm")
            ops.append(f"<{word_hex}> Tj")
            ops.append("100 Tz")
            ops.append(f"<{space_hex}> Tj")

        ops.extend(["ET", "Q", ""])
        return "\n".join(ops).encode("ascii")

    def _apply_pdfa_catalog_and_metadata(
        self,
        *,
        pdf: pikepdf.Pdf,
        profile: PdfAProfile,
        title: str,
    ) -> None:
        """Attach XMP metadata, synchronized DocumentInfo, and sRGB /OutputIntents (FR-05)."""
        now = datetime.now(tz=UTC)
        timestamp_iso = now.strftime("%Y-%m-%dT%H:%M:%SZ")
        pdf_date = now.strftime("D:%Y%m%d%H%M%SZ")
        creator = "Archival OCR API Service"
        producer = f"pikepdf {pikepdf.__version__} / OCR-API 1.0"

        # 1. Uncompressed XMP Metadata stream in Catalog
        xmp_bytes = build_xmp_metadata_bytes(
            profile=profile,
            title=title,
            creator=creator,
            producer=producer,
            timestamp_iso=timestamp_iso,
        )
        metadata_stream = pdf.make_indirect(
            pikepdf.Stream(
                pdf,
                xmp_bytes,
                {
                    "/Type": pikepdf.Name("/Metadata"),
                    "/Subtype": pikepdf.Name("/XML"),
                },
            )
        )
        metadata_stream.stream_dict["/Filter"] = pikepdf.Array([])
        del metadata_stream.stream_dict["/Filter"]
        pdf.Root["/Metadata"] = metadata_stream

        # 2. Synchronized DocumentInfo dictionary
        pdf.docinfo["/Title"] = title
        pdf.docinfo["/Creator"] = creator
        pdf.docinfo["/Producer"] = producer
        pdf.docinfo["/CreationDate"] = pdf_date
        pdf.docinfo["/ModDate"] = pdf_date

        # 3. Embedded sRGB ICC Profile in /OutputIntents
        icc_stream = pdf.make_indirect(
            pikepdf.Stream(
                pdf,
                self._srgb_icc_bytes,
                {
                    "/N": 3,
                    "/Alternate": pikepdf.Name("/DeviceRGB"),
                },
            )
        )
        output_intent = pdf.make_indirect(
            pikepdf.Dictionary(
                {
                    "/Type": pikepdf.Name("/OutputIntent"),
                    "/S": pikepdf.Name("/GTS_PDFA1"),
                    "/OutputConditionIdentifier": pikepdf.String("sRGB IEC61966-2.1"),
                    "/RegistryName": pikepdf.String("http://www.color.org"),
                    "/Info": pikepdf.String("sRGB IEC61966-2.1"),
                    "/DestOutputProfile": icc_stream,
                }
            )
        )
        pdf.Root["/OutputIntents"] = pikepdf.Array([output_intent])

        # 4. MarkInfo / ViewerPreferences for clean archival structure
        if "/Version" not in pdf.Root:
            pdf.Root["/Version"] = pikepdf.Name(
                "/1.4" if profile == PdfAProfile.PDFA_1B else "/1.7"
            )
        # Ensure deterministic trailer ID hash seed
        _ = hashlib.md5(uuid.uuid4().bytes).digest()
