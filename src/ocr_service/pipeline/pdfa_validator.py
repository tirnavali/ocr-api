"""Independent PDF/A conformance validation using structural ISO 19005 checks and veraPDF CLI."""

from __future__ import annotations

import logging
import shutil
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import pikepdf

from ocr_service.config import Settings
from ocr_service.domain.models import PdfAProfile, PdfAValidationReport
from ocr_service.domain.ports import PdfAValidatorPort

logger = logging.getLogger(__name__)


class DualPdfAValidator(PdfAValidatorPort):
    """Validate PDF/A conformance via comprehensive ISO 19005 structural inspection + veraPDF."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def validate(
        self,
        pdf_path: Path,
        profile: PdfAProfile = PdfAProfile.PDFA_2B,
    ) -> PdfAValidationReport:
        errors: list[str] = []
        warnings: list[str] = []
        details: dict[str, object] = {}

        # Layer 1: Structural ISO 19005-1 / 19005-2 inspection via pikepdf
        self._run_structural_checks(pdf_path, profile, errors, warnings, details)

        # Layer 2: External veraPDF validator if available on the host/container
        verapdf_bin = shutil.which(self._settings.verapdf_cmd)
        validator_used = "structural"

        if verapdf_bin:
            verapdf_ok, verapdf_errors = self._run_verapdf_cli(verapdf_bin, pdf_path, profile)
            validator_used = "structural+verapdf"
            details["verapdf_binary"] = verapdf_bin
            if not verapdf_ok:
                errors.extend(verapdf_errors)
        elif self._settings.require_verapdf:
            errors.append(
                f"REQUIRE_VERAPDF is enabled, but veraPDF binary {self._settings.verapdf_cmd!r} was not found."
            )
        else:
            warnings.append(
                "External veraPDF CLI not found on PATH; validated using built-in ISO 19005 structural inspector."
            )

        compliant = len(errors) == 0
        return PdfAValidationReport(
            compliant=compliant,
            profile=profile.value,
            validator_used=validator_used,
            errors=errors,
            warnings=warnings,
            details=details,
        )

    def _run_structural_checks(
        self,
        pdf_path: Path,
        profile: PdfAProfile,
        errors: list[str],
        warnings: list[str],
        details: dict[str, object],
    ) -> None:
        if not pdf_path.exists() or pdf_path.stat().st_size == 0:
            errors.append("Generated PDF file is missing or empty.")
            return

        raw_head = pdf_path.read_bytes()[:64]
        if not raw_head.startswith(b"%PDF-"):
            errors.append("File does not start with %PDF- header.")
            return

        expected_part = "1" if profile == PdfAProfile.PDFA_1B else "2"

        try:
            with pikepdf.open(pdf_path) as pdf:
                details["pdf_version"] = pdf.pdf_version
                details["page_count"] = len(pdf.pages)

                # 1. Encryption check (ISO 19005 forbids encryption)
                if pdf.is_encrypted:
                    errors.append("PDF/A forbids encryption, but PDF is encrypted.")

                # 2. Trailer /ID check
                trailer_id = pdf.trailer.get("/ID")
                if not trailer_id or len(trailer_id) < 2:
                    errors.append("PDF trailer is missing mandatory /ID array.")

                # 3. Document Catalog /Metadata XMP check
                meta_stream = pdf.Root.get("/Metadata")
                if not isinstance(meta_stream, pikepdf.Stream):
                    errors.append("Document Catalog is missing /Metadata XMP stream.")
                else:
                    if "/Filter" in meta_stream.stream_dict:
                        errors.append("PDF/A /Metadata XMP stream must be uncompressed.")
                    xmp_raw = meta_stream.read_raw_bytes()
                    self._validate_xmp_content(xmp_raw, expected_part, errors, details)

                # 4. Document Catalog /OutputIntents check
                output_intents = pdf.Root.get("/OutputIntents")
                if not output_intents or len(output_intents) == 0:
                    errors.append("Document Catalog is missing /OutputIntents array.")
                else:
                    has_pdfa_intent = False
                    for intent in output_intents:
                        if intent.get("/S") == pikepdf.Name("/GTS_PDFA1"):
                            dest_profile = intent.get("/DestOutputProfile")
                            if isinstance(dest_profile, pikepdf.Stream):
                                n_comp = int(dest_profile.get("/N", 0))
                                if n_comp in (1, 3, 4):
                                    has_pdfa_intent = True
                                    details["output_intent_components"] = n_comp
                    if not has_pdfa_intent:
                        errors.append(
                            "/OutputIntents does not contain a valid /GTS_PDFA1 profile with /DestOutputProfile."
                        )

                # 5. Page-level font embedding, ToUnicode, and XObject checks
                if len(pdf.pages) == 0:
                    errors.append("PDF contains 0 pages.")

                for page_idx, page in enumerate(pdf.pages, start=1):
                    res = page.obj.get("/Resources")
                    if not isinstance(res, pikepdf.Dictionary):
                        continue

                    fonts = res.get("/Font")
                    if isinstance(fonts, pikepdf.Dictionary):
                        for font_name, font_obj in fonts.items():
                            self._check_font_compliance(page_idx, str(font_name), font_obj, errors)

                    xobjs = res.get("/XObject")
                    if isinstance(xobjs, pikepdf.Dictionary):
                        for xname, xobj in xobjs.items():
                            self._check_xobject_compliance(page_idx, str(xname), xobj, errors)

        except Exception as exc:
            errors.append(f"Failed to open or inspect PDF structure: {exc}")

    @staticmethod
    def _validate_xmp_content(
        xmp_bytes: bytes,
        expected_part: str,
        errors: list[str],
        details: dict[str, object],
    ) -> None:
        try:
            text = xmp_bytes.decode("utf-8")
            root = ET.fromstring(text.strip())
        except Exception as exc:
            errors.append(f"Invalid XML in /Metadata XMP packet: {exc}")
            return

        ns = {"pdfaid": "http://www.aiim.org/pdfa/ns/id/"}
        part_el = root.find(".//pdfaid:part", ns)
        conf_el = root.find(".//pdfaid:conformance", ns)

        part_val = part_el.text.strip() if (part_el is not None and part_el.text) else None
        conf_val = conf_el.text.strip() if (conf_el is not None and conf_el.text) else None

        details["xmp_pdfaid_part"] = part_val
        details["xmp_pdfaid_conformance"] = conf_val

        if part_val != expected_part:
            errors.append(
                f"XMP pdfaid:part is {part_val!r}, expected {expected_part!r}."
            )
        if conf_val not in ("A", "B", "U"):
            errors.append(f"XMP pdfaid:conformance is {conf_val!r}, expected 'B', 'U', or 'A'.")

    @staticmethod
    def _check_font_compliance(
        page_num: int,
        font_key: str,
        font_obj: pikepdf.Object,
        errors: list[str],
    ) -> None:
        if not isinstance(font_obj, pikepdf.Dictionary):
            return
        subtype = font_obj.get("/Subtype")
        if subtype == pikepdf.Name("/Type0"):
            if "/ToUnicode" not in font_obj:
                errors.append(f"Page {page_num} Type0 font {font_key} is missing /ToUnicode CMap.")
            descendants = font_obj.get("/DescendantFonts")
            if not descendants or len(descendants) == 0:
                errors.append(f"Page {page_num} Type0 font {font_key} has no /DescendantFonts.")
                return
            cid_font = descendants[0]
            fdesc = cid_font.get("/FontDescriptor")
            if not isinstance(fdesc, pikepdf.Dictionary):
                errors.append(f"Page {page_num} CIDFont {font_key} is missing /FontDescriptor.")
                return
            if not any(k in fdesc for k in ("/FontFile", "/FontFile2", "/FontFile3")):
                errors.append(f"Page {page_num} CIDFont {font_key} is not embedded.")
        else:
            fdesc = font_obj.get("/FontDescriptor")
            if not isinstance(fdesc, pikepdf.Dictionary) or not any(
                k in fdesc for k in ("/FontFile", "/FontFile2", "/FontFile3")
            ):
                errors.append(f"Page {page_num} font {font_key} ({subtype}) is not embedded.")

    @staticmethod
    def _check_xobject_compliance(
        page_num: int,
        xname: str,
        xobj: pikepdf.Object,
        errors: list[str],
    ) -> None:
        if not isinstance(xobj, pikepdf.Stream):
            return
        if xobj.get("/Subtype") != pikepdf.Name("/Image"):
            return
        flt = xobj.get("/Filter")
        if flt == pikepdf.Name("/LZWDecode") or (
            isinstance(flt, pikepdf.Array) and pikepdf.Name("/LZWDecode") in flt
        ):
            errors.append(f"Page {page_num} Image {xname} uses forbidden /LZWDecode filter.")
        if "/F" in xobj.stream_dict:
            errors.append(f"Page {page_num} Image {xname} uses forbidden external stream /F.")

    @staticmethod
    def _run_verapdf_cli(
        verapdf_bin: str,
        pdf_path: Path,
        profile: PdfAProfile,
    ) -> tuple[bool, list[str]]:
        flavour = "1b" if profile == PdfAProfile.PDFA_1B else "2b"
        try:
            proc = subprocess.run(
                [verapdf_bin, "--flavour", flavour, "--format", "mrr", str(pdf_path)],
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )
            xml_out = proc.stdout or ""
            root = ET.fromstring(xml_out)
            report_el = root.find(".//validationReport")
            if report_el is not None:
                is_compliant = report_el.attrib.get("isCompliant", "false").lower() == "true"
                if is_compliant:
                    return True, []
                failed_rules = [
                    el.attrib.get("specification", "") + ":" + el.attrib.get("clause", "")
                    for el in report_el.findall(".//rule[@status='failed']")
                ]
                return False, [f"veraPDF failed rules: {failed_rules}"]
            return proc.returncode == 0, (
                [] if proc.returncode == 0 else [f"veraPDF exited with code {proc.returncode}"]
            )
        except Exception as exc:
            return False, [f"veraPDF execution error: {exc}"]
