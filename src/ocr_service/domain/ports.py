"""Architectural port interfaces for OCR Engine, PDF/A Validator, and Storage."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from ocr_service.domain.models import (
    NlpMode,
    PageNlpAnalysis,
    PageOcrResult,
    PdfAProfile,
    PdfAValidationReport,
    WordBox,
)


@dataclass(slots=True)
class OcrRecognitionOutput:
    """Raw output from an OCR engine on a working image."""

    words: list[WordBox]
    raw_text: str
    detected_orientation_deg: int = 0
    mean_confidence: float = 0.0


class OcrEnginePort(Protocol):
    """Abstraction over the underlying OCR engine (FR-04)."""

    def recognize_page(
        self,
        working_image_path: Path,
        *,
        language: str = "tur",
        dpi: int = 300,
        psm: int = 3,
        timeout_seconds: int = 180,
    ) -> OcrRecognitionOutput:
        """Run OCR on a preprocessed working image and return word boxes + text."""
        ...

    def detect_orientation(
        self,
        working_image_path: Path,
        *,
        timeout_seconds: int = 30,
    ) -> int:
        """Return detected cardinal orientation in degrees (0, 90, 180, 270)."""
        ...

    def check_health(self, required_language: str = "tur") -> tuple[bool, str]:
        """Verify OCR binary and required language model availability."""
        ...


class PdfAValidatorPort(Protocol):
    """Abstraction over PDF/A conformance validation (FR-05)."""

    def validate(
        self,
        pdf_path: Path,
        profile: PdfAProfile = PdfAProfile.PDFA_2B,
    ) -> PdfAValidationReport:
        """Validate that the generated PDF conforms to the target PDF/A profile."""
        ...


class TurkishNlpAnalyzerPort(Protocol):
    """Abstraction over CPU-based Turkish Topic Assignment and NER extraction."""

    def analyze_page(
        self,
        text: str,
        words: list[WordBox],
        *,
        page_height_px: int = 0,
        mode: NlpMode = NlpMode.FAST,
        custom_topics: list[str] | None = None,
    ) -> PageNlpAnalysis:
        """Run Turkish Topic Classification, Keyphrase Extraction, and NER on a page."""
        ...

    def summarize_document(
        self,
        page_results: list[PageOcrResult],
    ) -> dict[str, Any]:
        """Aggregate page-level topics, keywords, and named entities across the document."""
        ...
