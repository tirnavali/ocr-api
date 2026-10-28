"""Core domain entities, value objects, and job state transition rules."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any


class JobStatus(StrEnum):
    """Explicit job lifecycle states (FR-10)."""

    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    EXPIRED = "expired"


TERMINAL_STATUSES: frozenset[JobStatus] = frozenset(
    {JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED}
)


class JobType(StrEnum):
    """Kind of work a job performs; each type is claimed by its own worker pool."""

    OCR = "ocr"
    PACKAGE = "package"


class CallbackStatus(StrEnum):
    """Delivery state of the signed completion callback for a job."""

    NONE = "none"  # no callback_url configured
    WAITING = "waiting"  # job not terminal yet
    PENDING = "pending"  # terminal; delivery due at callback_next_at
    DELIVERED = "delivered"
    FAILED = "failed"  # gave up after CALLBACK_MAX_ATTEMPTS


@dataclass(frozen=True, slots=True)
class CdnSourceRef:
    """Job input referenced by its internal CDN path; the worker downloads it itself."""

    path: str
    filename: str

    def to_dict(self) -> dict[str, str]:
        return {"path": self.path, "filename": self.filename}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CdnSourceRef:
        return cls(path=str(data["path"]), filename=str(data["filename"]))


VALID_STATE_TRANSITIONS: dict[JobStatus, set[JobStatus]] = {
    JobStatus.QUEUED: {
        JobStatus.PROCESSING,
        JobStatus.CANCELLED,
        JobStatus.FAILED,
        JobStatus.EXPIRED,
    },
    JobStatus.PROCESSING: {
        JobStatus.COMPLETED,
        JobStatus.QUEUED,  # Retry after transient failure or worker crash recovery
        JobStatus.FAILED,
        JobStatus.CANCELLED,
    },
    JobStatus.COMPLETED: {
        JobStatus.EXPIRED,
    },
    JobStatus.FAILED: {
        JobStatus.QUEUED,  # Manual or scheduled retry
        JobStatus.EXPIRED,
    },
    JobStatus.CANCELLED: {
        JobStatus.EXPIRED,
    },
    JobStatus.EXPIRED: set(),
}


def can_transition(current: JobStatus, target: JobStatus) -> bool:
    """Return True if transitioning from current to target state is valid."""
    if current == target:
        return True
    return target in VALID_STATE_TRANSITIONS.get(current, set())


class DpiSource(StrEnum):
    DECLARED = "declared"
    ESTIMATED = "estimated"
    FALLBACK = "fallback"
    PDF_USER_SPACE = "pdf_user_space"


class LayoutMode(StrEnum):
    AUTO = "auto"
    SINGLE_COLUMN = "single_column"
    MULTI_COLUMN = "multi_column"


class PdfAProfile(StrEnum):
    PDFA_2B = "PDF/A-2b"
    PDFA_1B = "PDF/A-1b"


class NlpMode(StrEnum):
    """Turkish NLP execution mode on CPU."""

    OFF = "off"
    FAST = "fast"
    ONNX = "onnx"


class EntityLabel(StrEnum):
    """Named Entity Recognition (NER) labels for Turkish documents."""

    PER = "PER"  # Kişi (Person)
    LOC = "LOC"  # Yer / Şehir / Ülke (Location)
    ORG = "ORG"  # Kurum / Meclis / Bakanlık (Organization)
    DATE = "DATE"  # Tarih / Dönem (Date)
    LAW_REF = "LAW_REF"  # Kanun / Karar / Madde No (Legal Reference)


@dataclass(frozen=True, slots=True)
class BBoxPx:
    """Bounding box in image pixel coordinates (origin: top-left)."""

    left: float
    top: float
    width: float
    height: float

    @property
    def right(self) -> float:
        return self.left + self.width

    @property
    def bottom(self) -> float:
        return self.top + self.height

    @property
    def center_x(self) -> float:
        return self.left + self.width / 2.0

    @property
    def center_y(self) -> float:
        return self.top + self.height / 2.0

    def to_dict(self) -> dict[str, float]:
        return {
            "left": round(self.left, 2),
            "top": round(self.top, 2),
            "width": round(self.width, 2),
            "height": round(self.height, 2),
        }


@dataclass(frozen=True, slots=True)
class BBoxPt:
    """Bounding box in PDF user space points (1 pt = 1/72 inch, origin: bottom-left)."""

    x: float  # Left X in points
    y: float  # Bottom Y in points
    width: float
    height: float

    @property
    def right(self) -> float:
        return self.x + self.width

    @property
    def top(self) -> float:
        return self.y + self.height

    def to_dict(self) -> dict[str, float]:
        return {
            "x": round(self.x, 3),
            "y": round(self.y, 3),
            "width": round(self.width, 3),
            "height": round(self.height, 3),
        }


@dataclass(slots=True)
class WordBox:
    """Recognized word with pixel and PDF point geometry and OCR confidence."""

    text: str
    confidence: float
    bbox_working_px: BBoxPx
    bbox_source_px: BBoxPx
    bbox_pdf_pt: BBoxPt | None = None
    baseline_pdf_y: float | None = None
    block_num: int = 0
    par_num: int = 0
    line_num: int = 0
    word_num: int = 0
    column_index: int = 0
    reading_order_index: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "confidence": round(self.confidence, 2),
            "bbox_source_px": self.bbox_source_px.to_dict(),
            "bbox_pdf_pt": self.bbox_pdf_pt.to_dict() if self.bbox_pdf_pt else None,
            "column_index": self.column_index,
            "reading_order_index": self.reading_order_index,
        }


@dataclass(slots=True)
class TopicPrediction:
    """Predicted document/page topic category with confidence and matched Turkish terms."""

    category_id: str
    category: str
    confidence: float
    matched_terms: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "category_id": self.category_id,
            "category": self.category,
            "confidence": round(self.confidence, 3),
            "matched_terms": self.matched_terms,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TopicPrediction:
        return cls(
            category_id=str(data.get("category_id", "")),
            category=str(data.get("category", "")),
            confidence=float(data.get("confidence", 0.0)),
            matched_terms=[str(t) for t in data.get("matched_terms", [])],
        )


@dataclass(slots=True)
class KeywordItem:
    """Extracted Turkish keyphrase/term with relevance score."""

    term: str
    score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "term": self.term,
            "score": round(self.score, 3),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> KeywordItem:
        return cls(
            term=str(data.get("term", "")),
            score=float(data.get("score", 0.0)),
        )


@dataclass(slots=True)
class NamedEntity:
    """Recognized Turkish Named Entity linked to WordBox pixel and PDF point coordinates."""

    text: str
    normalized_text: str
    label: EntityLabel
    confidence: float
    source: str = "hybrid_rule"
    word_indices: list[int] = field(default_factory=list)
    bbox_source_px: BBoxPx | None = None
    bbox_pdf_pt: BBoxPt | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "normalized_text": self.normalized_text,
            "label": self.label.value,
            "confidence": round(self.confidence, 3),
            "source": self.source,
            "word_indices": self.word_indices,
            "bbox_source_px": self.bbox_source_px.to_dict() if self.bbox_source_px else None,
            "bbox_pdf_pt": self.bbox_pdf_pt.to_dict() if self.bbox_pdf_pt else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> NamedEntity:
        px = data.get("bbox_source_px")
        pt = data.get("bbox_pdf_pt")
        return cls(
            text=str(data.get("text", "")),
            normalized_text=str(data.get("normalized_text", data.get("text", ""))),
            label=EntityLabel(data.get("label", EntityLabel.PER.value)),
            confidence=float(data.get("confidence", 0.0)),
            source=str(data.get("source", "hybrid_rule")),
            word_indices=[int(i) for i in data.get("word_indices", [])],
            bbox_source_px=BBoxPx(**px) if isinstance(px, dict) else None,
            bbox_pdf_pt=BBoxPt(**pt) if isinstance(pt, dict) else None,
        )


@dataclass(slots=True)
class PageNlpAnalysis:
    """Per-page Turkish Topic Assignment, Keyphrase Extraction, and NER output."""

    engine_used: str
    processing_duration_ms: int = 0
    topics: list[TopicPrediction] = field(default_factory=list)
    keywords: list[KeywordItem] = field(default_factory=list)
    entities: list[NamedEntity] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "engine_used": self.engine_used,
            "processing_duration_ms": self.processing_duration_ms,
            "topics": [t.to_dict() for t in self.topics],
            "keywords": [k.to_dict() for k in self.keywords],
            "entities": [e.to_dict() for e in self.entities],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PageNlpAnalysis:
        return cls(
            engine_used=str(data.get("engine_used", "hybrid_fast_tr")),
            processing_duration_ms=int(data.get("processing_duration_ms", 0)),
            topics=[TopicPrediction.from_dict(t) for t in data.get("topics", [])],
            keywords=[KeywordItem.from_dict(k) for k in data.get("keywords", [])],
            entities=[NamedEntity.from_dict(e) for e in data.get("entities", [])],
        )


@dataclass(slots=True)
class JobSubmissionOptions:
    """Per-job configurable OCR, PDF/A, and Turkish NLP parameters."""

    language: str = "tur"
    force_ocr: bool = False
    working_dpi: int = 300
    preserve_source_images: bool = True
    deskew_working_image: bool = True
    layout_mode: LayoutMode = LayoutMode.AUTO
    pdfa_profile: PdfAProfile = PdfAProfile.PDFA_2B
    enable_nlp: bool = True
    nlp_mode: NlpMode = NlpMode.FAST
    custom_topics: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "language": self.language,
            "force_ocr": self.force_ocr,
            "working_dpi": self.working_dpi,
            "preserve_source_images": self.preserve_source_images,
            "deskew_working_image": self.deskew_working_image,
            "layout_mode": self.layout_mode.value,
            "pdfa_profile": self.pdfa_profile.value,
            "enable_nlp": self.enable_nlp,
            "nlp_mode": self.nlp_mode.value,
            "custom_topics": self.custom_topics,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> JobSubmissionOptions:
        raw_nlp_mode = data.get("nlp_mode", NlpMode.FAST.value)
        enable_nlp = bool(data.get("enable_nlp", True))
        if raw_nlp_mode == NlpMode.OFF.value:
            enable_nlp = False
        return cls(
            language=str(data.get("language", "tur")),
            force_ocr=bool(data.get("force_ocr", False)),
            working_dpi=int(data.get("working_dpi", 300)),
            preserve_source_images=bool(data.get("preserve_source_images", True)),
            deskew_working_image=bool(data.get("deskew_working_image", True)),
            layout_mode=LayoutMode(data.get("layout_mode", LayoutMode.AUTO.value)),
            pdfa_profile=PdfAProfile(data.get("pdfa_profile", PdfAProfile.PDFA_2B.value)),
            enable_nlp=enable_nlp,
            nlp_mode=NlpMode(raw_nlp_mode),
            custom_topics=[str(t).strip() for t in data.get("custom_topics", []) if str(t).strip()],
        )


@dataclass(slots=True)
class PageSourceDescriptor:
    """Normalized descriptor for a single input page extracted from PDF, TIFF, or image."""

    page_number: int
    source_path: Path
    source_format: str  # "JPEG", "PNG", "TIFF", "PDF_PAGE"
    width_px: int
    height_px: int
    width_pt: float
    height_pt: float
    dpi_x: float
    dpi_y: float
    dpi_source: DpiSource
    rotation_deg: int = 0
    has_usable_text: bool = False
    existing_text: str = ""
    source_pdf_page_index: int | None = None
    tiff_frame_index: int | None = None
    color_mode: str = "RGB"


@dataclass(slots=True)
class PageOcrResult:
    """Complete OCR and geometry output for a single page."""

    page_number: int
    width_px: int
    height_px: int
    width_pt: float
    height_pt: float
    dpi_x: float
    dpi_y: float
    dpi_source: DpiSource
    rotation_deg: int
    deskew_angle_deg: float
    ocr_applied: bool
    mean_confidence: float
    column_count: int
    text: str
    words: list[WordBox] = field(default_factory=list)
    processing_duration_ms: int = 0
    image_preservation_mode: str = "lossless_passthrough"
    warnings: list[str] = field(default_factory=list)
    nlp: PageNlpAnalysis | None = None

    def to_dict(self, include_boxes: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "page_number": self.page_number,
            "width_px": self.width_px,
            "height_px": self.height_px,
            "width_pt": round(self.width_pt, 2),
            "height_pt": round(self.height_pt, 2),
            "dpi_x": round(self.dpi_x, 2),
            "dpi_y": round(self.dpi_y, 2),
            "dpi_source": self.dpi_source.value,
            "rotation_deg": self.rotation_deg,
            "deskew_angle_deg": round(self.deskew_angle_deg, 3),
            "ocr_applied": self.ocr_applied,
            "mean_confidence": round(self.mean_confidence, 2),
            "column_count": self.column_count,
            "processing_duration_ms": self.processing_duration_ms,
            "image_preservation_mode": self.image_preservation_mode,
            "text": self.text,
            "nlp": self.nlp.to_dict() if self.nlp else None,
        }
        if self.warnings:
            payload["warnings"] = self.warnings
        if include_boxes:
            payload["words"] = [w.to_dict() for w in self.words]
        return payload


@dataclass(slots=True)
class PdfAValidationReport:
    """Result of PDF/A conformance validation."""

    compliant: bool
    profile: str
    validator_used: str  # "structural+verapdf" or "structural"
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "compliant": self.compliant,
            "profile": self.profile,
            "validator_used": self.validator_used,
            "errors": self.errors,
            "warnings": self.warnings,
            "details": self.details,
        }


@dataclass(slots=True)
class JobEntity:
    """Domain representation of a durable OCR job."""

    job_id: str
    status: JobStatus
    options: JobSubmissionOptions
    input_filenames: list[str]
    content_sha256: str
    idempotency_key: str | None
    correlation_id: str
    page_count: int = 0
    completed_pages: int = 0
    attempt_count: int = 0
    max_retries: int = 2
    cancel_requested: bool = False
    worker_id: str | None = None
    error_code: str | None = None
    error_message: str | None = None
    retryable: bool = False
    pdfa_compliant: bool | None = None
    pdfa_validator: str | None = None
    processing_duration_ms: int | None = None
    created_at: datetime | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    expires_at: datetime | None = None
    updated_at: datetime | None = None
    metadata_json: dict[str, Any] = field(default_factory=dict)
    job_type: JobType = JobType.OCR
    source_refs: list[CdnSourceRef] = field(default_factory=list)
    callback_url: str | None = None
    callback_status: CallbackStatus = CallbackStatus.NONE
    callback_attempts: int = 0
    callback_last_error: str | None = None
