"""Pydantic v2 request and response schemas with OpenAPI documentation (FR-01, FR-07, FR-08)."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ocr_service.domain.models import (
    CallbackStatus,
    DpiSource,
    JobStatus,
    JobType,
    LayoutMode,
    NlpMode,
    PdfAProfile,
)


class CdnSourceSchema(BaseModel):
    path: str = Field(
        ...,
        min_length=1,
        max_length=1024,
        description="CDN-relative file path, e.g. 'AOS\\\\ab\\\\cd\\\\belge.pdf'.",
    )
    filename: str | None = Field(
        default=None,
        max_length=255,
        description="Original file name; defaults to the last path segment.",
    )


class CdnOcrJobRequest(BaseModel):
    """OCR job whose inputs the worker downloads from the internal CDN (no upload)."""

    sources: list[CdnSourceSchema] = Field(
        ..., min_length=1, max_length=500, description="1 PDF or 1..N images on the CDN."
    )
    language: str | None = None
    force_ocr: bool | None = None
    working_dpi: int | None = Field(default=None, ge=100, le=600)
    preserve_source_images: bool | None = None
    deskew_working_image: bool | None = None
    layout_mode: LayoutMode = LayoutMode.AUTO
    pdfa_profile: PdfAProfile | None = None
    enable_nlp: bool = False
    nlp_mode: NlpMode = NlpMode.OFF
    custom_topics: list[str] = Field(default_factory=list)
    callback_url: str | None = Field(
        default=None,
        max_length=512,
        description="Receives a signed POST when the job reaches a terminal state.",
    )


class ErrorDetailSchema(BaseModel):
    code: str = Field(..., examples=["UNSUPPORTED_MEDIA_TYPE"])
    message: str = Field(..., examples=["Unsupported file format."])
    retryable: bool = Field(default=False)
    correlation_id: str = Field(..., examples=["req-123456"])
    details: dict[str, Any] = Field(default_factory=dict)


class ErrorEnvelopeSchema(BaseModel):
    error: ErrorDetailSchema


class JobSubmitResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    job_id: str = Field(..., description="Unique job identifier.")
    status: JobStatus = Field(..., description="Current lifecycle status of the job.")
    created_at: datetime | None = Field(default=None)
    idempotent_replay: bool = Field(
        default=False,
        description="True if this response matched an existing job via Idempotency-Key.",
    )
    status_url: str
    result_url: str
    pages_url: str
    pdf_url: str
    metadata_url: str


class JobStatusResponse(BaseModel):
    job_id: str
    job_type: JobType = JobType.OCR
    status: JobStatus
    callback_status: CallbackStatus = CallbackStatus.NONE
    callback_attempts: int = 0
    callback_last_error: str | None = None
    page_count: int = 0
    completed_pages: int = 0
    progress_percent: float = 0.0
    attempt_count: int = 0
    max_retries: int = 2
    cancel_requested: bool = False
    error_code: str | None = None
    error_message: str | None = None
    retryable: bool = False
    pdfa_profile: str
    pdfa_compliant: bool | None = None
    created_at: datetime | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    expires_at: datetime | None = None
    processing_duration_ms: int | None = None
    pdf_url: str | None = None
    result_url: str | None = None
    pages_url: str | None = None
    metadata_url: str


class BBoxPxSchema(BaseModel):
    left: float
    top: float
    width: float
    height: float


class BBoxPtSchema(BaseModel):
    x: float
    y: float
    width: float
    height: float


class WordBoxSchema(BaseModel):
    text: str
    confidence: float
    bbox_source_px: BBoxPxSchema
    bbox_pdf_pt: BBoxPtSchema | None = None
    column_index: int = 0
    reading_order_index: int = 0


class TopicPredictionSchema(BaseModel):
    category_id: str
    category: str
    confidence: float
    matched_terms: list[str] = Field(default_factory=list)


class KeywordItemSchema(BaseModel):
    term: str
    score: float


class NamedEntitySchema(BaseModel):
    text: str
    normalized_text: str
    label: str
    confidence: float
    source: str = "hybrid_rule"
    word_indices: list[int] = Field(default_factory=list)
    bbox_source_px: BBoxPxSchema | None = None
    bbox_pdf_pt: BBoxPtSchema | None = None


class PageNlpAnalysisSchema(BaseModel):
    engine_used: str = "hybrid_fast_tr"
    processing_duration_ms: int = 0
    topics: list[TopicPredictionSchema] = Field(default_factory=list)
    keywords: list[KeywordItemSchema] = Field(default_factory=list)
    entities: list[NamedEntitySchema] = Field(default_factory=list)


class PageResultSchema(BaseModel):
    page_number: int
    text: str
    width_px: int
    height_px: int
    width_pt: float
    height_pt: float
    dpi_x: float
    dpi_y: float
    dpi_source: DpiSource
    rotation_deg: int = 0
    deskew_angle_deg: float = 0.0
    ocr_applied: bool = True
    mean_confidence: float = 0.0
    column_count: int = 1
    processing_duration_ms: int = 0
    image_preservation_mode: str = "lossless_passthrough"
    warnings: list[str] = Field(default_factory=list)
    nlp: PageNlpAnalysisSchema | None = None
    words: list[WordBoxSchema] | None = None


class JobStructuredResultResponse(BaseModel):
    """Structured OCR result matching FR-07."""

    job_id: str
    status: JobStatus
    page_count: int
    pdf_url: str
    pdfa_profile: PdfAProfile
    pdfa_compliant: bool
    processing_duration_ms: int | None = None
    created_at: datetime | None = None
    completed_at: datetime | None = None
    expires_at: datetime | None = None
    document_nlp_summary: dict[str, Any] | None = None
    pages: list[PageResultSchema]


class JobPagesListResponse(BaseModel):
    job_id: str
    status: JobStatus
    total_pages: int
    returned_pages: int
    pages: list[PageResultSchema]


class PageTelemetrySchema(BaseModel):
    page_number: int
    width_px: int
    height_px: int
    width_pt: float
    height_pt: float
    dpi_x: float
    dpi_y: float
    dpi_source: str
    rotation_deg: int
    deskew_angle_deg: float
    ocr_applied: bool
    mean_confidence: float
    column_count: int
    word_count: int
    processing_duration_ms: int
    image_preservation_mode: str
    warnings: list[str] = Field(default_factory=list)


class JobMetadataResponse(BaseModel):
    job_id: str
    status: JobStatus
    options: dict[str, Any]
    input_filenames: list[str]
    content_sha256: str
    correlation_id: str
    page_count: int
    completed_pages: int
    attempt_count: int
    max_retries: int
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
    events: list[dict[str, Any]] = Field(default_factory=list)
    pdfa_validation: dict[str, Any] | None = None
    document_nlp_summary: dict[str, Any] | None = None
    page_telemetry: list[PageTelemetrySchema] = Field(default_factory=list)


class JobCancelResponse(BaseModel):
    job_id: str
    status: JobStatus
    cancel_requested: bool
    message: str


class HealthLiveResponse(BaseModel):
    status: str = "ok"
    service: str
    version: str


class HealthReadyResponse(BaseModel):
    status: str
    database_ok: bool
    storage_ok: bool
    ocr_engine_ok: bool
    ocr_engine_detail: str
    live_workers: int
    details: dict[str, Any] = Field(default_factory=dict)
