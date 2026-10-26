"""SQLAlchemy ORM models for durable OCR jobs, page checkpoints, and worker heartbeats."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Index, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def utcnow() -> datetime:
    return datetime.now(tz=UTC)


class Base(DeclarativeBase):
    pass


class JobRecord(Base):
    """Persistent job state and durable queue entry."""

    __tablename__ = "ocr_jobs"

    job_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    options_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    input_filenames_json: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    input_paths_json: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    content_sha256: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    idempotency_key: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    correlation_id: Mapped[str] = mapped_column(String(64), nullable=False, default="-")

    page_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    completed_pages: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    attempt_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    max_retries: Mapped[int] = mapped_column(Integer, nullable=False, default=2)
    cancel_requested: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    worker_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    lease_expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    next_retry_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utcnow
    )

    error_code: Mapped[str | None] = mapped_column(String(64), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    retryable: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    pdfa_compliant: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    pdfa_validator: Mapped[str | None] = mapped_column(String(64), nullable=True)
    processing_duration_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utcnow, index=True
    )
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True, index=True
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utcnow, onupdate=utcnow
    )

    pages: Mapped[list[JobPageRecord]] = relationship(
        "JobPageRecord",
        back_populates="job",
        cascade="all, delete-orphan",
        order_by="JobPageRecord.page_number",
    )

    __table_args__ = (
        Index("ix_ocr_jobs_queue_claim", "status", "next_retry_at", "created_at"),
        Index("ix_ocr_jobs_lease", "status", "lease_expires_at"),
    )


class JobPageRecord(Base):
    """Per-page extracted OCR text, geometry, and processing telemetry."""

    __tablename__ = "ocr_job_pages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("ocr_jobs.job_id", ondelete="CASCADE"), nullable=False, index=True
    )
    page_number: Mapped[int] = mapped_column(Integer, nullable=False)
    width_px: Mapped[int] = mapped_column(Integer, nullable=False)
    height_px: Mapped[int] = mapped_column(Integer, nullable=False)
    width_pt: Mapped[float] = mapped_column(Float, nullable=False)
    height_pt: Mapped[float] = mapped_column(Float, nullable=False)
    dpi_x: Mapped[float] = mapped_column(Float, nullable=False)
    dpi_y: Mapped[float] = mapped_column(Float, nullable=False)
    dpi_source: Mapped[str] = mapped_column(String(32), nullable=False)
    rotation_deg: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    deskew_angle_deg: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    ocr_applied: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    mean_confidence: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    column_count: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    processing_duration_ms: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    image_preservation_mode: Mapped[str] = mapped_column(
        String(64), nullable=False, default="lossless_passthrough"
    )
    text: Mapped[str] = mapped_column(Text, nullable=False, default="")
    words_json: Mapped[list[dict[str, Any]]] = mapped_column(JSON, nullable=False, default=list)
    warnings_json: Mapped[Any] = mapped_column(JSON, nullable=False, default=list)

    job: Mapped[JobRecord] = relationship("JobRecord", back_populates="pages")

    __table_args__ = (Index("ix_ocr_job_pages_unique", "job_id", "page_number", unique=True),)


class WorkerHeartbeatRecord(Base):
    """Track live worker processes for readiness probes and cluster observability."""

    __tablename__ = "ocr_worker_heartbeats"

    worker_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    hostname: Mapped[str] = mapped_column(String(128), nullable=False)
    pid: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="idle")
    active_job_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    last_heartbeat_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utcnow
    )
    jobs_processed: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
