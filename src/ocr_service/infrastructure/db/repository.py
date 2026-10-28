"""Durable job queue repository with lease management, crash recovery, and retention."""

from __future__ import annotations

import logging
import os
import socket
from collections.abc import Collection
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from sqlalchemy import delete, func, select, text

from ocr_service.domain.models import (
    BBoxPt,
    BBoxPx,
    CallbackStatus,
    CdnSourceRef,
    DpiSource,
    JobEntity,
    JobStatus,
    JobSubmissionOptions,
    JobType,
    PageNlpAnalysis,
    PageOcrResult,
    WordBox,
    can_transition,
)
from ocr_service.exceptions import (
    JobNotFoundError,
    JobStateConflictError,
    QueueBackpressureError,
)
from ocr_service.infrastructure.db.database import DatabaseSessionManager
from ocr_service.infrastructure.db.models import (
    JobPageRecord,
    JobRecord,
    WorkerHeartbeatRecord,
    utcnow,
)

logger = logging.getLogger(__name__)


def _ensure_utc(dt: datetime | None) -> datetime | None:
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def _arm_callback_on_terminal(rec: JobRecord, now: datetime) -> None:
    """Queue the completion callback once a job reaches a terminal state (outbox pattern).

    Called in the same transaction that writes the terminal status, so a crash between the
    state change and the HTTP delivery can never lose the notification.
    """
    if rec.callback_url and rec.callback_status == CallbackStatus.WAITING.value:
        rec.callback_status = CallbackStatus.PENDING.value
        rec.callback_next_at = now


class JobRepository:
    """Transactional persistence for jobs, durable queue leases, and page results."""

    def __init__(self, db: DatabaseSessionManager) -> None:
        self._db = db

    @staticmethod
    def _to_entity(rec: JobRecord) -> JobEntity:
        return JobEntity(
            job_id=rec.job_id,
            status=JobStatus(rec.status),
            options=JobSubmissionOptions.from_dict(rec.options_json or {}),
            input_filenames=list(rec.input_filenames_json or []),
            content_sha256=rec.content_sha256,
            idempotency_key=rec.idempotency_key,
            correlation_id=rec.correlation_id,
            page_count=rec.page_count,
            completed_pages=rec.completed_pages,
            attempt_count=rec.attempt_count,
            max_retries=rec.max_retries,
            cancel_requested=rec.cancel_requested,
            worker_id=rec.worker_id,
            error_code=rec.error_code,
            error_message=rec.error_message,
            retryable=rec.retryable,
            pdfa_compliant=rec.pdfa_compliant,
            pdfa_validator=rec.pdfa_validator,
            processing_duration_ms=rec.processing_duration_ms,
            created_at=_ensure_utc(rec.created_at),
            started_at=_ensure_utc(rec.started_at),
            completed_at=_ensure_utc(rec.completed_at),
            expires_at=_ensure_utc(rec.expires_at),
            updated_at=_ensure_utc(rec.updated_at),
            metadata_json=dict(rec.metadata_json or {}),
            job_type=JobType(rec.job_type or JobType.OCR.value),
            source_refs=[CdnSourceRef.from_dict(r) for r in (rec.source_refs_json or [])],
            callback_url=rec.callback_url,
            callback_status=CallbackStatus(rec.callback_status or CallbackStatus.NONE.value),
            callback_attempts=rec.callback_attempts or 0,
            callback_last_error=rec.callback_last_error,
        )

    @staticmethod
    def _to_page_result(rec: JobPageRecord, include_boxes: bool = True) -> PageOcrResult:
        words: list[WordBox] = []
        if include_boxes and rec.words_json:
            for w in rec.words_json:
                src_b = w["bbox_source_px"]
                pdf_b = w.get("bbox_pdf_pt")
                words.append(
                    WordBox(
                        text=w["text"],
                        confidence=float(w["confidence"]),
                        bbox_working_px=BBoxPx(
                            left=src_b["left"],
                            top=src_b["top"],
                            width=src_b["width"],
                            height=src_b["height"],
                        ),
                        bbox_source_px=BBoxPx(
                            left=src_b["left"],
                            top=src_b["top"],
                            width=src_b["width"],
                            height=src_b["height"],
                        ),
                        bbox_pdf_pt=(
                            BBoxPt(
                                x=pdf_b["x"],
                                y=pdf_b["y"],
                                width=pdf_b["width"],
                                height=pdf_b["height"],
                            )
                            if pdf_b
                            else None
                        ),
                        column_index=int(w.get("column_index", 0)),
                        reading_order_index=int(w.get("reading_order_index", 0)),
                    )
                )
        raw_warnings_or_envelope = rec.warnings_json
        if isinstance(raw_warnings_or_envelope, dict):
            warnings_list = [str(x) for x in raw_warnings_or_envelope.get("warnings", [])]
            nlp_data = raw_warnings_or_envelope.get("nlp")
            nlp_obj = PageNlpAnalysis.from_dict(nlp_data) if isinstance(nlp_data, dict) else None
        elif isinstance(raw_warnings_or_envelope, list):
            warnings_list = [str(x) for x in raw_warnings_or_envelope]
            nlp_obj = None
        else:
            warnings_list = []
            nlp_obj = None

        return PageOcrResult(
            page_number=rec.page_number,
            width_px=rec.width_px,
            height_px=rec.height_px,
            width_pt=rec.width_pt,
            height_pt=rec.height_pt,
            dpi_x=rec.dpi_x,
            dpi_y=rec.dpi_y,
            dpi_source=DpiSource(rec.dpi_source),
            rotation_deg=rec.rotation_deg,
            deskew_angle_deg=rec.deskew_angle_deg,
            ocr_applied=rec.ocr_applied,
            mean_confidence=rec.mean_confidence,
            column_count=rec.column_count,
            text=rec.text,
            words=words,
            processing_duration_ms=rec.processing_duration_ms,
            image_preservation_mode=rec.image_preservation_mode,
            warnings=warnings_list,
            nlp=nlp_obj,
        )

    def find_idempotent_job(
        self, idempotency_key: str, content_sha256: str
    ) -> JobEntity | None:
        """Return existing non-expired job matching idempotency_key + content_sha256."""
        with self._db.session() as sess:
            stmt = (
                select(JobRecord)
                .where(
                    JobRecord.idempotency_key == idempotency_key,
                    JobRecord.content_sha256 == content_sha256,
                    JobRecord.status.in_(
                        [
                            JobStatus.QUEUED.value,
                            JobStatus.PROCESSING.value,
                            JobStatus.COMPLETED.value,
                        ]
                    ),
                )
                .order_by(JobRecord.created_at.desc())
                .limit(1)
            )
            rec = sess.execute(stmt).scalar_one_or_none()
            return self._to_entity(rec) if rec else None

    def enqueue_job(
        self,
        *,
        job_id: str,
        options: JobSubmissionOptions,
        input_filenames: list[str],
        input_paths: list[Path],
        content_sha256: str,
        idempotency_key: str | None,
        correlation_id: str,
        max_retries: int,
        max_queue_depth: int,
        job_type: JobType = JobType.OCR,
        source_refs: list[CdnSourceRef] | None = None,
        callback_url: str | None = None,
        extra_metadata: dict[str, Any] | None = None,
    ) -> JobEntity:
        """Persist a new job in QUEUED state, enforcing queue backpressure (FR-02, FR-15).

        Backpressure is counted per job type, so a large OCR backlog never blocks
        submissions of other job types.
        """
        now = utcnow()
        with self._db.session() as sess:
            queued_count = (
                sess.execute(
                    select(func.count(JobRecord.job_id)).where(
                        JobRecord.status == JobStatus.QUEUED.value,
                        JobRecord.job_type == job_type.value,
                    )
                ).scalar_one()
                or 0
            )
            if queued_count >= max_queue_depth:
                raise QueueBackpressureError(
                    f"Job queue depth ({queued_count}) reached maximum capacity ({max_queue_depth}).",
                    details={
                        "queued_jobs": queued_count,
                        "max_queue_depth": max_queue_depth,
                        "job_type": job_type.value,
                    },
                )

            rec = JobRecord(
                job_id=job_id,
                job_type=job_type.value,
                source_refs_json=[r.to_dict() for r in (source_refs or [])],
                callback_url=callback_url,
                callback_status=(
                    CallbackStatus.WAITING.value if callback_url else CallbackStatus.NONE.value
                ),
                callback_attempts=0,
                status=JobStatus.QUEUED.value,
                options_json=options.to_dict(),
                input_filenames_json=input_filenames,
                input_paths_json=[str(p) for p in input_paths],
                content_sha256=content_sha256,
                idempotency_key=idempotency_key,
                correlation_id=correlation_id,
                page_count=0,
                completed_pages=0,
                attempt_count=0,
                max_retries=max_retries,
                cancel_requested=False,
                next_retry_at=now,
                created_at=now,
                updated_at=now,
                metadata_json={
                    **(extra_metadata or {}),
                    "events": [{"state": "queued", "at": now.isoformat()}],
                },
            )
            sess.add(rec)
            sess.flush()
            return self._to_entity(rec)

    def get_job(self, job_id: str) -> JobEntity:
        with self._db.session() as sess:
            rec = sess.get(JobRecord, job_id)
            if rec is None:
                raise JobNotFoundError(f"Job {job_id!r} was not found.")
            return self._to_entity(rec)

    def get_job_input_paths(self, job_id: str) -> list[Path]:
        with self._db.session() as sess:
            rec = sess.get(JobRecord, job_id)
            if rec is None:
                raise JobNotFoundError(f"Job {job_id!r} was not found.")
            return [Path(p) for p in (rec.input_paths_json or [])]

    def set_job_input_paths(self, job_id: str, input_paths: list[Path]) -> None:
        """Record the local paths of inputs the worker downloaded for a CDN-referenced job."""
        with self._db.session() as sess:
            rec = sess.get(JobRecord, job_id)
            if rec is None:
                raise JobNotFoundError(f"Job {job_id!r} was not found.")
            rec.input_paths_json = [str(p) for p in input_paths]
            rec.updated_at = utcnow()

    def get_job_pages(
        self,
        job_id: str,
        *,
        page_start: int | None = None,
        page_end: int | None = None,
        include_boxes: bool = False,
    ) -> list[PageOcrResult]:
        with self._db.session() as sess:
            stmt = select(JobPageRecord).where(JobPageRecord.job_id == job_id)
            if page_start is not None:
                stmt = stmt.where(JobPageRecord.page_number >= page_start)
            if page_end is not None:
                stmt = stmt.where(JobPageRecord.page_number <= page_end)
            stmt = stmt.order_by(JobPageRecord.page_number.asc())
            rows = sess.execute(stmt).scalars().all()
            return [self._to_page_result(r, include_boxes=include_boxes) for r in rows]

    def claim_next_job(
        self,
        *,
        worker_id: str,
        lease_seconds: int,
        max_concurrent_jobs: int,
        job_types: Collection[str] | None = None,
        max_concurrent_by_type: dict[str, int] | None = None,
    ) -> JobEntity | None:
        """Atomically claim the oldest eligible QUEUED job if active concurrency permits.

        Only ``job_types`` are considered (default: OCR). Concurrency is capped per job type:
        ``max_concurrent_by_type[type]`` if given, otherwise ``max_concurrent_jobs``.
        """
        now = utcnow()
        lease_until = now + timedelta(seconds=lease_seconds)
        wanted_types = list(job_types) if job_types else [JobType.OCR.value]
        caps = max_concurrent_by_type or {}

        with self._db.session() as sess:
            # Acquire immediate write lock on SQLite to prevent race conditions across processes
            if self._db.engine.dialect.name == "sqlite":
                sess.execute(text("BEGIN IMMEDIATE"))

            active_by_type = dict(
                sess.execute(
                    select(JobRecord.job_type, func.count(JobRecord.job_id))
                    .where(
                        JobRecord.status == JobStatus.PROCESSING.value,
                        JobRecord.lease_expires_at > now,
                        JobRecord.job_type.in_(wanted_types),
                    )
                    .group_by(JobRecord.job_type)
                ).all()
            )
            eligible_types = [
                t
                for t in wanted_types
                if active_by_type.get(t, 0) < caps.get(t, max_concurrent_jobs)
            ]
            if not eligible_types:
                return None

            candidate = (
                sess.execute(
                    select(JobRecord)
                    .where(
                        JobRecord.status == JobStatus.QUEUED.value,
                        JobRecord.next_retry_at <= now,
                        JobRecord.job_type.in_(eligible_types),
                    )
                    .order_by(JobRecord.created_at.asc())
                    .limit(1)
                )
                .scalars()
                .first()
            )
            if candidate is None:
                return None

            if candidate.cancel_requested:
                candidate.status = JobStatus.CANCELLED.value
                candidate.completed_at = now
                candidate.updated_at = now
                _arm_callback_on_terminal(candidate, now)
                sess.flush()
                return None

            candidate.status = JobStatus.PROCESSING.value
            candidate.worker_id = worker_id
            candidate.lease_expires_at = lease_until
            candidate.attempt_count += 1
            if candidate.started_at is None:
                candidate.started_at = now
            candidate.updated_at = now

            meta = dict(candidate.metadata_json or {})
            events = list(meta.get("events", []))
            events.append(
                {
                    "state": "processing",
                    "worker_id": worker_id,
                    "attempt": candidate.attempt_count,
                    "at": now.isoformat(),
                }
            )
            meta["events"] = events
            candidate.metadata_json = meta
            sess.flush()
            return self._to_entity(candidate)

    def set_job_page_count(self, job_id: str, page_count: int) -> None:
        with self._db.session() as sess:
            rec = sess.get(JobRecord, job_id)
            if rec:
                rec.page_count = page_count
                rec.updated_at = utcnow()

    def save_page_checkpoint(
        self,
        job_id: str,
        worker_id: str,
        page_result: PageOcrResult,
        lease_seconds: int,
    ) -> bool:
        """Persist a single page's OCR result, renew worker lease, and return cancel_requested."""
        now = utcnow()
        with self._db.session() as sess:
            rec = sess.get(JobRecord, job_id)
            if rec is None:
                return True

            existing = sess.execute(
                select(JobPageRecord).where(
                    JobPageRecord.job_id == job_id,
                    JobPageRecord.page_number == page_result.page_number,
                )
            ).scalar_one_or_none()

            words_serialized = [w.to_dict() for w in page_result.words]
            warnings_envelope: Any = (
                {
                    "warnings": page_result.warnings,
                    "nlp": page_result.nlp.to_dict(),
                }
                if page_result.nlp is not None
                else page_result.warnings
            )
            if existing is None:
                page_rec = JobPageRecord(
                    job_id=job_id,
                    page_number=page_result.page_number,
                    width_px=page_result.width_px,
                    height_px=page_result.height_px,
                    width_pt=page_result.width_pt,
                    height_pt=page_result.height_pt,
                    dpi_x=page_result.dpi_x,
                    dpi_y=page_result.dpi_y,
                    dpi_source=page_result.dpi_source.value,
                    rotation_deg=page_result.rotation_deg,
                    deskew_angle_deg=page_result.deskew_angle_deg,
                    ocr_applied=page_result.ocr_applied,
                    mean_confidence=page_result.mean_confidence,
                    column_count=page_result.column_count,
                    processing_duration_ms=page_result.processing_duration_ms,
                    image_preservation_mode=page_result.image_preservation_mode,
                    text=page_result.text,
                    words_json=words_serialized,
                    warnings_json=warnings_envelope,
                )
                sess.add(page_rec)
            else:
                existing.width_px = page_result.width_px
                existing.height_px = page_result.height_px
                existing.width_pt = page_result.width_pt
                existing.height_pt = page_result.height_pt
                existing.dpi_x = page_result.dpi_x
                existing.dpi_y = page_result.dpi_y
                existing.dpi_source = page_result.dpi_source.value
                existing.rotation_deg = page_result.rotation_deg
                existing.deskew_angle_deg = page_result.deskew_angle_deg
                existing.ocr_applied = page_result.ocr_applied
                existing.mean_confidence = page_result.mean_confidence
                existing.column_count = page_result.column_count
                existing.processing_duration_ms = page_result.processing_duration_ms
                existing.image_preservation_mode = page_result.image_preservation_mode
                existing.text = page_result.text
                existing.words_json = words_serialized
                existing.warnings_json = warnings_envelope

            sess.flush()
            completed_cnt = (
                sess.execute(
                    select(func.count(JobPageRecord.id)).where(JobPageRecord.job_id == job_id)
                ).scalar_one()
                or 0
            )
            rec.completed_pages = completed_cnt
            rec.worker_id = worker_id
            rec.lease_expires_at = now + timedelta(seconds=lease_seconds)
            rec.updated_at = now
            return bool(rec.cancel_requested or rec.status == JobStatus.CANCELLED.value)

    def set_job_progress(self, job_id: str, *, total: int, completed: int) -> None:
        """Record item progress (pages for OCR, files for packaging) for status polling."""
        with self._db.session() as sess:
            rec = sess.get(JobRecord, job_id)
            if rec is None:
                raise JobNotFoundError(f"Job {job_id!r} was not found.")
            rec.page_count = total
            rec.completed_pages = completed
            rec.updated_at = utcnow()

    def merge_job_metadata(self, job_id: str, data: dict[str, Any]) -> None:
        """Shallow-merge keys into the job's metadata (e.g. a package result summary)."""
        with self._db.session() as sess:
            rec = sess.get(JobRecord, job_id)
            if rec is None:
                raise JobNotFoundError(f"Job {job_id!r} was not found.")
            meta = dict(rec.metadata_json or {})
            meta.update(data)
            rec.metadata_json = meta
            rec.updated_at = utcnow()

    def renew_job_lease(self, job_id: str, worker_id: str, lease_seconds: int) -> bool:
        """Extend the worker lease and check if cancellation was requested."""
        now = utcnow()
        with self._db.session() as sess:
            rec = sess.get(JobRecord, job_id)
            if rec is None or rec.status != JobStatus.PROCESSING.value:
                return True
            if rec.worker_id == worker_id:
                rec.lease_expires_at = now + timedelta(seconds=lease_seconds)
                rec.updated_at = now
            return bool(rec.cancel_requested)

    def mark_completed(
        self,
        *,
        job_id: str,
        page_count: int,
        processing_duration_ms: int,
        retention_hours: float,
        pdfa_compliant: bool | None,
        pdfa_validator: str | None,
        extra_metadata: dict[str, Any] | None = None,
    ) -> JobEntity:
        now = utcnow()
        expires_at = now + timedelta(hours=retention_hours)
        with self._db.session() as sess:
            rec = sess.get(JobRecord, job_id)
            if rec is None:
                raise JobNotFoundError(f"Job {job_id!r} not found.")

            current = JobStatus(rec.status)
            if not can_transition(current, JobStatus.COMPLETED):
                raise JobStateConflictError(
                    f"Cannot transition job {job_id} from {current.value} to completed."
                )

            rec.status = JobStatus.COMPLETED.value
            rec.page_count = page_count
            rec.completed_pages = page_count
            rec.processing_duration_ms = processing_duration_ms
            rec.pdfa_compliant = pdfa_compliant
            rec.pdfa_validator = pdfa_validator
            rec.completed_at = now
            rec.expires_at = expires_at
            rec.lease_expires_at = None
            rec.error_code = None
            rec.error_message = None
            rec.updated_at = now
            _arm_callback_on_terminal(rec, now)

            meta = dict(rec.metadata_json or {})
            if extra_metadata:
                meta.update(extra_metadata)
            events = list(meta.get("events", []))
            events.append({"state": "completed", "at": now.isoformat()})
            meta["events"] = events
            rec.metadata_json = meta
            sess.flush()
            return self._to_entity(rec)

    def mark_failed_or_retry(
        self,
        *,
        job_id: str,
        error_code: str,
        error_message: str,
        retryable: bool,
        failed_retention_hours: float,
        backoff_base_seconds: float = 2.0,
    ) -> JobEntity:
        """Schedule retry if retryable and attempts remain; otherwise transition to FAILED."""
        now = utcnow()
        with self._db.session() as sess:
            rec = sess.get(JobRecord, job_id)
            if rec is None:
                raise JobNotFoundError(f"Job {job_id!r} not found.")

            meta = dict(rec.metadata_json or {})
            events = list(meta.get("events", []))

            can_retry = retryable and (rec.attempt_count <= rec.max_retries)
            if can_retry:
                delay = backoff_base_seconds * (2 ** max(0, rec.attempt_count - 1))
                rec.status = JobStatus.QUEUED.value
                rec.next_retry_at = now + timedelta(seconds=delay)
                rec.lease_expires_at = None
                rec.worker_id = None
                rec.error_code = error_code
                rec.error_message = error_message
                rec.retryable = True
                rec.updated_at = now
                events.append(
                    {
                        "state": "retry_scheduled",
                        "error_code": error_code,
                        "attempt": rec.attempt_count,
                        "next_retry_at": rec.next_retry_at.isoformat(),
                        "at": now.isoformat(),
                    }
                )
            else:
                rec.status = JobStatus.FAILED.value
                rec.completed_at = now
                rec.expires_at = now + timedelta(hours=failed_retention_hours)
                rec.lease_expires_at = None
                rec.error_code = error_code
                rec.error_message = error_message
                rec.retryable = retryable
                rec.updated_at = now
                _arm_callback_on_terminal(rec, now)
                events.append(
                    {
                        "state": "failed",
                        "error_code": error_code,
                        "retryable": retryable,
                        "at": now.isoformat(),
                    }
                )

            meta["events"] = events
            rec.metadata_json = meta
            sess.flush()
            return self._to_entity(rec)

    def request_cancel(self, job_id: str, retention_hours: float = 1.0) -> JobEntity:
        """Cancel a QUEUED job immediately or flag a PROCESSING job for cooperative cancellation."""
        now = utcnow()
        with self._db.session() as sess:
            rec = sess.get(JobRecord, job_id)
            if rec is None:
                raise JobNotFoundError(f"Job {job_id!r} not found.")

            current = JobStatus(rec.status)
            if current == JobStatus.CANCELLED:
                return self._to_entity(rec)

            if current in {JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.EXPIRED}:
                raise JobStateConflictError(
                    f"Job {job_id!r} is already in terminal state {current.value!r} and cannot be cancelled."
                )

            rec.cancel_requested = True
            meta = dict(rec.metadata_json or {})
            events = list(meta.get("events", []))

            if current == JobStatus.QUEUED:
                rec.status = JobStatus.CANCELLED.value
                rec.completed_at = now
                rec.expires_at = now + timedelta(hours=retention_hours)
                rec.lease_expires_at = None
                _arm_callback_on_terminal(rec, now)
                events.append({"state": "cancelled", "at": now.isoformat()})
            else:
                events.append({"state": "cancel_requested", "at": now.isoformat()})

            meta["events"] = events
            rec.metadata_json = meta
            rec.updated_at = now
            sess.flush()
            return self._to_entity(rec)

    def finalize_cancelled_job(self, job_id: str, retention_hours: float = 1.0) -> JobEntity:
        now = utcnow()
        with self._db.session() as sess:
            rec = sess.get(JobRecord, job_id)
            if rec is None:
                raise JobNotFoundError(f"Job {job_id!r} not found.")
            rec.status = JobStatus.CANCELLED.value
            rec.cancel_requested = True
            rec.completed_at = now
            rec.expires_at = now + timedelta(hours=retention_hours)
            rec.lease_expires_at = None
            rec.updated_at = now
            _arm_callback_on_terminal(rec, now)
            meta = dict(rec.metadata_json or {})
            events = list(meta.get("events", []))
            events.append({"state": "cancelled", "at": now.isoformat()})
            meta["events"] = events
            rec.metadata_json = meta
            sess.flush()
            return self._to_entity(rec)

    def recover_stale_jobs(self, failed_retention_hours: float = 12.0) -> list[str]:
        """Recover jobs stuck in PROCESSING with expired worker leases (worker crash recovery)."""
        now = utcnow()
        recovered_ids: list[str] = []
        with self._db.session() as sess:
            stale_jobs = (
                sess.execute(
                    select(JobRecord).where(
                        JobRecord.status == JobStatus.PROCESSING.value,
                        JobRecord.lease_expires_at.is_not(None),
                        JobRecord.lease_expires_at < now,
                    )
                )
                .scalars()
                .all()
            )
            for rec in stale_jobs:
                meta = dict(rec.metadata_json or {})
                events = list(meta.get("events", []))
                if rec.cancel_requested:
                    rec.status = JobStatus.CANCELLED.value
                    rec.completed_at = now
                    rec.expires_at = now + timedelta(hours=1.0)
                    rec.lease_expires_at = None
                    _arm_callback_on_terminal(rec, now)
                    events.append({"state": "cancelled_after_stale_lease", "at": now.isoformat()})
                elif rec.attempt_count <= rec.max_retries:
                    rec.status = JobStatus.QUEUED.value
                    rec.next_retry_at = now
                    rec.lease_expires_at = None
                    rec.error_code = "WORKER_CRASH_RECOVERED"
                    rec.error_message = (
                        f"Worker {rec.worker_id} lease expired; job re-queued for recovery."
                    )
                    rec.retryable = True
                    events.append(
                        {
                            "state": "recovered_to_queued",
                            "stale_worker": rec.worker_id,
                            "attempt": rec.attempt_count,
                            "at": now.isoformat(),
                        }
                    )
                else:
                    rec.status = JobStatus.FAILED.value
                    rec.completed_at = now
                    rec.expires_at = now + timedelta(hours=failed_retention_hours)
                    rec.lease_expires_at = None
                    rec.error_code = "WORKER_CRASH_MAX_RETRIES"
                    rec.error_message = (
                        f"Worker {rec.worker_id} terminated unexpectedly and max retries ({rec.max_retries}) exhausted."
                    )
                    rec.retryable = False
                    _arm_callback_on_terminal(rec, now)
                    events.append(
                        {
                            "state": "failed_after_stale_lease",
                            "stale_worker": rec.worker_id,
                            "at": now.isoformat(),
                        }
                    )
                meta["events"] = events
                rec.metadata_json = meta
                rec.updated_at = now
                recovered_ids.append(rec.job_id)

        return recovered_ids

    def find_jobs_due_for_expiration(self, now: datetime | None = None) -> list[str]:
        """Return terminal jobs (COMPLETED, FAILED, CANCELLED) whose expires_at has passed."""
        check_time = now or utcnow()
        with self._db.session() as sess:
            stmt = select(JobRecord.job_id).where(
                JobRecord.status.in_(
                    [
                        JobStatus.COMPLETED.value,
                        JobStatus.FAILED.value,
                        JobStatus.CANCELLED.value,
                    ]
                ),
                JobRecord.expires_at.is_not(None),
                JobRecord.expires_at <= check_time,
            )
            return list(sess.execute(stmt).scalars().all())

    def mark_job_expired(self, job_id: str, metadata_retention_days: int = 30) -> bool:
        """Transition a terminal job to EXPIRED, clearing page word arrays while keeping summary metadata."""
        now = utcnow()
        with self._db.session() as sess:
            rec = sess.get(JobRecord, job_id)
            if rec is None:
                return False
            current = JobStatus(rec.status)
            if current in {JobStatus.QUEUED, JobStatus.PROCESSING}:
                # Never expire an active job (FR-09)
                return False

            rec.status = JobStatus.EXPIRED.value
            rec.updated_at = now
            meta = dict(rec.metadata_json or {})
            meta["expired_at"] = now.isoformat()
            meta["metadata_purge_after"] = (
                now + timedelta(days=metadata_retention_days)
            ).isoformat()
            events = list(meta.get("events", []))
            events.append({"state": "expired", "at": now.isoformat()})
            meta["events"] = events
            rec.metadata_json = meta

            # Delete heavy per-page OCR text & word bounding boxes from DB upon artifact expiration
            sess.execute(delete(JobPageRecord).where(JobPageRecord.job_id == job_id))
            return True

    def claim_due_callbacks(self, *, limit: int, lease_seconds: int) -> list[JobEntity]:
        """Lease a batch of due callbacks so concurrent worker processes never send the same one.

        The lease is just ``callback_next_at`` pushed into the future: if the sender dies
        mid-delivery, the callback simply becomes due again after ``lease_seconds``.
        """
        now = utcnow()
        with self._db.session() as sess:
            if self._db.engine.dialect.name == "sqlite":
                sess.execute(text("BEGIN IMMEDIATE"))
            rows = (
                sess.execute(
                    select(JobRecord)
                    .where(
                        JobRecord.callback_status == CallbackStatus.PENDING.value,
                        JobRecord.callback_next_at <= now,
                    )
                    .order_by(JobRecord.callback_next_at.asc())
                    .limit(limit)
                )
                .scalars()
                .all()
            )
            for rec in rows:
                rec.callback_next_at = now + timedelta(seconds=lease_seconds)
            sess.flush()
            return [self._to_entity(r) for r in rows]

    def record_callback_result(
        self,
        job_id: str,
        *,
        delivered: bool,
        error: str | None,
        max_attempts: int,
        backoff_base_seconds: float = 5.0,
        backoff_max_seconds: float = 3600.0,
    ) -> CallbackStatus:
        """Record one delivery attempt; schedule an exponential-backoff retry or give up."""
        now = utcnow()
        with self._db.session() as sess:
            rec = sess.get(JobRecord, job_id)
            if rec is None:
                raise JobNotFoundError(f"Job {job_id!r} not found.")
            rec.callback_attempts = (rec.callback_attempts or 0) + 1
            if delivered:
                rec.callback_status = CallbackStatus.DELIVERED.value
                rec.callback_next_at = None
                rec.callback_last_error = None
            elif rec.callback_attempts >= max_attempts:
                rec.callback_status = CallbackStatus.FAILED.value
                rec.callback_next_at = None
                rec.callback_last_error = error
                logger.error(
                    "Giving up on job completion callback",
                    extra={"job_id": job_id, "attempts": rec.callback_attempts, "error": error},
                )
            else:
                delay = min(
                    backoff_max_seconds,
                    backoff_base_seconds * (2 ** (rec.callback_attempts - 1)),
                )
                rec.callback_next_at = now + timedelta(seconds=delay)
                rec.callback_last_error = error
            return CallbackStatus(rec.callback_status)

    def record_worker_heartbeat(
        self,
        worker_id: str,
        status: str = "idle",
        active_job_id: str | None = None,
        increment_processed: bool = False,
    ) -> None:
        now = utcnow()
        with self._db.session() as sess:
            rec = sess.get(WorkerHeartbeatRecord, worker_id)
            if rec is None:
                rec = WorkerHeartbeatRecord(
                    worker_id=worker_id,
                    hostname=socket.gethostname(),
                    pid=os.getpid(),
                    status=status,
                    active_job_id=active_job_id,
                    last_heartbeat_at=now,
                    jobs_processed=1 if increment_processed else 0,
                )
                sess.add(rec)
            else:
                rec.status = status
                rec.active_job_id = active_job_id
                rec.last_heartbeat_at = now
                if increment_processed:
                    rec.jobs_processed += 1

    def get_queue_and_system_stats(self) -> dict[str, Any]:
        """Collect queue depth, status counts, average durations, and active worker count."""
        now = utcnow()
        with self._db.session() as sess:
            status_rows = sess.execute(
                select(JobRecord.status, func.count(JobRecord.job_id)).group_by(JobRecord.status)
            ).all()
            status_counts = {row[0]: int(row[1]) for row in status_rows}

            avg_doc_ms = sess.execute(
                select(func.avg(JobRecord.processing_duration_ms)).where(
                    JobRecord.status == JobStatus.COMPLETED.value,
                    JobRecord.processing_duration_ms.is_not(None),
                )
            ).scalar_one()

            total_pages = (
                sess.execute(select(func.count(JobPageRecord.id))).scalar_one() or 0
            )
            avg_page_ms = sess.execute(
                select(func.avg(JobPageRecord.processing_duration_ms))
            ).scalar_one()

            error_rows = sess.execute(
                select(JobRecord.error_code, func.count(JobRecord.job_id))
                .where(JobRecord.error_code.is_not(None))
                .group_by(JobRecord.error_code)
            ).all()
            error_counts = {str(r[0]): int(r[1]) for r in error_rows if r[0]}

            live_cutoff = now - timedelta(seconds=60)
            live_workers = (
                sess.execute(
                    select(func.count(WorkerHeartbeatRecord.worker_id)).where(
                        WorkerHeartbeatRecord.last_heartbeat_at >= live_cutoff
                    )
                ).scalar_one()
                or 0
            )

            return {
                "jobs_by_status": {
                    s.value: status_counts.get(s.value, 0) for s in JobStatus
                },
                "queue_depth": status_counts.get(JobStatus.QUEUED.value, 0),
                "active_jobs": status_counts.get(JobStatus.PROCESSING.value, 0),
                "total_pages_stored": int(total_pages),
                "avg_document_duration_ms": round(float(avg_doc_ms or 0.0), 2),
                "avg_page_duration_ms": round(float(avg_page_ms or 0.0), 2),
                "error_counters": error_counts,
                "live_workers": int(live_workers),
            }
