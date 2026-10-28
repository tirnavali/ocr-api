"""Versioned OCR job submission, status, result, PDF/A download, metadata, and cancellation endpoints."""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, Header, Query, UploadFile, status
from fastapi.responses import FileResponse

from ocr_service.api.dependencies import (
    get_app_settings,
    get_job_repository,
    get_job_storage,
    verify_api_key,
)
from ocr_service.api.schemas import (
    BBoxPtSchema,
    BBoxPxSchema,
    CdnOcrJobRequest,
    JobCancelResponse,
    JobDeleteResponse,
    JobMetadataResponse,
    JobPagesListResponse,
    JobStatusResponse,
    JobStructuredResultResponse,
    JobSubmitResponse,
    PackageJobRequest,
    PackagePartSchema,
    PackageResultSchema,
    PackageSkippedFileSchema,
    PageNlpAnalysisSchema,
    PageResultSchema,
    PageTelemetrySchema,
    WordBoxSchema,
)
from ocr_service.config import Settings
from ocr_service.domain.models import (
    CdnSourceRef,
    JobEntity,
    JobStatus,
    JobSubmissionOptions,
    JobType,
    LayoutMode,
    NlpMode,
    PackageEntry,
    PageOcrResult,
    PdfAProfile,
)
from ocr_service.exceptions import (
    JobExpiredError,
    JobNotFoundError,
    JobNotReadyError,
    ValidationError,
)
from ocr_service.infrastructure.callbacks.notifier import validate_callback_url
from ocr_service.infrastructure.cdn.cdn_client import normalize_cdn_path
from ocr_service.infrastructure.db.repository import JobRepository
from ocr_service.infrastructure.storage.local_storage import LocalJobStorage, sanitize_filename
from ocr_service.logging_config import correlation_id_ctx
from ocr_service.pipeline.inspector import detect_file_format_by_magic
from ocr_service.pipeline.packager import (
    PACKAGE_ENTRIES_FILENAME,
    package_part_path,
    write_package_entries,
)

logger = logging.getLogger(__name__)

router = APIRouter(dependencies=[Depends(verify_api_key)], tags=["OCR Jobs"])


def _build_submit_response(job: JobEntity, idempotent_replay: bool = False) -> JobSubmitResponse:
    jid = job.job_id
    return JobSubmitResponse(
        job_id=jid,
        status=job.status,
        created_at=job.created_at,
        idempotent_replay=idempotent_replay,
        status_url=f"/api/v1/jobs/{jid}",
        result_url=f"/api/v1/jobs/{jid}/result",
        pages_url=f"/api/v1/jobs/{jid}/pages",
        pdf_url=f"/api/v1/jobs/{jid}/pdf",
        metadata_url=f"/api/v1/jobs/{jid}/metadata",
    )


def _page_to_schema(page: PageOcrResult, include_boxes: bool = False) -> PageResultSchema:
    words_schema: list[WordBoxSchema] | None = None
    if include_boxes:
        words_schema = [
            WordBoxSchema(
                text=w.text,
                confidence=round(w.confidence, 2),
                bbox_source_px=BBoxPxSchema(
                    left=round(w.bbox_source_px.left, 2),
                    top=round(w.bbox_source_px.top, 2),
                    width=round(w.bbox_source_px.width, 2),
                    height=round(w.bbox_source_px.height, 2),
                ),
                bbox_pdf_pt=(
                    BBoxPtSchema(
                        x=round(w.bbox_pdf_pt.x, 3),
                        y=round(w.bbox_pdf_pt.y, 3),
                        width=round(w.bbox_pdf_pt.width, 3),
                        height=round(w.bbox_pdf_pt.height, 3),
                    )
                    if w.bbox_pdf_pt
                    else None
                ),
                column_index=w.column_index,
                reading_order_index=w.reading_order_index,
            )
            for w in page.words
        ]

    nlp_schema = (
        PageNlpAnalysisSchema.model_validate(page.nlp.to_dict())
        if page.nlp is not None
        else None
    )

    return PageResultSchema(
        page_number=page.page_number,
        text=page.text,
        width_px=page.width_px,
        height_px=page.height_px,
        width_pt=round(page.width_pt, 2),
        height_pt=round(page.height_pt, 2),
        dpi_x=round(page.dpi_x, 2),
        dpi_y=round(page.dpi_y, 2),
        dpi_source=page.dpi_source,
        rotation_deg=page.rotation_deg,
        deskew_angle_deg=round(page.deskew_angle_deg, 3),
        ocr_applied=page.ocr_applied,
        mean_confidence=round(page.mean_confidence, 2),
        column_count=page.column_count,
        processing_duration_ms=page.processing_duration_ms,
        image_preservation_mode=page.image_preservation_mode,
        warnings=page.warnings,
        nlp=nlp_schema,
        words=words_schema,
    )


@router.post(
    "/jobs",
    response_model=JobSubmitResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Submit an asynchronous OCR & PDF/A-2b job",
)
@router.post(
    "/ocr/jobs",
    response_model=JobSubmitResponse,
    status_code=status.HTTP_202_ACCEPTED,
    include_in_schema=True,
    summary="Submit an asynchronous OCR & PDF/A-2b job (alias)",
)
def submit_ocr_job(
    settings: Annotated[Settings, Depends(get_app_settings)],
    repo: Annotated[JobRepository, Depends(get_job_repository)],
    storage: Annotated[LocalJobStorage, Depends(get_job_storage)],
    files: Annotated[
        list[UploadFile],
        File(description="One PDF or one/more images (JPEG, PNG, single/multipage TIFF)."),
    ],
    language: Annotated[str | None, Form(description="Tesseract OCR language code.")] = None,
    force_ocr: Annotated[
        bool | None,
        Form(description="Force rasterization and OCR even if PDF pages already have usable text."),
    ] = None,
    working_dpi: Annotated[
        int | None,
        Form(ge=100, le=600, description="Target resolution for temporary OCR working image."),
    ] = None,
    preserve_source_images: Annotated[
        bool | None,
        Form(description="Preserve original image streams/resolution in output PDF/A."),
    ] = None,
    deskew_working_image: Annotated[
        bool | None,
        Form(description="Deskew temporary OCR working copy and inverse-map word coordinates."),
    ] = None,
    layout_mode: Annotated[
        LayoutMode,
        Form(description="Document layout and reading-order mode."),
    ] = LayoutMode.AUTO,
    pdfa_profile: Annotated[
        PdfAProfile | None,
        Form(description="Target PDF/A conformance profile."),
    ] = None,
    enable_nlp: Annotated[
        bool,
        Form(description="Enable per-page Turkish Topic Assignment, Keyphrases, and NER."),
    ] = True,
    nlp_mode: Annotated[
        NlpMode,
        Form(description="Turkish CPU NLP execution mode ('fast', 'onnx', or 'off')."),
    ] = NlpMode.FAST,
    custom_topics: Annotated[
        str | None,
        Form(description="Optional comma-separated custom Turkish topic labels to score."),
    ] = None,
    callback_url: Annotated[
        str | None,
        Form(description="Receives a signed POST when the job reaches a terminal state."),
    ] = None,
    idempotency_key: Annotated[
        str | None,
        Header(alias="Idempotency-Key", description="Optional client idempotency key."),
    ] = None,
) -> JobSubmitResponse:
    """Validate uploaded files, persist to isolated job storage, and enqueue durable OCR job."""
    if not files:
        raise ValidationError("At least one input file must be uploaded.", code="NO_INPUT_FILES")
    checked_callback_url = validate_callback_url(settings, callback_url)

    job_id = f"job-{uuid.uuid4().hex}"
    upload_streams = [
        (sanitize_filename(f.filename, fallback=f"input_{i:04d}.bin"), f.file)
        for i, f in enumerate(files)
    ]

    saved_paths, content_sha256, total_bytes = storage.save_upload_streams(job_id, upload_streams)

    try:
        # Validate magic bytes of all uploaded files immediately (FR-03, FR-16, FR-17)
        detected_formats = [detect_file_format_by_magic(p) for p in saved_paths]
        if "PDF" in detected_formats and len(saved_paths) > 1:
            raise ValidationError(
                "Cannot combine a PDF with other files in a single submission; "
                "upload either 1 PDF or 1..N images.",
                code="MIXED_PDF_AND_IMAGE_INPUTS",
            )
    except Exception:
        storage.purge_job_artifacts(job_id)
        raise

    # Check idempotency key for duplicate submission prevention (FR-02)
    if idempotency_key:
        existing = repo.find_idempotent_job(idempotency_key, content_sha256)
        if existing is not None:
            storage.purge_job_artifacts(job_id)
            return _build_submit_response(existing, idempotent_replay=True)

    parsed_custom_topics = (
        [t.strip() for t in custom_topics.split(",") if t.strip()]
        if custom_topics
        else []
    )
    options = _build_options(
        settings,
        language=language,
        force_ocr=force_ocr,
        working_dpi=working_dpi,
        preserve_source_images=preserve_source_images,
        deskew_working_image=deskew_working_image,
        layout_mode=layout_mode,
        pdfa_profile=pdfa_profile,
        enable_nlp=enable_nlp,
        nlp_mode=nlp_mode,
        custom_topics=parsed_custom_topics,
    )

    try:
        job = repo.enqueue_job(
            job_id=job_id,
            options=options,
            input_filenames=[name for name, _ in upload_streams],
            input_paths=saved_paths,
            content_sha256=content_sha256,
            idempotency_key=idempotency_key,
            correlation_id=correlation_id_ctx.get(),
            max_retries=settings.job_max_retries,
            max_queue_depth=settings.max_queue_depth,
            callback_url=checked_callback_url,
        )
    except Exception:
        storage.purge_job_artifacts(job_id)
        raise

    logger.info(
        "Accepted new OCR job",
        extra={
            "job_id": job.job_id,
            "file_count": len(saved_paths),
            "total_bytes": total_bytes,
            "formats": detected_formats,
        },
    )
    return _build_submit_response(job, idempotent_replay=False)


@router.post(
    "/jobs/from-cdn",
    response_model=JobSubmitResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Submit an OCR job whose inputs are fetched from the internal CDN",
)
def submit_cdn_ocr_job(
    request: CdnOcrJobRequest,
    settings: Annotated[Settings, Depends(get_app_settings)],
    repo: Annotated[JobRepository, Depends(get_job_repository)],
    idempotency_key: Annotated[
        str | None,
        Header(alias="Idempotency-Key", description="Optional client idempotency key."),
    ] = None,
) -> JobSubmitResponse:
    """Enqueue an OCR job by CDN reference; the worker downloads the files itself.

    Nothing is downloaded here, so the API process stays light and the caller never has
    to stream the document through itself.
    """
    if not (settings.cdn_base_url and settings.cdn_verification_id):
        raise ValidationError(
            "CDN inputs are disabled: CDN_BASE_URL and CDN_VERIFICATION_ID are not configured.",
            code="CDN_NOT_CONFIGURED",
        )
    checked_callback_url = validate_callback_url(settings, request.callback_url)

    refs: list[CdnSourceRef] = []
    for src in request.sources:
        normalized = normalize_cdn_path(src.path)
        filename = src.filename or normalized.rsplit("/", 1)[-1]
        refs.append(CdnSourceRef(path=src.path.strip(), filename=filename))

    # The CDN paths identify the content for idempotency (the bytes are not fetched yet).
    content_sha256 = hashlib.sha256(
        json.dumps([r.path for r in refs], ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    if idempotency_key:
        existing = repo.find_idempotent_job(idempotency_key, content_sha256)
        if existing is not None:
            return _build_submit_response(existing, idempotent_replay=True)

    options = _build_options(
        settings,
        language=request.language,
        force_ocr=request.force_ocr,
        working_dpi=request.working_dpi,
        preserve_source_images=request.preserve_source_images,
        deskew_working_image=request.deskew_working_image,
        layout_mode=request.layout_mode,
        pdfa_profile=request.pdfa_profile,
        enable_nlp=request.enable_nlp,
        nlp_mode=request.nlp_mode,
        custom_topics=[t.strip() for t in request.custom_topics if t.strip()],
    )
    job = repo.enqueue_job(
        job_id=f"job-{uuid.uuid4().hex}",
        options=options,
        input_filenames=[r.filename for r in refs],
        input_paths=[],
        content_sha256=content_sha256,
        idempotency_key=idempotency_key,
        correlation_id=correlation_id_ctx.get(),
        max_retries=settings.job_max_retries,
        max_queue_depth=settings.max_queue_depth,
        job_type=JobType.OCR,
        source_refs=refs,
        callback_url=checked_callback_url,
    )
    logger.info(
        "Accepted new CDN-referenced OCR job",
        extra={"job_id": job.job_id, "source_count": len(refs)},
    )
    return _build_submit_response(job, idempotent_replay=False)


@router.post(
    "/jobs/package",
    response_model=JobSubmitResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Submit a zip package job built from internal CDN files",
)
def submit_package_job(
    request: PackageJobRequest,
    settings: Annotated[Settings, Depends(get_app_settings)],
    storage: Annotated[LocalJobStorage, Depends(get_job_storage)],
    repo: Annotated[JobRepository, Depends(get_job_repository)],
    idempotency_key: Annotated[
        str | None,
        Header(alias="Idempotency-Key", description="Optional client idempotency key."),
    ] = None,
) -> JobSubmitResponse:
    """Enqueue a package job; a package-pool worker fetches the files and writes the parts.

    The entry list is kept in the job directory instead of the job table, so a request
    with tens of thousands of files does not slow down every status query.
    """
    if not (settings.cdn_base_url and settings.cdn_verification_id):
        raise ValidationError(
            "Package jobs are disabled: CDN_BASE_URL and CDN_VERIFICATION_ID are not configured.",
            code="CDN_NOT_CONFIGURED",
        )
    if len(request.entries) > settings.max_package_entries:
        raise ValidationError(
            f"A package may contain at most {settings.max_package_entries} files "
            f"(requested {len(request.entries)}).",
            code="PACKAGE_TOO_MANY_ENTRIES",
        )
    if request.max_part_mb > settings.max_package_part_mb:
        raise ValidationError(
            f"max_part_mb may not exceed {settings.max_package_part_mb}.",
            code="PACKAGE_PART_TOO_LARGE",
        )
    checked_callback_url = validate_callback_url(settings, request.callback_url)

    entries: list[PackageEntry] = []
    for e in request.entries:
        normalize_cdn_path(e.path)
        entries.append(PackageEntry(path=e.path.strip(), name=e.name, ref=e.ref))

    package_options = {
        "max_part_mb": request.max_part_mb,
        "part_name_prefix": request.part_name_prefix,
        "manifest_title": request.manifest_title,
    }
    content_sha256 = hashlib.sha256(
        json.dumps(
            [package_options, [e.to_dict() for e in entries]], ensure_ascii=False
        ).encode("utf-8")
    ).hexdigest()
    if idempotency_key:
        existing = repo.find_idempotent_job(idempotency_key, content_sha256)
        if existing is not None:
            return _build_submit_response(existing, idempotent_replay=True)

    storage.check_storage_capacity()
    job_id = f"job-{uuid.uuid4().hex}"
    paths = storage.get_job_paths(job_id, create=True)
    entries_path = paths.input_dir / PACKAGE_ENTRIES_FILENAME
    write_package_entries(entries_path, entries)
    try:
        job = repo.enqueue_job(
            job_id=job_id,
            options=_build_options(
                settings,
                language=None,
                force_ocr=None,
                working_dpi=None,
                preserve_source_images=None,
                deskew_working_image=None,
                layout_mode=LayoutMode.AUTO,
                pdfa_profile=None,
                enable_nlp=False,
                nlp_mode=NlpMode.OFF,
                custom_topics=[],
            ),
            input_filenames=[PACKAGE_ENTRIES_FILENAME],
            input_paths=[entries_path],
            content_sha256=content_sha256,
            idempotency_key=idempotency_key,
            correlation_id=correlation_id_ctx.get(),
            max_retries=settings.job_max_retries,
            max_queue_depth=settings.max_queue_depth,
            job_type=JobType.PACKAGE,
            callback_url=checked_callback_url,
            extra_metadata={"package_request": package_options},
        )
    except Exception:
        storage.purge_job_artifacts(job_id)
        raise
    logger.info(
        "Accepted new package job",
        extra={"job_id": job.job_id, "entry_count": len(entries)},
    )
    return _build_submit_response(job, idempotent_replay=False)


def _package_result_schema(job: JobEntity) -> PackageResultSchema | None:
    data = job.metadata_json.get("package")
    if job.job_type != JobType.PACKAGE or not data:
        return None
    return PackageResultSchema(
        entry_count=data.get("entry_count", 0),
        ok_count=data.get("ok_count", 0),
        error_count=data.get("error_count", 0),
        total_bytes=data.get("total_bytes", 0),
        parts=[
            PackagePartSchema(
                **p, download_url=f"/api/v1/jobs/{job.job_id}/packages/{p['part_no']}"
            )
            for p in data.get("parts", [])
        ],
        errors=[PackageSkippedFileSchema(**e) for e in data.get("errors", [])],
        errors_truncated=bool(data.get("errors_truncated")),
    )


@router.get(
    "/jobs/{job_id}/packages/{part_no}",
    response_class=FileResponse,
    summary="Download one zip part of a completed package job (supports Range requests)",
)
def download_package_part(
    job_id: str,
    part_no: int,
    storage: Annotated[LocalJobStorage, Depends(get_job_storage)],
    repo: Annotated[JobRepository, Depends(get_job_repository)],
) -> FileResponse:
    storage.validate_job_id(job_id)
    job = repo.get_job(job_id)
    if job.job_type != JobType.PACKAGE:
        raise ValidationError(f"Job {job_id!r} is not a package job.", code="NOT_A_PACKAGE_JOB")
    _assert_job_completed(job)

    parts = (job.metadata_json.get("package") or {}).get("parts", [])
    part = next((p for p in parts if p.get("part_no") == part_no), None)
    if part is None:
        raise JobNotFoundError(f"Package part {part_no} of job {job_id!r} does not exist.")
    path = package_part_path(storage, job_id, part["filename"])
    if not path.exists():
        raise JobExpiredError(f"Package part {part_no} of job {job_id!r} is no longer on disk.")
    return FileResponse(
        path=path,
        media_type="application/zip",
        filename=part["filename"],
        content_disposition_type="attachment",
    )


@router.delete(
    "/jobs/{job_id}",
    response_model=JobDeleteResponse,
    summary="Delete a job's artifacts now (cancels it first if still running)",
)
def delete_job(
    job_id: str,
    settings: Annotated[Settings, Depends(get_app_settings)],
    storage: Annotated[LocalJobStorage, Depends(get_job_storage)],
    repo: Annotated[JobRepository, Depends(get_job_repository)],
) -> JobDeleteResponse:
    """Free disk space before retention would; used when a user deletes an export."""
    storage.validate_job_id(job_id)
    job = repo.get_job(job_id)
    if job.status in (JobStatus.QUEUED, JobStatus.PROCESSING):
        updated = repo.request_cancel(job_id)
        if updated.status == JobStatus.CANCELLED:
            storage.purge_job_artifacts(job_id)
            return JobDeleteResponse(
                job_id=job_id, status=updated.status, message="Job cancelled and artifacts purged."
            )
        return JobDeleteResponse(
            job_id=job_id,
            status=updated.status,
            message="Cancellation requested; the worker purges artifacts when it stops.",
        )
    if job.status != JobStatus.EXPIRED:
        repo.mark_job_expired(job_id, metadata_retention_days=settings.metadata_retention_days)
    storage.purge_job_artifacts(job_id)
    return JobDeleteResponse(job_id=job_id, status=JobStatus.EXPIRED, message="Job artifacts purged.")


def _build_options(
    settings: Settings,
    *,
    language: str | None,
    force_ocr: bool | None,
    working_dpi: int | None,
    preserve_source_images: bool | None,
    deskew_working_image: bool | None,
    layout_mode: LayoutMode,
    pdfa_profile: PdfAProfile | None,
    enable_nlp: bool,
    nlp_mode: NlpMode,
    custom_topics: list[str],
) -> JobSubmissionOptions:
    """Merge per-request OCR options with service defaults."""
    return JobSubmissionOptions(
        language=language or settings.ocr_language,
        force_ocr=settings.ocr_force if force_ocr is None else force_ocr,
        working_dpi=working_dpi or settings.ocr_working_dpi,
        preserve_source_images=(
            settings.preserve_source_images
            if preserve_source_images is None
            else preserve_source_images
        ),
        deskew_working_image=(
            settings.deskew_working_image if deskew_working_image is None else deskew_working_image
        ),
        layout_mode=layout_mode,
        pdfa_profile=pdfa_profile or PdfAProfile(settings.pdf_a_profile),
        enable_nlp=enable_nlp and (nlp_mode != NlpMode.OFF),
        nlp_mode=nlp_mode,
        custom_topics=custom_topics,
    )


@router.get(
    "/jobs/{job_id}",
    response_model=JobStatusResponse,
    summary="Retrieve OCR job lifecycle status and progress",
)
def get_job_status(
    job_id: str,
    storage: Annotated[LocalJobStorage, Depends(get_job_storage)],
    repo: Annotated[JobRepository, Depends(get_job_repository)],
) -> JobStatusResponse:
    storage.validate_job_id(job_id)
    job = repo.get_job(job_id)

    progress = 0.0
    if job.status == JobStatus.COMPLETED:
        progress = 100.0
    elif job.page_count > 0:
        progress = round(min(99.0, (job.completed_pages / job.page_count) * 100.0), 1)

    is_done = job.status == JobStatus.COMPLETED and job.job_type == JobType.OCR
    return JobStatusResponse(
        job_id=job.job_id,
        job_type=job.job_type,
        status=job.status,
        callback_status=job.callback_status,
        callback_attempts=job.callback_attempts,
        callback_last_error=job.callback_last_error,
        page_count=job.page_count,
        completed_pages=job.completed_pages,
        progress_percent=progress,
        attempt_count=job.attempt_count,
        max_retries=job.max_retries,
        cancel_requested=job.cancel_requested,
        error_code=job.error_code,
        error_message=job.error_message,
        retryable=job.retryable,
        pdfa_profile=job.options.pdfa_profile.value,
        pdfa_compliant=job.pdfa_compliant,
        created_at=job.created_at,
        started_at=job.started_at,
        completed_at=job.completed_at,
        expires_at=job.expires_at,
        processing_duration_ms=job.processing_duration_ms,
        pdf_url=f"/api/v1/jobs/{job.job_id}/pdf" if is_done else None,
        result_url=f"/api/v1/jobs/{job.job_id}/result" if is_done else None,
        pages_url=f"/api/v1/jobs/{job.job_id}/pages" if is_done else None,
        metadata_url=f"/api/v1/jobs/{job.job_id}/metadata",
        package=_package_result_schema(job),
    )


def _assert_job_completed(job: JobEntity) -> None:
    if job.status == JobStatus.EXPIRED:
        raise JobExpiredError(
            f"Job {job.job_id!r} artifacts have expired and been cleaned up.",
            details={
                "job_id": job.job_id,
                "status": job.status.value,
                "expired_at": job.metadata_json.get("expired_at"),
            },
        )
    if job.status in (JobStatus.QUEUED, JobStatus.PROCESSING):
        raise JobNotReadyError(
            f"Job {job.job_id!r} is currently {job.status.value!r} "
            f"({job.completed_pages}/{job.page_count} pages completed).",
            details={
                "job_id": job.job_id,
                "status": job.status.value,
                "completed_pages": job.completed_pages,
                "page_count": job.page_count,
            },
        )
    if job.status in (JobStatus.FAILED, JobStatus.CANCELLED):
        raise JobNotReadyError(
            f"Job {job.job_id!r} terminated with status {job.status.value!r}: {job.error_message or 'Cancelled'}",
            retryable=False,
            details={
                "job_id": job.job_id,
                "status": job.status.value,
                "error_code": job.error_code,
            },
        )


@router.get(
    "/jobs/{job_id}/result",
    response_model=JobStructuredResultResponse,
    summary="Retrieve structured OCR output and page text (FR-07)",
)
def get_job_result(
    job_id: str,
    storage: Annotated[LocalJobStorage, Depends(get_job_storage)],
    repo: Annotated[JobRepository, Depends(get_job_repository)],
    include_boxes: Annotated[
        bool,
        Query(description="Include word-level bounding boxes and confidence scores."),
    ] = False,
) -> JobStructuredResultResponse:
    storage.validate_job_id(job_id)
    job = repo.get_job(job_id)
    _assert_job_completed(job)

    pages = repo.get_job_pages(job_id, include_boxes=include_boxes)
    return JobStructuredResultResponse(
        job_id=job.job_id,
        status=job.status,
        page_count=job.page_count,
        pdf_url=f"/api/v1/jobs/{job.job_id}/pdf",
        pdfa_profile=job.options.pdfa_profile,
        pdfa_compliant=bool(job.pdfa_compliant),
        processing_duration_ms=job.processing_duration_ms,
        created_at=job.created_at,
        completed_at=job.completed_at,
        expires_at=job.expires_at,
        document_nlp_summary=job.metadata_json.get("document_nlp_summary"),
        pages=[_page_to_schema(p, include_boxes=include_boxes) for p in pages],
    )


@router.get(
    "/jobs/{job_id}/pages",
    response_model=JobPagesListResponse,
    summary="Retrieve per-page extracted OCR text and optional word geometry",
)
def get_job_pages(
    job_id: str,
    storage: Annotated[LocalJobStorage, Depends(get_job_storage)],
    repo: Annotated[JobRepository, Depends(get_job_repository)],
    page_start: Annotated[int | None, Query(ge=1, description="Start page number (1-based).")] = None,
    page_end: Annotated[int | None, Query(ge=1, description="End page number (inclusive).")] = None,
    include_boxes: Annotated[
        bool,
        Query(description="Include word-level bounding boxes and confidence scores."),
    ] = False,
) -> JobPagesListResponse:
    storage.validate_job_id(job_id)
    job = repo.get_job(job_id)
    _assert_job_completed(job)

    pages = repo.get_job_pages(
        job_id,
        page_start=page_start,
        page_end=page_end,
        include_boxes=include_boxes,
    )
    return JobPagesListResponse(
        job_id=job.job_id,
        status=job.status,
        total_pages=job.page_count,
        returned_pages=len(pages),
        pages=[_page_to_schema(p, include_boxes=include_boxes) for p in pages],
    )


@router.get(
    "/jobs/{job_id}/pdf",
    response_class=FileResponse,
    summary="Download or preview completed searchable PDF/A document",
)
def download_job_pdf(
    job_id: str,
    storage: Annotated[LocalJobStorage, Depends(get_job_storage)],
    repo: Annotated[JobRepository, Depends(get_job_repository)],
    inline: Annotated[
        bool,
        Query(description="If true, serve with Content-Disposition: inline for browser PDF preview."),
    ] = True,
) -> FileResponse:
    storage.validate_job_id(job_id)
    job = repo.get_job(job_id)
    _assert_job_completed(job)

    paths = storage.get_job_paths(job_id, create=False)
    if not paths.output_pdf_path.exists():
        raise JobExpiredError(
            f"Output PDF/A file for job {job_id!r} is no longer available on disk."
        )

    download_name = f"{job_id}_searchable_pdfa.pdf"
    if job.input_filenames:
        stem = job.input_filenames[0].rsplit(".", 1)[0]
        download_name = sanitize_filename(f"{stem}_pdfa.pdf", fallback=download_name)

    return FileResponse(
        path=paths.output_pdf_path,
        media_type="application/pdf",
        filename=download_name,
        content_disposition_type="inline" if inline else "attachment",
    )


@router.get(
    "/jobs/{job_id}/metadata",
    response_model=JobMetadataResponse,
    summary="Retrieve detailed processing metadata, telemetry, and audit events",
)
def get_job_metadata(
    job_id: str,
    storage: Annotated[LocalJobStorage, Depends(get_job_storage)],
    repo: Annotated[JobRepository, Depends(get_job_repository)],
) -> JobMetadataResponse:
    storage.validate_job_id(job_id)
    job = repo.get_job(job_id)
    pages = repo.get_job_pages(job_id, include_boxes=True)

    telemetry = [
        PageTelemetrySchema(
            page_number=p.page_number,
            width_px=p.width_px,
            height_px=p.height_px,
            width_pt=round(p.width_pt, 2),
            height_pt=round(p.height_pt, 2),
            dpi_x=round(p.dpi_x, 2),
            dpi_y=round(p.dpi_y, 2),
            dpi_source=p.dpi_source.value,
            rotation_deg=p.rotation_deg,
            deskew_angle_deg=round(p.deskew_angle_deg, 3),
            ocr_applied=p.ocr_applied,
            mean_confidence=round(p.mean_confidence, 2),
            column_count=p.column_count,
            word_count=len(p.words),
            processing_duration_ms=p.processing_duration_ms,
            image_preservation_mode=p.image_preservation_mode,
            warnings=p.warnings,
        )
        for p in pages
    ]

    return JobMetadataResponse(
        job_id=job.job_id,
        status=job.status,
        options=job.options.to_dict(),
        input_filenames=job.input_filenames,
        content_sha256=job.content_sha256,
        correlation_id=job.correlation_id,
        page_count=job.page_count,
        completed_pages=job.completed_pages,
        attempt_count=job.attempt_count,
        max_retries=job.max_retries,
        worker_id=job.worker_id,
        error_code=job.error_code,
        error_message=job.error_message,
        retryable=job.retryable,
        pdfa_compliant=job.pdfa_compliant,
        pdfa_validator=job.pdfa_validator,
        processing_duration_ms=job.processing_duration_ms,
        created_at=job.created_at,
        started_at=job.started_at,
        completed_at=job.completed_at,
        expires_at=job.expires_at,
        events=list(job.metadata_json.get("events", [])),
        pdfa_validation=job.metadata_json.get("pdfa_validation"),
        document_nlp_summary=job.metadata_json.get("document_nlp_summary"),
        page_telemetry=telemetry,
    )


@router.post(
    "/jobs/{job_id}/cancel",
    response_model=JobCancelResponse,
    summary="Request cancellation of a queued or processing OCR job",
)
def cancel_job(
    job_id: str,
    storage: Annotated[LocalJobStorage, Depends(get_job_storage)],
    repo: Annotated[JobRepository, Depends(get_job_repository)],
) -> JobCancelResponse:
    storage.validate_job_id(job_id)
    updated = repo.request_cancel(job_id)
    if updated.status == JobStatus.CANCELLED:
        storage.purge_job_artifacts(job_id)
        msg = "Job cancelled immediately and temporary artifacts purged."
    else:
        msg = "Cancellation requested; worker will abort after current page."

    return JobCancelResponse(
        job_id=updated.job_id,
        status=updated.status,
        cancel_requested=updated.cancel_requested,
        message=msg,
    )
