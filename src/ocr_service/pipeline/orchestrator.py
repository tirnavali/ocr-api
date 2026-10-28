"""End-to-end page-by-page OCR and PDF/A-2b pipeline orchestrator with checkpointing."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from ocr_service.config import Settings
from ocr_service.domain.models import (
    JobEntity,
    NlpMode,
    PageOcrResult,
    PageSourceDescriptor,
)
from ocr_service.domain.ports import (
    OcrEnginePort,
    PdfAValidatorPort,
    TurkishNlpAnalyzerPort,
)
from ocr_service.exceptions import (
    JobCancelledException,
    JobTimeoutError,
    OcrServiceError,
    PdfAConformanceError,
    ValidationError,
)
from ocr_service.infrastructure.cdn.cdn_client import InternalCdnClient
from ocr_service.infrastructure.db.repository import JobRepository
from ocr_service.infrastructure.storage.local_storage import LocalJobStorage, sanitize_filename
from ocr_service.logging_config import correlation_id_ctx, job_id_ctx
from ocr_service.pipeline.geometry import (
    source_bbox_to_pdf_points,
    working_bbox_to_source_bbox,
)
from ocr_service.pipeline.inspector import DocumentInspector, detect_file_format_by_magic
from ocr_service.pipeline.layout import analyze_layout_and_reading_order
from ocr_service.pipeline.ocr_engine import TesseractOcrEngine
from ocr_service.pipeline.pdfa_generator import PdfAGenerator
from ocr_service.pipeline.pdfa_validator import DualPdfAValidator
from ocr_service.pipeline.preprocessor import WorkingImagePreprocessor
from ocr_service.pipeline.turkish_nlp import TurkishNlpAnalyzer

logger = logging.getLogger(__name__)


class OcrPipelineOrchestrator:
    """Coordinate document inspection, page-by-page OCR, PDF/A generation, and validation."""

    def __init__(
        self,
        settings: Settings,
        repo: JobRepository,
        storage: LocalJobStorage,
        ocr_engine: OcrEnginePort | None = None,
        pdfa_validator: PdfAValidatorPort | None = None,
        nlp_analyzer: TurkishNlpAnalyzerPort | None = None,
        cdn_client: InternalCdnClient | None = None,
    ) -> None:
        self._settings = settings
        self._repo = repo
        self._storage = storage
        self._cdn_client = cdn_client or InternalCdnClient(settings)
        self._inspector = DocumentInspector(settings)
        self._preprocessor = WorkingImagePreprocessor()
        self._ocr_engine = ocr_engine or TesseractOcrEngine(settings)
        self._pdfa_generator = PdfAGenerator(settings)
        self._pdfa_validator = pdfa_validator or DualPdfAValidator(settings)
        self._nlp_analyzer = nlp_analyzer or TurkishNlpAnalyzer()

    def execute_job(self, job: JobEntity, worker_id: str) -> JobEntity:
        """Execute a claimed job with page-level checkpointing, timeout, and cancellation checks."""
        token_job = job_id_ctx.set(job.job_id)
        token_corr = correlation_id_ctx.set(job.correlation_id)
        job_start_ns = time.perf_counter_ns()

        try:
            paths = self._storage.get_job_paths(job.job_id, create=True)
            input_paths = self._repo.get_job_input_paths(job.job_id)
            if job.source_refs and not (input_paths and all(p.exists() for p in input_paths)):
                input_paths = self._fetch_cdn_inputs(job, paths.input_dir, worker_id, job_start_ns)

            logger.info(
                "Starting OCR job execution",
                extra={
                    "worker_id": worker_id,
                    "input_count": len(input_paths),
                    "attempt": job.attempt_count,
                    "language": job.options.language,
                    "pdfa_profile": job.options.pdfa_profile.value,
                },
            )

            descriptors = self._inspector.inspect_inputs(
                input_paths=input_paths,
                work_dir=paths.work_dir,
                force_ocr=job.options.force_ocr,
                working_dpi=job.options.working_dpi,
            )
            total_pages = len(descriptors)
            self._repo.set_job_page_count(job.job_id, total_pages)

            # Check for already checkpointed pages (crash recovery resume support)
            existing_pages = {
                p.page_number: p
                for p in self._repo.get_job_pages(job.job_id, include_boxes=True)
            }

            page_results: list[PageOcrResult] = []
            for desc in descriptors:
                self._check_timeout_and_cancellation(job.job_id, worker_id, job_start_ns)

                if desc.page_number in existing_pages:
                    logger.info(
                        "Reusing checkpointed page result from prior attempt",
                        extra={"page_number": desc.page_number, "total_pages": total_pages},
                    )
                    cached_page = existing_pages[desc.page_number]
                    if (
                        cached_page.nlp is None
                        and job.options.enable_nlp
                        and job.options.nlp_mode != NlpMode.OFF
                    ):
                        cached_page.nlp = self._nlp_analyzer.analyze_page(
                            cached_page.text,
                            cached_page.words,
                            page_height_px=cached_page.height_px,
                            mode=job.options.nlp_mode,
                            custom_topics=job.options.custom_topics,
                        )
                    page_results.append(cached_page)
                    continue

                page_res = self._process_single_page(job, desc, paths.work_dir)
                page_results.append(page_res)

                cancel_flag = self._repo.save_page_checkpoint(
                    job_id=job.job_id,
                    worker_id=worker_id,
                    page_result=page_res,
                    lease_seconds=self._settings.worker_lease_seconds,
                )
                if cancel_flag:
                    raise JobCancelledException(f"Job {job.job_id} cancelled by client request.")

            # Compose final searchable PDF/A document
            self._check_timeout_and_cancellation(job.job_id, worker_id, job_start_ns)
            doc_title = (
                job.input_filenames[0] if job.input_filenames else f"OCR Document {job.job_id}"
            )
            self._pdfa_generator.generate_pdfa(
                descriptors=descriptors,
                page_results=page_results,
                output_pdf_path=paths.output_pdf_path,
                profile=job.options.pdfa_profile,
                title=doc_title,
            )

            # Persist updated image_preservation_mode and warnings from PDF/A composition
            for page_res in page_results:
                self._repo.save_page_checkpoint(
                    job_id=job.job_id,
                    worker_id=worker_id,
                    page_result=page_res,
                    lease_seconds=self._settings.worker_lease_seconds,
                )

            # Validate PDF/A conformance (FR-05)
            validation_report = self._pdfa_validator.validate(
                paths.output_pdf_path, profile=job.options.pdfa_profile
            )
            if not validation_report.compliant:
                raise PdfAConformanceError(
                    f"Generated PDF failed {job.options.pdfa_profile.value} validation: "
                    f"{'; '.join(validation_report.errors)}",
                    details=validation_report.to_dict(),
                )

            total_duration_ms = int((time.perf_counter_ns() - job_start_ns) / 1_000_000)
            document_nlp_summary = (
                self._nlp_analyzer.summarize_document(page_results)
                if job.options.enable_nlp and job.options.nlp_mode != NlpMode.OFF
                else None
            )

            # Write structured JSON output summary to disk and clean temporary working directory
            result_summary = {
                "job_id": job.job_id,
                "status": "completed",
                "page_count": total_pages,
                "pdfa_profile": job.options.pdfa_profile.value,
                "pdfa_validation": validation_report.to_dict(),
                "processing_duration_ms": total_duration_ms,
                "document_nlp_summary": document_nlp_summary,
                "pages": [p.to_dict(include_boxes=True) for p in page_results],
            }
            paths.output_result_json_path.write_text(
                json.dumps(result_summary, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            self._storage.cleanup_working_dir(job.job_id)

            completed_job = self._repo.mark_completed(
                job_id=job.job_id,
                page_count=total_pages,
                processing_duration_ms=total_duration_ms,
                retention_hours=self._settings.retention_hours_for(job.job_type.value),
                pdfa_compliant=validation_report.compliant,
                pdfa_validator=validation_report.validator_used,
                extra_metadata={
                    "pdfa_validation": validation_report.to_dict(),
                    "output_pdf_size_bytes": paths.output_pdf_path.stat().st_size,
                    "document_nlp_summary": document_nlp_summary,
                },
            )
            logger.info(
                "Completed OCR job successfully",
                extra={
                    "page_count": total_pages,
                    "duration_ms": total_duration_ms,
                    "pdfa_validator": validation_report.validator_used,
                },
            )
            return completed_job

        except JobCancelledException:
            logger.info("Job execution cancelled", extra={"worker_id": worker_id})
            self._storage.purge_job_artifacts(job.job_id)
            return self._repo.finalize_cancelled_job(job.job_id)

        except OcrServiceError as exc:
            logger.warning(
                "Job failed with structured error",
                extra={
                    "error_code": exc.code,
                    "retryable": exc.retryable,
                    "error_message": exc.message,
                },
            )
            updated = self._repo.mark_failed_or_retry(
                job_id=job.job_id,
                error_code=exc.code,
                error_message=exc.message,
                retryable=exc.retryable,
                failed_retention_hours=self._settings.failed_retention_hours,
            )
            if updated.status.value == "failed":
                self._storage.cleanup_working_dir(job.job_id)
            return updated

        except Exception as exc:
            logger.exception("Unexpected error during job execution")
            updated = self._repo.mark_failed_or_retry(
                job_id=job.job_id,
                error_code="INTERNAL_WORKER_ERROR",
                error_message=str(exc),
                retryable=True,
                failed_retention_hours=self._settings.failed_retention_hours,
            )
            if updated.status.value == "failed":
                self._storage.cleanup_working_dir(job.job_id)
            return updated

        finally:
            job_id_ctx.reset(token_job)
            correlation_id_ctx.reset(token_corr)

    def _process_single_page(
        self,
        job: JobEntity,
        descriptor: PageSourceDescriptor,
        work_dir: Path,
    ) -> PageOcrResult:
        page_start_ns = time.perf_counter_ns()

        # Fast path: Input PDF page already has usable Unicode text and force_ocr is False (FR-03)
        if descriptor.has_usable_text and not job.options.force_ocr:
            nlp_result = (
                self._nlp_analyzer.analyze_page(
                    descriptor.existing_text,
                    [],
                    page_height_px=descriptor.height_px,
                    mode=job.options.nlp_mode,
                    custom_topics=job.options.custom_topics,
                )
                if job.options.enable_nlp and job.options.nlp_mode != NlpMode.OFF
                else None
            )
            duration_ms = int((time.perf_counter_ns() - page_start_ns) / 1_000_000)
            return PageOcrResult(
                page_number=descriptor.page_number,
                width_px=descriptor.width_px,
                height_px=descriptor.height_px,
                width_pt=descriptor.width_pt,
                height_pt=descriptor.height_pt,
                dpi_x=descriptor.dpi_x,
                dpi_y=descriptor.dpi_y,
                dpi_source=descriptor.dpi_source,
                rotation_deg=descriptor.rotation_deg,
                deskew_angle_deg=0.0,
                ocr_applied=False,
                mean_confidence=100.0,
                column_count=1,
                text=descriptor.existing_text,
                words=[],
                processing_duration_ms=duration_ms,
                image_preservation_mode="pdf_native_preserved",
                nlp=nlp_result,
            )

        # Prepare initial working image (Track B)
        prepared = self._preprocessor.prepare_working_page(
            descriptor=descriptor,
            work_dir=work_dir,
            target_working_dpi=job.options.working_dpi,
            enable_deskew=job.options.deskew_working_image,
            cardinal_rotation_cw=0,
        )

        try:
            ocr_out = self._ocr_engine.recognize_page(
                prepared.working_image_path,
                language=job.options.language,
                dpi=prepared.working_dpi,
                psm=3,
                timeout_seconds=self._settings.page_timeout_seconds,
            )

            # If confidence is very low or few words found, check if the page is rotated 90/180/270
            cardinal_rot_applied = 0
            if len(ocr_out.words) < 3 or ocr_out.mean_confidence < 45.0:
                osd_deg = self._ocr_engine.detect_orientation(prepared.working_image_path)
                if osd_deg in (90, 180, 270):
                    # Rotate working image to upright orientation (360 - osd_deg clockwise)
                    correction_cw = (360 - osd_deg) % 360
                    prepared_rot = self._preprocessor.prepare_working_page(
                        descriptor=descriptor,
                        work_dir=work_dir,
                        target_working_dpi=job.options.working_dpi,
                        enable_deskew=job.options.deskew_working_image,
                        cardinal_rotation_cw=correction_cw,
                    )
                    ocr_rot = self._ocr_engine.recognize_page(
                        prepared_rot.working_image_path,
                        language=job.options.language,
                        dpi=prepared_rot.working_dpi,
                        psm=3,
                        timeout_seconds=self._settings.page_timeout_seconds,
                    )
                    if len(ocr_rot.words) > len(ocr_out.words) or (
                        ocr_rot.mean_confidence > ocr_out.mean_confidence + 10.0
                    ):
                        prepared = prepared_rot
                        ocr_out = ocr_rot
                        cardinal_rot_applied = correction_cw

            # Transform every word's bounding box: Working Px -> Source Px -> PDF User Space Pt
            for word in ocr_out.words:
                src_bbox = working_bbox_to_source_bbox(
                    word.bbox_working_px,
                    prepared.transform,
                )
                word.bbox_source_px = src_bbox
                pdf_bbox, baseline_y = source_bbox_to_pdf_points(
                    src_bbox,
                    source_width_px=descriptor.width_px,
                    source_height_px=descriptor.height_px,
                    page_width_pt=descriptor.width_pt,
                    page_height_pt=descriptor.height_pt,
                )
                word.bbox_pdf_pt = pdf_bbox
                word.baseline_pdf_y = baseline_y

            # Perform multi-column layout & logical reading-order reconstruction (FR-11)
            layout_res = analyze_layout_and_reading_order(
                ocr_out.words,
                page_width_px=descriptor.width_px,
                page_height_px=descriptor.height_px,
                layout_mode=job.options.layout_mode,
            )

            nlp_result = (
                self._nlp_analyzer.analyze_page(
                    layout_res.logical_text,
                    layout_res.ordered_words,
                    page_height_px=descriptor.height_px,
                    mode=job.options.nlp_mode,
                    custom_topics=job.options.custom_topics,
                )
                if job.options.enable_nlp and job.options.nlp_mode != NlpMode.OFF
                else None
            )

            duration_ms = int((time.perf_counter_ns() - page_start_ns) / 1_000_000)
            return PageOcrResult(
                page_number=descriptor.page_number,
                width_px=descriptor.width_px,
                height_px=descriptor.height_px,
                width_pt=descriptor.width_pt,
                height_pt=descriptor.height_pt,
                dpi_x=descriptor.dpi_x,
                dpi_y=descriptor.dpi_y,
                dpi_source=descriptor.dpi_source,
                rotation_deg=(descriptor.rotation_deg + cardinal_rot_applied) % 360,
                deskew_angle_deg=prepared.transform.deskew_angle_deg,
                ocr_applied=True,
                mean_confidence=ocr_out.mean_confidence,
                column_count=layout_res.column_count,
                text=layout_res.logical_text,
                words=layout_res.ordered_words,
                processing_duration_ms=duration_ms,
                nlp=nlp_result,
            )
        finally:
            if prepared.working_image_path.exists():
                prepared.working_image_path.unlink(missing_ok=True)

    def _fetch_cdn_inputs(
        self,
        job: JobEntity,
        input_dir: Path,
        worker_id: str,
        job_start_ns: int,
    ) -> list[Path]:
        """Download CDN-referenced inputs into the job's input dir and apply upload validation.

        Downloads are re-done on a retry if any file is missing, so a crash mid-download
        never leaves a job with a truncated input.
        """
        self._storage.check_storage_capacity()
        input_paths: list[Path] = []
        for index, ref in enumerate(job.source_refs):
            self._check_timeout_and_cancellation(job.job_id, worker_id, job_start_ns)
            dest = input_dir / f"{index:04d}_{sanitize_filename(ref.filename)}"
            size = self._cdn_client.download_to(
                ref.path, dest, max_bytes=self._settings.max_upload_bytes
            )
            logger.info(
                "Fetched CDN input", extra={"cdn_path": ref.path, "size_bytes": size}
            )
            input_paths.append(dest)

        detected_formats = [detect_file_format_by_magic(p) for p in input_paths]
        if "PDF" in detected_formats and len(input_paths) > 1:
            raise ValidationError(
                "Cannot combine a PDF with other files in a single job; "
                "reference either 1 PDF or 1..N images.",
                code="MIXED_PDF_AND_IMAGE_INPUTS",
            )
        self._repo.set_job_input_paths(job.job_id, input_paths)
        return input_paths

    def _check_timeout_and_cancellation(
        self,
        job_id: str,
        worker_id: str,
        job_start_ns: int,
    ) -> None:
        elapsed_s = (time.perf_counter_ns() - job_start_ns) / 1_000_000_000.0
        if elapsed_s > self._settings.job_timeout_seconds:
            raise JobTimeoutError(
                f"Job {job_id} exceeded timeout of {self._settings.job_timeout_seconds} seconds."
            )
        if self._repo.renew_job_lease(
            job_id, worker_id, lease_seconds=self._settings.worker_lease_seconds
        ):
            raise JobCancelledException(f"Job {job_id} cancelled by client request.")
