"""Structured exception hierarchy with transient vs. permanent failure classification."""

from __future__ import annotations

from typing import Any


class OcrServiceError(Exception):
    """Base exception for all application errors."""

    code: str = "INTERNAL_ERROR"
    http_status: int = 500
    retryable: bool = False

    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        http_status: int | None = None,
        retryable: bool | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        if code is not None:
            self.code = code
        if http_status is not None:
            self.http_status = http_status
        if retryable is not None:
            self.retryable = retryable
        self.details: dict[str, Any] = details or {}


# --- Client / Input Validation Errors (Permanent, Non-Retryable) ---


class ValidationError(OcrServiceError):
    code = "VALIDATION_ERROR"
    http_status = 400
    retryable = False


class UnsupportedMediaTypeError(ValidationError):
    code = "UNSUPPORTED_MEDIA_TYPE"
    http_status = 415


class PayloadTooLargeError(ValidationError):
    code = "PAYLOAD_TOO_LARGE"
    http_status = 413


class CorruptedDocumentError(ValidationError):
    code = "CORRUPTED_DOCUMENT"
    http_status = 422


class PasswordProtectedPdfError(ValidationError):
    code = "PASSWORD_PROTECTED_PDF"
    http_status = 422


class ImageDimensionsExceededError(ValidationError):
    code = "IMAGE_DIMENSIONS_EXCEEDED"
    http_status = 422


class PageLimitExceededError(ValidationError):
    code = "PAGE_LIMIT_EXCEEDED"
    http_status = 422


# --- Auth / Security Errors ---


class AuthenticationError(OcrServiceError):
    code = "UNAUTHORIZED"
    http_status = 401
    retryable = False


# --- Resource / State Errors ---


class JobNotFoundError(OcrServiceError):
    code = "JOB_NOT_FOUND"
    http_status = 404
    retryable = False


class JobNotReadyError(OcrServiceError):
    code = "JOB_NOT_READY"
    http_status = 409
    retryable = True


class JobExpiredError(OcrServiceError):
    code = "JOB_EXPIRED"
    http_status = 410
    retryable = False


class JobStateConflictError(OcrServiceError):
    code = "JOB_STATE_CONFLICT"
    http_status = 409
    retryable = False


class QueueBackpressureError(OcrServiceError):
    code = "QUEUE_FULL"
    http_status = 503
    retryable = True


class StorageQuotaExceededError(OcrServiceError):
    code = "STORAGE_EXHAUSTED"
    http_status = 507
    retryable = True


# --- Pipeline Processing Errors ---


class TransientProcessingError(OcrServiceError):
    """Temporary error (e.g., subprocess timeout, resource contention) eligible for retry."""

    code = "TRANSIENT_PROCESSING_ERROR"
    http_status = 503
    retryable = True


class PermanentProcessingError(OcrServiceError):
    """Deterministic processing error that should not be retried."""

    code = "PERMANENT_PROCESSING_ERROR"
    http_status = 422
    retryable = False


class OcrEngineExecutionError(TransientProcessingError):
    code = "OCR_ENGINE_FAILURE"


class JobTimeoutError(TransientProcessingError):
    code = "JOB_TIMEOUT"


class JobCancelledException(OcrServiceError):
    code = "JOB_CANCELLED"
    http_status = 409
    retryable = False


class PdfAConformanceError(PermanentProcessingError):
    code = "PDFA_VALIDATION_FAILED"
    http_status = 500
    retryable = False
