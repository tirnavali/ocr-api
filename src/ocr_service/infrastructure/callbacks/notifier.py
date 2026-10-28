"""HMAC-signed job completion callbacks delivered from a durable outbox.

Receiver contract
-----------------
``POST <callback_url>`` with a JSON body (see ``build_callback_payload``) and headers:

- ``X-Ocr-Timestamp``: Unix seconds when the request was signed.
- ``X-Ocr-Signature``: ``sha256=<hex>`` of HMAC-SHA256(secret, ``"<timestamp>." + raw_body``).

The receiver must verify the signature over the raw body bytes, reject stale timestamps,
and be idempotent on ``job_id``: a callback is retried until it gets a 2xx response, so
it may arrive more than once.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import time
from typing import Any

import httpx

from ocr_service.config import Settings
from ocr_service.domain.models import JobEntity
from ocr_service.exceptions import ValidationError
from ocr_service.infrastructure.db.repository import JobRepository

logger = logging.getLogger(__name__)

SIGNATURE_HEADER = "X-Ocr-Signature"
TIMESTAMP_HEADER = "X-Ocr-Timestamp"


def validate_callback_url(settings: Settings, callback_url: str | None) -> str | None:
    """Accept only callback URLs under a configured prefix (prevents SSRF via job submission)."""
    if not callback_url:
        return None
    url = callback_url.strip()
    if not settings.callback_allowed_prefixes:
        raise ValidationError(
            "Callbacks are disabled: CALLBACK_ALLOWED_PREFIXES is not configured.",
            code="CALLBACKS_DISABLED",
        )
    if not any(url.startswith(prefix) for prefix in settings.callback_allowed_prefixes):
        raise ValidationError(
            "callback_url is not under an allowed prefix.",
            code="CALLBACK_URL_NOT_ALLOWED",
            details={"callback_url": url},
        )
    return url


def sign_payload(secret: str, timestamp: str, body: bytes) -> str:
    digest = hmac.new(secret.encode("utf-8"), timestamp.encode("ascii") + b"." + body, hashlib.sha256)
    return "sha256=" + digest.hexdigest()


def build_callback_payload(job: JobEntity) -> dict[str, Any]:
    """Terminal job summary; result artifacts are fetched separately via the listed URLs."""
    completed = job.status.value == "completed"
    base = f"/api/v1/jobs/{job.job_id}"
    is_package = job.job_type.value == "package"
    return {
        "job_id": job.job_id,
        "job_type": job.job_type.value,
        "status": job.status.value,
        "correlation_id": job.correlation_id,
        "idempotency_key": job.idempotency_key,
        "page_count": job.page_count,
        "error_code": job.error_code,
        "error_message": job.error_message,
        "completed_at": job.completed_at.isoformat() if job.completed_at else None,
        "expires_at": job.expires_at.isoformat() if job.expires_at else None,
        "status_url": base,
        "result_url": f"{base}/result" if completed and not is_package else None,
        "pdf_url": f"{base}/pdf" if completed and not is_package else None,
        # Package jobs report their parts (and skipped files) inline, so the receiver can
        # record them without a second round trip; parts download from packages_url/{part_no}.
        "package": job.metadata_json.get("package") if is_package else None,
        "packages_url": f"{base}/packages" if completed and is_package else None,
    }


class CallbackNotifier:
    """Deliver due callbacks with retries; delivery state lives in the job table."""

    def __init__(
        self,
        settings: Settings,
        repo: JobRepository,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._settings = settings
        self._repo = repo
        self._transport = transport

    def deliver_due(self, limit: int = 20) -> int:
        """Send every callback that is due now; return how many were delivered."""
        lease_seconds = int(self._settings.callback_timeout_seconds) + 30
        due = self._repo.claim_due_callbacks(limit=limit, lease_seconds=lease_seconds)
        delivered = 0
        for job in due:
            ok, error = self._send(job)
            self._repo.record_callback_result(
                job.job_id,
                delivered=ok,
                error=error,
                max_attempts=self._settings.callback_max_attempts,
            )
            delivered += int(ok)
        return delivered

    def _send(self, job: JobEntity) -> tuple[bool, str | None]:
        if not job.callback_url or not self._settings.callback_hmac_secret:
            return False, "Callback URL or CALLBACK_HMAC_SECRET missing."
        body = json.dumps(build_callback_payload(job), ensure_ascii=False).encode("utf-8")
        timestamp = str(int(time.time()))
        headers = {
            "Content-Type": "application/json; charset=utf-8",
            TIMESTAMP_HEADER: timestamp,
            SIGNATURE_HEADER: sign_payload(self._settings.callback_hmac_secret, timestamp, body),
            "X-Correlation-ID": job.correlation_id,
        }
        try:
            with httpx.Client(
                timeout=self._settings.callback_timeout_seconds,
                verify=self._settings.callback_verify_tls,
                transport=self._transport,
                follow_redirects=False,
            ) as client:
                response = client.post(job.callback_url, content=body, headers=headers)
        except httpx.HTTPError as exc:
            logger.warning(
                "Callback delivery failed", extra={"job_id": job.job_id, "error": str(exc)}
            )
            return False, f"{type(exc).__name__}: {exc}"
        if 200 <= response.status_code < 300:
            logger.info("Callback delivered", extra={"job_id": job.job_id})
            return True, None
        logger.warning(
            "Callback rejected by receiver",
            extra={"job_id": job.job_id, "status_code": response.status_code},
        )
        return False, f"HTTP {response.status_code}: {response.text[:300]}"
