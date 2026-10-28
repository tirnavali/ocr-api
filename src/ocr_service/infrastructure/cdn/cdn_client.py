"""Streaming downloads of job inputs from the internal CDN (TBMM Framework file protocol)."""

from __future__ import annotations

import base64
import logging
import re
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx

from ocr_service.config import Settings
from ocr_service.exceptions import (
    PayloadTooLargeError,
    PermanentProcessingError,
    TransientProcessingError,
    ValidationError,
)

logger = logging.getLogger(__name__)

CHUNK_SIZE = 256 * 1024
# CDN paths are relative ("AOS\\ab\\cd\\file.pdf"); anything that could escape the
# configured base URL (scheme, host, parent segments) is rejected to prevent SSRF.
_SAFE_CDN_PATH_RE = re.compile(r"^[A-Za-z0-9._\-\\/ ]+$")


class CdnSourceNotFoundError(PermanentProcessingError):
    """The referenced file does not exist on the CDN; retrying cannot fix it."""

    code = "CDN_SOURCE_NOT_FOUND"


class CdnFetchError(TransientProcessingError):
    """CDN unreachable or answered with a server error; worth retrying."""

    code = "CDN_FETCH_FAILED"


def normalize_cdn_path(path: str) -> str:
    """Validate a CDN-relative path and return it with forward slashes, no leading slash."""
    candidate = (path or "").strip()
    if not candidate or not _SAFE_CDN_PATH_RE.match(candidate):
        raise ValidationError(f"Invalid CDN path: {path!r}", code="INVALID_CDN_PATH")
    normalized = candidate.replace("\\", "/").lstrip("/")
    segments = normalized.split("/")
    if any(seg in ("", ".", "..") for seg in segments):
        raise ValidationError(f"Invalid CDN path: {path!r}", code="INVALID_CDN_PATH")
    return normalized


def _token_time_header() -> str:
    """Mirror the .NET client: base64 of UTC now + 5 minutes in ISO 's' format."""
    expires = datetime.now(tz=UTC) + timedelta(minutes=5)
    return base64.b64encode(expires.strftime("%Y-%m-%dT%H:%M:%S").encode("utf-8")).decode("ascii")


class InternalCdnClient:
    """Download CDN files straight to disk without buffering them in memory."""

    def __init__(self, settings: Settings, transport: httpx.BaseTransport | None = None) -> None:
        self._settings = settings
        self._transport = transport

    @property
    def configured(self) -> bool:
        return bool(self._settings.cdn_base_url and self._settings.cdn_verification_id)

    def build_url(self, cdn_path: str) -> str:
        base = (self._settings.cdn_base_url or "").rstrip("/") + "/"
        return base + normalize_cdn_path(cdn_path)

    def download_to(
        self,
        cdn_path: str,
        destination: Path,
        *,
        max_bytes: int,
        on_progress: Callable[[int], None] | None = None,
    ) -> int:
        """Stream ``cdn_path`` into ``destination``; return the byte count.

        ``on_progress`` is called with the bytes written so far after every chunk, so a
        long download can renew the worker lease or abort on cancellation by raising.

        A 404 is permanent (the file was never uploaded or was deleted) so the job fails
        immediately instead of burning retries; network errors and 5xx are transient.
        """
        if not self.configured:
            raise PermanentProcessingError(
                "CDN_BASE_URL and CDN_VERIFICATION_ID must be configured for CDN inputs.",
                code="CDN_NOT_CONFIGURED",
            )
        url = self.build_url(cdn_path)
        headers = {
            "UygulamaId": self._settings.cdn_verification_id or "",
            "TokenTime": _token_time_header(),
        }
        tmp_path = destination.with_suffix(destination.suffix + ".part")
        written = 0
        try:
            with httpx.Client(
                timeout=self._settings.cdn_timeout_seconds,
                verify=self._settings.cdn_verify_tls,
                transport=self._transport,
                follow_redirects=False,
            ) as client, client.stream("GET", url, headers=headers) as response:
                if response.status_code == 404:
                    raise CdnSourceNotFoundError(
                        f"CDN file not found: {cdn_path}",
                        details={"cdn_path": cdn_path},
                    )
                if response.status_code >= 400:
                    raise CdnFetchError(
                        f"CDN answered HTTP {response.status_code} for {cdn_path}",
                        details={"cdn_path": cdn_path, "status_code": response.status_code},
                    )
                with tmp_path.open("wb") as fh:
                    for chunk in response.iter_bytes(CHUNK_SIZE):
                        written += len(chunk)
                        if written > max_bytes:
                            raise PayloadTooLargeError(
                                f"CDN file {cdn_path} exceeds the {max_bytes} byte limit.",
                                code="CDN_SOURCE_TOO_LARGE",
                            )
                        fh.write(chunk)
                        if on_progress is not None:
                            on_progress(written)
            if written == 0:
                raise CdnSourceNotFoundError(
                    f"CDN file is empty: {cdn_path}", details={"cdn_path": cdn_path}
                )
            tmp_path.replace(destination)
            return written
        except httpx.HTTPError as exc:
            raise CdnFetchError(
                f"CDN request failed for {cdn_path}: {exc}", details={"cdn_path": cdn_path}
            ) from exc
        finally:
            tmp_path.unlink(missing_ok=True)
