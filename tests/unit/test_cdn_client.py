"""Unit tests for streaming CDN input downloads and CDN path validation."""

from __future__ import annotations

import base64
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest

from ocr_service.config import Settings
from ocr_service.exceptions import PayloadTooLargeError, ValidationError
from ocr_service.infrastructure.cdn.cdn_client import (
    CdnFetchError,
    CdnSourceNotFoundError,
    InternalCdnClient,
    normalize_cdn_path,
)


@pytest.fixture()
def cdn_settings(tmp_path: Path) -> Settings:
    return Settings(
        TEMP_STORAGE_PATH=tmp_path,
        CDN_BASE_URL="http://cdn.test/File/",
        CDN_VERIFICATION_ID="verif-123",
    )


def test_normalize_cdn_path_accepts_windows_style_paths() -> None:
    assert normalize_cdn_path("AOS\\ab\\cd\\belge.pdf") == "AOS/ab/cd/belge.pdf"
    assert normalize_cdn_path("/AOS/ab/belge 1.pdf") == "AOS/ab/belge 1.pdf"


@pytest.mark.parametrize(
    "bad",
    ["", "../etc/passwd", "AOS\\..\\secret", "http://evil/x.pdf", "AOS//x.pdf", "a?b=c"],
)
def test_normalize_cdn_path_rejects_escapes(bad: str) -> None:
    with pytest.raises(ValidationError):
        normalize_cdn_path(bad)


def test_download_streams_file_with_framework_headers(cdn_settings: Settings, tmp_path: Path) -> None:
    seen: dict[str, httpx.Request] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["req"] = request
        return httpx.Response(200, content=b"%PDF-1.7 fake")

    client = InternalCdnClient(cdn_settings, transport=httpx.MockTransport(handler))
    dest = tmp_path / "out.pdf"
    size = client.download_to("AOS\\ab\\belge.pdf", dest, max_bytes=1024)

    assert size == len(b"%PDF-1.7 fake")
    assert dest.read_bytes() == b"%PDF-1.7 fake"
    req = seen["req"]
    assert str(req.url) == "http://cdn.test/File/AOS/ab/belge.pdf"
    assert req.headers["UygulamaId"] == "verif-123"
    token_time = base64.b64decode(req.headers["TokenTime"]).decode()
    expires = datetime.strptime(token_time, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=UTC)
    assert expires > datetime.now(tz=UTC)
    assert not dest.with_suffix(".pdf.part").exists()


def test_missing_file_is_permanent_failure(cdn_settings: Settings, tmp_path: Path) -> None:
    client = InternalCdnClient(
        cdn_settings, transport=httpx.MockTransport(lambda r: httpx.Response(404))
    )
    with pytest.raises(CdnSourceNotFoundError) as exc_info:
        client.download_to("AOS\\x.pdf", tmp_path / "x.pdf", max_bytes=1024)
    assert exc_info.value.retryable is False
    assert not (tmp_path / "x.pdf").exists()


def test_server_error_and_network_error_are_transient(cdn_settings: Settings, tmp_path: Path) -> None:
    client = InternalCdnClient(
        cdn_settings, transport=httpx.MockTransport(lambda r: httpx.Response(503))
    )
    with pytest.raises(CdnFetchError) as exc_info:
        client.download_to("AOS\\x.pdf", tmp_path / "x.pdf", max_bytes=1024)
    assert exc_info.value.retryable is True

    def boom(_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    client = InternalCdnClient(cdn_settings, transport=httpx.MockTransport(boom))
    with pytest.raises(CdnFetchError):
        client.download_to("AOS\\x.pdf", tmp_path / "x.pdf", max_bytes=1024)


def test_oversized_file_is_rejected_without_leaving_partial(cdn_settings: Settings, tmp_path: Path) -> None:
    client = InternalCdnClient(
        cdn_settings, transport=httpx.MockTransport(lambda r: httpx.Response(200, content=b"x" * 2048))
    )
    with pytest.raises(PayloadTooLargeError):
        client.download_to("AOS\\big.pdf", tmp_path / "big.pdf", max_bytes=1024)
    assert list(tmp_path.glob("big.pdf*")) == []
