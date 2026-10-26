"""Unit tests for configuration validation and CPU thread limits."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from pydantic import ValidationError

from ocr_service.config import Settings


def test_settings_defaults_and_derived_db_url(tmp_path: Path) -> None:
    s = Settings(TEMP_STORAGE_PATH=tmp_path)
    assert s.ocr_language == "tur"
    assert s.pdf_a_profile == "PDF/A-2b"
    assert s.database_url is not None
    assert s.database_url.startswith("sqlite:///")
    assert s.max_upload_bytes == s.max_upload_mb * 1024 * 1024


def test_settings_auth_requires_api_keys(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="AUTH_ENABLED is true"):
        Settings(TEMP_STORAGE_PATH=tmp_path, AUTH_ENABLED=True, API_KEYS=[])


def test_settings_heartbeat_must_be_less_than_lease(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="WORKER_HEARTBEAT_SECONDS"):
        Settings(
            TEMP_STORAGE_PATH=tmp_path,
            WORKER_HEARTBEAT_SECONDS=60,
            WORKER_LEASE_SECONDS=60,
        )


def test_enforce_cpu_thread_limits(tmp_path: Path) -> None:
    s = Settings(TEMP_STORAGE_PATH=tmp_path, OCR_THREADS_PER_WORKER=2)
    s.enforce_cpu_thread_limits()
    assert os.environ["OMP_THREAD_LIMIT"] == "2"
    assert os.environ["OPENBLAS_NUM_THREADS"] == "2"
