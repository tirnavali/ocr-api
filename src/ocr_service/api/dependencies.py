"""FastAPI dependency injection providers and API authentication (FR-17)."""

from __future__ import annotations

import secrets
from typing import Annotated

from fastapi import Depends, Header, Request

from ocr_service.config import Settings
from ocr_service.exceptions import AuthenticationError
from ocr_service.infrastructure.db.repository import JobRepository
from ocr_service.infrastructure.metrics.collector import MetricsCollector
from ocr_service.infrastructure.storage.local_storage import LocalJobStorage
from ocr_service.pipeline.ocr_engine import TesseractOcrEngine


def get_app_settings(request: Request) -> Settings:
    return request.app.state.settings  # type: ignore[no-any-return]


def get_job_repository(request: Request) -> JobRepository:
    return request.app.state.repo  # type: ignore[no-any-return]


def get_job_storage(request: Request) -> LocalJobStorage:
    return request.app.state.storage  # type: ignore[no-any-return]


def get_metrics_collector(request: Request) -> MetricsCollector:
    return request.app.state.metrics  # type: ignore[no-any-return]


def get_ocr_engine(request: Request) -> TesseractOcrEngine:
    return request.app.state.ocr_engine  # type: ignore[no-any-return]


def verify_api_key(
    settings: Annotated[Settings, Depends(get_app_settings)],
    x_api_key: Annotated[str | None, Header(alias="X-API-Key")] = None,
    authorization: Annotated[str | None, Header(alias="Authorization")] = None,
) -> None:
    """Enforce API key authentication when AUTH_ENABLED=true (FR-17)."""
    if not settings.auth_enabled:
        return

    token = x_api_key
    if not token and authorization:
        scheme, _, param = authorization.partition(" ")
        if scheme.lower() == "bearer" and param:
            token = param.strip()

    if not token:
        raise AuthenticationError(
            "Missing API key. Provide 'X-API-Key' or 'Authorization: Bearer <key>' header."
        )

    matched = any(secrets.compare_digest(token, valid_key) for valid_key in settings.api_keys)
    if not matched:
        raise AuthenticationError("Invalid API key.")
