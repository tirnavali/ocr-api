"""Health, readiness, and observability metrics endpoints (FR-08, FR-18)."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, Response, status

from ocr_service import __version__
from ocr_service.api.dependencies import (
    get_app_settings,
    get_job_repository,
    get_job_storage,
    get_metrics_collector,
    get_ocr_engine,
    verify_api_key,
)
from ocr_service.api.schemas import HealthLiveResponse, HealthReadyResponse
from ocr_service.config import Settings
from ocr_service.infrastructure.db.repository import JobRepository
from ocr_service.infrastructure.metrics.collector import MetricsCollector
from ocr_service.infrastructure.storage.local_storage import LocalJobStorage
from ocr_service.pipeline.ocr_engine import TesseractOcrEngine

router = APIRouter(tags=["Health & Observability"])


@router.get(
    "/health/live",
    response_model=HealthLiveResponse,
    summary="Liveness check for container orchestrators",
)
def health_live(
    settings: Annotated[Settings, Depends(get_app_settings)],
) -> HealthLiveResponse:
    return HealthLiveResponse(
        status="ok",
        service=settings.app_name,
        version=__version__,
    )


@router.get(
    "/health/ready",
    response_model=HealthReadyResponse,
    summary="Readiness check verifying DB, storage quota, and Tesseract language packs",
)
def health_ready(
    response: Response,
    settings: Annotated[Settings, Depends(get_app_settings)],
    repo: Annotated[JobRepository, Depends(get_job_repository)],
    storage: Annotated[LocalJobStorage, Depends(get_job_storage)],
    ocr_engine: Annotated[TesseractOcrEngine, Depends(get_ocr_engine)],
) -> HealthReadyResponse:
    db_ok = True
    live_workers = 0
    try:
        stats = repo.get_queue_and_system_stats()
        live_workers = int(stats.get("live_workers", 0))
    except Exception:
        db_ok = False

    storage_ok = True
    storage_details: dict[str, Any] = {}
    try:
        storage_details = storage.check_storage_capacity()
    except Exception as exc:
        storage_ok = False
        storage_details = {"error": str(exc)}

    ocr_ok, ocr_detail = ocr_engine.check_health(required_language=settings.ocr_language)

    overall_ready = db_ok and storage_ok and ocr_ok
    if not overall_ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return HealthReadyResponse(
        status="ready" if overall_ready else "degraded",
        database_ok=db_ok,
        storage_ok=storage_ok,
        ocr_engine_ok=ocr_ok,
        ocr_engine_detail=ocr_detail,
        live_workers=live_workers,
        details={"storage": storage_details},
    )


@router.get(
    "/metrics",
    dependencies=[Depends(verify_api_key)],
    summary="Operational metrics: queue depth, timings, error counters, CPU/RAM",
)
def get_metrics(
    collector: Annotated[MetricsCollector, Depends(get_metrics_collector)],
) -> dict[str, Any]:
    return collector.collect()
