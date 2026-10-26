"""API v1 main router."""

from __future__ import annotations

from fastapi import APIRouter

from ocr_service.api.v1.health import router as health_router
from ocr_service.api.v1.jobs import router as jobs_router

v1_router = APIRouter(prefix="/api/v1")
v1_router.include_router(jobs_router)
v1_router.include_router(health_router)
