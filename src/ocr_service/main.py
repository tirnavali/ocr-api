"""FastAPI application factory, middleware, exception handlers, and lifespan management."""

from __future__ import annotations

import logging
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from ocr_service import __version__
from ocr_service.api.v1.router import v1_router
from ocr_service.config import Settings, get_settings
from ocr_service.exceptions import OcrServiceError
from ocr_service.infrastructure.db.database import DatabaseSessionManager
from ocr_service.infrastructure.db.repository import JobRepository
from ocr_service.infrastructure.metrics.collector import MetricsCollector
from ocr_service.infrastructure.storage.local_storage import LocalJobStorage
from ocr_service.logging_config import configure_logging, correlation_id_ctx
from ocr_service.pipeline.ocr_engine import TesseractOcrEngine

logger = logging.getLogger(__name__)


def create_app(custom_settings: Settings | None = None) -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = custom_settings or get_settings()
    settings.enforce_cpu_thread_limits()
    configure_logging(settings.log_level, settings.log_json)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
        db = DatabaseSessionManager(settings)
        storage = LocalJobStorage(settings)
        repo = JobRepository(db)
        ocr_engine = TesseractOcrEngine(settings)
        metrics = MetricsCollector(repo, storage)

        app.state.settings = settings
        app.state.db = db
        app.state.storage = storage
        app.state.repo = repo
        app.state.ocr_engine = ocr_engine
        app.state.metrics = metrics

        logger.info(
            "OCR API service initialized",
            extra={
                "env": settings.app_env,
                "storage_dir": str(storage.base_dir),
                "ocr_language": settings.ocr_language,
                "pdfa_profile": settings.pdf_a_profile,
            },
        )
        try:
            yield
        finally:
            db.dispose()
            logger.info("OCR API service shut down")

    app = FastAPI(
        title="Archival Turkish OCR & Searchable PDF/A-2b API",
        description=(
            "Production-grade asynchronous OCR API for digitized newspapers, journals, "
            "historical archival documents, multipage TIFFs, and scanned PDFs."
        ),
        version=__version__,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    @app.middleware("http")
    async def correlation_id_middleware(request: Request, call_next):  # type: ignore[no-untyped-def]
        corr_id = request.headers.get("X-Correlation-ID") or f"req-{uuid.uuid4().hex[:12]}"
        token = correlation_id_ctx.set(corr_id)
        try:
            response = await call_next(request)
            response.headers["X-Correlation-ID"] = corr_id
            return response
        finally:
            correlation_id_ctx.reset(token)

    @app.exception_handler(OcrServiceError)
    async def handle_ocr_service_error(
        _request: Request, exc: OcrServiceError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=exc.http_status,
            content={
                "error": {
                    "code": exc.code,
                    "message": exc.message,
                    "retryable": exc.retryable,
                    "correlation_id": correlation_id_ctx.get(),
                    "details": exc.details,
                }
            },
        )

    @app.exception_handler(RequestValidationError)
    async def handle_request_validation_error(
        _request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "code": "REQUEST_VALIDATION_ERROR",
                    "message": "Invalid request parameters or form data.",
                    "retryable": False,
                    "correlation_id": correlation_id_ctx.get(),
                    "details": {"errors": exc.errors()},
                }
            },
        )

    @app.exception_handler(Exception)
    async def handle_unhandled_exception(
        _request: Request, exc: Exception
    ) -> JSONResponse:
        logger.exception("Unhandled API exception")
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "code": "INTERNAL_SERVER_ERROR",
                    "message": "An unexpected internal server error occurred.",
                    "retryable": True,
                    "correlation_id": correlation_id_ctx.get(),
                    "details": {"type": type(exc).__name__},
                }
            },
        )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Correlation-ID", "Content-Disposition"],
    )

    static_ui_path = Path(__file__).resolve().parent / "static" / "index.html"

    @app.get("/", response_class=FileResponse, include_in_schema=False)
    @app.get("/ui", response_class=FileResponse, include_in_schema=False)
    def serve_test_ui() -> FileResponse:
        return FileResponse(static_ui_path, media_type="text/html")

    app.include_router(v1_router)
    return app


def run_api() -> None:
    """CLI entrypoint for running the Uvicorn API server."""
    settings = get_settings()
    uvicorn.run(
        "ocr_service.main:create_app",
        factory=True,
        host=settings.api_host,
        port=settings.api_port,
        log_level=settings.log_level.lower(),
    )
