"""Centralized application configuration validated at startup."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

from ocr_service.domain.models import JobType

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Operational configuration for API and Worker processes."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Environment & Service
    app_env: Literal["development", "testing", "production"] = Field(
        default="development", alias="APP_ENV"
    )
    app_name: str = Field(default="ocr-api-service", alias="APP_NAME")
    api_host: str = Field(default="0.0.0.0", alias="API_HOST")
    api_port: int = Field(default=8000, ge=1, le=65535, alias="API_PORT")
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO", alias="LOG_LEVEL"
    )
    log_json: bool = Field(default=True, alias="LOG_JSON")

    # Security & Authentication (FR-17)
    auth_enabled: bool = Field(default=False, alias="AUTH_ENABLED")
    # NoDecode: env values are comma-separated, not JSON (an empty API_KEYS= must not crash).
    api_keys: Annotated[list[str], NoDecode] = Field(default_factory=list, alias="API_KEYS")
    max_upload_mb: int = Field(default=500, ge=1, le=4096, alias="MAX_UPLOAD_MB")
    max_image_pixels: int = Field(
        default=175_000_000, ge=1_000_000, le=500_000_000, alias="MAX_IMAGE_PIXELS"
    )
    max_pages_per_job: int = Field(default=1000, ge=1, le=5000, alias="MAX_PAGES_PER_JOB")

    # OCR Engine & Concurrency (FR-04, FR-13, FR-15)
    ocr_language: str = Field(default="tur", alias="OCR_LANGUAGE")
    ocr_worker_count: int = Field(default=4, ge=1, le=32, alias="OCR_WORKER_COUNT")
    ocr_threads_per_worker: int = Field(
        default=1, ge=1, le=8, alias="OCR_THREADS_PER_WORKER"
    )
    ocr_working_dpi: int = Field(default=300, ge=100, le=600, alias="OCR_WORKING_DPI")
    default_assumed_dpi: int = Field(
        default=300, ge=72, le=600, alias="DEFAULT_ASSUMED_DPI"
    )
    ocr_force: bool = Field(default=False, alias="OCR_FORCE")
    tesseract_cmd: str = Field(default="tesseract", alias="TESSERACT_CMD")
    tessdata_dir: Path | None = Field(default=None, alias="TESSDATA_DIR")

    # Job Lifecycle & Queue (FR-02, FR-10)
    job_max_retries: int = Field(default=2, ge=0, le=10, alias="JOB_MAX_RETRIES")
    job_timeout_seconds: int = Field(
        default=1800, ge=10, le=14400, alias="JOB_TIMEOUT_SECONDS"
    )
    page_timeout_seconds: int = Field(
        default=180, ge=5, le=1200, alias="PAGE_TIMEOUT_SECONDS"
    )
    worker_lease_seconds: int = Field(
        default=120, ge=15, le=1800, alias="WORKER_LEASE_SECONDS"
    )
    worker_heartbeat_seconds: int = Field(
        default=15, ge=2, le=300, alias="WORKER_HEARTBEAT_SECONDS"
    )
    worker_poll_interval_seconds: float = Field(
        default=0.5, ge=0.05, le=30.0, alias="WORKER_POLL_INTERVAL_SECONDS"
    )
    max_concurrent_jobs: int = Field(
        default=4, ge=1, le=64, alias="MAX_CONCURRENT_JOBS"
    )
    max_queue_depth: int = Field(default=500, ge=1, le=10000, alias="MAX_QUEUE_DEPTH")

    # Storage & Retention (FR-09)
    temp_storage_path: Path = Field(
        default=PROJECT_ROOT / "data" / "ocr", alias="TEMP_STORAGE_PATH"
    )
    database_url: str | None = Field(default=None, alias="DATABASE_URL")
    result_retention_hours: float = Field(
        default=24.0, gt=0.0, le=8760.0, alias="RESULT_RETENTION_HOURS"
    )
    failed_retention_hours: float = Field(
        default=12.0, gt=0.0, le=8760.0, alias="FAILED_RETENTION_HOURS"
    )
    metadata_retention_days: int = Field(
        default=30, ge=1, le=365, alias="METADATA_RETENTION_DAYS"
    )
    max_disk_usage_gb: float = Field(
        default=50.0, ge=0.01, le=10000.0, alias="MAX_DISK_USAGE_GB"
    )
    min_free_disk_mb: int = Field(default=512, ge=64, le=100000, alias="MIN_FREE_DISK_MB")
    cleanup_interval_seconds: float = Field(
        default=60.0, ge=5.0, le=3600.0, alias="CLEANUP_INTERVAL_SECONDS"
    )

    # Job Types & Per-Type Worker Pools
    # Each worker process only claims the job types listed here, so OCR and packaging
    # run in separate pools with independent concurrency caps.
    worker_job_types: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: [JobType.OCR.value], alias="WORKER_JOB_TYPES"
    )
    max_concurrent_package_jobs: int = Field(
        default=1, ge=1, le=16, alias="MAX_CONCURRENT_PACKAGE_JOBS"
    )
    package_retention_hours: float = Field(
        default=168.0, gt=0.0, le=8760.0, alias="PACKAGE_RETENTION_HOURS"
    )
    package_job_timeout_seconds: int = Field(
        default=21600, ge=60, le=172800, alias="PACKAGE_JOB_TIMEOUT_SECONDS"
    )
    max_package_entries: int = Field(
        default=100_000, ge=1, le=500_000, alias="MAX_PACKAGE_ENTRIES"
    )
    max_package_file_mb: int = Field(
        default=2048, ge=1, le=65536, alias="MAX_PACKAGE_FILE_MB"
    )
    max_package_part_mb: int = Field(
        default=4096, ge=1, le=65536, alias="MAX_PACKAGE_PART_MB"
    )

    # Internal CDN Source Fetching (job inputs referenced by CDN path instead of upload)
    cdn_base_url: str | None = Field(default=None, alias="CDN_BASE_URL")
    cdn_verification_id: str | None = Field(default=None, alias="CDN_VERIFICATION_ID")
    cdn_timeout_seconds: float = Field(
        default=180.0, ge=5.0, le=3600.0, alias="CDN_TIMEOUT_SECONDS"
    )
    cdn_verify_tls: bool = Field(default=True, alias="CDN_VERIFY_TLS")

    # Signed Completion Callbacks
    callback_hmac_secret: str | None = Field(default=None, alias="CALLBACK_HMAC_SECRET")
    callback_allowed_prefixes: Annotated[list[str], NoDecode] = Field(
        default_factory=list, alias="CALLBACK_ALLOWED_PREFIXES"
    )
    callback_timeout_seconds: float = Field(
        default=15.0, ge=1.0, le=120.0, alias="CALLBACK_TIMEOUT_SECONDS"
    )
    callback_max_attempts: int = Field(default=12, ge=1, le=100, alias="CALLBACK_MAX_ATTEMPTS")
    callback_verify_tls: bool = Field(default=True, alias="CALLBACK_VERIFY_TLS")

    # PDF/A & Quality Preservation (FR-05, FR-12, FR-14)
    pdf_a_profile: Literal["PDF/A-2b", "PDF/A-1b"] = Field(
        default="PDF/A-2b", alias="PDF_A_PROFILE"
    )
    preserve_source_images: bool = Field(
        default=True, alias="PRESERVE_SOURCE_IMAGES"
    )
    deskew_working_image: bool = Field(default=True, alias="DESKEW_WORKING_IMAGE")
    verapdf_cmd: str = Field(default="verapdf", alias="VERAPDF_CMD")
    require_verapdf: bool = Field(default=False, alias="REQUIRE_VERAPDF")

    # Bundled Resources
    unicode_font_path: Path = Field(
        default=PROJECT_ROOT / "resources" / "fonts" / "ArchiveUnicode-Regular.ttf",
        alias="UNICODE_FONT_PATH",
    )
    srgb_icc_path: Path = Field(
        default=PROJECT_ROOT / "resources" / "icc" / "sRGB.icc",
        alias="SRGB_ICC_PATH",
    )

    @field_validator("api_keys", mode="before")
    @classmethod
    def parse_api_keys(cls, v: str | list[str] | None) -> list[str]:
        if v is None:
            return []
        if isinstance(v, str):
            return [k.strip() for k in v.split(",") if k.strip()]
        return v

    @field_validator("worker_job_types", "callback_allowed_prefixes", mode="before")
    @classmethod
    def parse_csv_list(cls, v: str | list[str] | None) -> list[str]:
        if v is None:
            return []
        if isinstance(v, str):
            return [k.strip() for k in v.split(",") if k.strip()]
        return v

    @model_validator(mode="after")
    def validate_and_derive(self) -> Settings:
        if self.auth_enabled and not self.api_keys:
            raise ValueError("AUTH_ENABLED is true, but no API_KEYS were configured.")
        known_job_types = {t.value for t in JobType}
        unknown = [t for t in self.worker_job_types if t not in known_job_types]
        if unknown or not self.worker_job_types:
            raise ValueError(
                f"WORKER_JOB_TYPES must be a non-empty subset of {sorted(known_job_types)}; "
                f"got {self.worker_job_types}."
            )
        if self.callback_allowed_prefixes and not self.callback_hmac_secret:
            raise ValueError(
                "CALLBACK_ALLOWED_PREFIXES is set, but CALLBACK_HMAC_SECRET is missing."
            )
        if self.worker_heartbeat_seconds >= self.worker_lease_seconds:
            raise ValueError(
                "WORKER_HEARTBEAT_SECONDS must be strictly less than WORKER_LEASE_SECONDS."
            )
        if self.database_url is None:
            db_file = self.temp_storage_path / "ocr_state.sqlite3"
            self.database_url = f"sqlite:///{db_file.resolve()}"
        return self

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @property
    def jobs_storage_dir(self) -> Path:
        return self.temp_storage_path / "jobs"

    def max_concurrent_jobs_for(self, job_type: str) -> int:
        """Concurrency cap per job type; each type is limited independently."""
        if job_type == JobType.PACKAGE.value:
            return self.max_concurrent_package_jobs
        return self.max_concurrent_jobs

    def job_timeout_for(self, job_type: str) -> int:
        """Wall-clock limit of one job attempt; packaging thousands of files takes hours."""
        if job_type == JobType.PACKAGE.value:
            return self.package_job_timeout_seconds
        return self.job_timeout_seconds

    def retention_hours_for(self, job_type: str) -> float:
        """How long completed artifacts of a job type are kept before expiry."""
        if job_type == JobType.PACKAGE.value:
            return self.package_retention_hours
        return self.result_retention_hours

    def enforce_cpu_thread_limits(self) -> None:
        """Configure environment variables to prevent Tesseract/OpenCV CPU oversubscription."""
        threads_str = str(self.ocr_threads_per_worker)
        os.environ["OMP_THREAD_LIMIT"] = threads_str
        os.environ["OMP_NUM_THREADS"] = threads_str
        os.environ["OPENBLAS_NUM_THREADS"] = threads_str
        os.environ["MKL_NUM_THREADS"] = threads_str
        os.environ["VECLIB_MAXIMUM_THREADS"] = threads_str
        os.environ["NUMEXPR_NUM_THREADS"] = threads_str


def get_settings() -> Settings:
    """Return a validated Settings instance."""
    return Settings()
