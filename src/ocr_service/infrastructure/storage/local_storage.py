"""Job-isolated local filesystem storage with path traversal prevention and quota checks."""

from __future__ import annotations

import hashlib
import logging
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from ocr_service.config import Settings
from ocr_service.exceptions import PayloadTooLargeError, StorageQuotaExceededError, ValidationError

logger = logging.getLogger(__name__)

SAFE_FILENAME_RE = re.compile(r"[^A-Za-z0-9._-]")
SAFE_JOB_ID_RE = re.compile(r"^[A-Za-z0-9_-]{8,64}$")
CHUNK_SIZE = 64 * 1024  # 64 KiB streaming chunks


@dataclass(frozen=True, slots=True)
class JobPaths:
    """Isolated directory layout for a single OCR job."""

    job_dir: Path
    input_dir: Path
    work_dir: Path
    output_dir: Path
    output_pdf_path: Path
    output_result_json_path: Path


def sanitize_filename(filename: str | None, fallback: str = "upload.bin") -> str:
    """Prevent path traversal and normalize unsafe filenames (FR-17)."""
    if not filename:
        return fallback
    # Strip any directory components (POSIX and Windows)
    base = filename.replace("\\", "/").split("/")[-1].strip()
    base = base.lstrip(".")
    if not base:
        return fallback
    cleaned = SAFE_FILENAME_RE.sub("_", base)
    if len(cleaned) > 120:
        stem, dot, ext = cleaned.rpartition(".")
        if dot:
            cleaned = f"{stem[:100]}.{ext[:15]}"
        else:
            cleaned = cleaned[:120]
    return cleaned or fallback


class LocalJobStorage:
    """Manage job-isolated files, disk usage quotas, and safe artifact cleanup."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._base_dir = settings.temp_storage_path.resolve()
        self._jobs_dir = (self._base_dir / "jobs").resolve()
        self._jobs_dir.mkdir(parents=True, exist_ok=True)

    @property
    def base_dir(self) -> Path:
        return self._base_dir

    @property
    def jobs_dir(self) -> Path:
        return self._jobs_dir

    def validate_job_id(self, job_id: str) -> str:
        if not SAFE_JOB_ID_RE.match(job_id):
            raise ValidationError(
                f"Invalid job identifier format: {job_id!r}",
                code="INVALID_JOB_ID",
            )
        return job_id

    def get_job_paths(self, job_id: str, create: bool = False) -> JobPaths:
        self.validate_job_id(job_id)
        job_dir = (self._jobs_dir / job_id).resolve()
        if not str(job_dir).startswith(str(self._jobs_dir)):
            raise ValidationError("Path traversal detected in job_id.", code="PATH_TRAVERSAL")

        input_dir = job_dir / "input"
        work_dir = job_dir / "work"
        output_dir = job_dir / "output"
        if create:
            input_dir.mkdir(parents=True, exist_ok=True)
            work_dir.mkdir(parents=True, exist_ok=True)
            output_dir.mkdir(parents=True, exist_ok=True)

        return JobPaths(
            job_dir=job_dir,
            input_dir=input_dir,
            work_dir=work_dir,
            output_dir=output_dir,
            output_pdf_path=output_dir / f"{job_id}_pdfa.pdf",
            output_result_json_path=output_dir / f"{job_id}_result.json",
        )

    def check_storage_capacity(self, incoming_bytes: int = 0) -> dict[str, float]:
        """Verify free disk space and configured storage directory quota (FR-09, FR-16)."""
        usage = shutil.disk_usage(self._base_dir)
        free_mb = usage.free / (1024 * 1024)
        used_storage_bytes = self.compute_directory_size(self._jobs_dir)
        max_quota_bytes = int(self._settings.max_disk_usage_gb * 1024 * 1024 * 1024)

        if free_mb < self._settings.min_free_disk_mb:
            raise StorageQuotaExceededError(
                f"Insufficient free disk space ({free_mb:.1f} MB free, "
                f"minimum required {self._settings.min_free_disk_mb} MB).",
                details={"free_mb": round(free_mb, 2)},
            )
        if used_storage_bytes + incoming_bytes > max_quota_bytes:
            raise StorageQuotaExceededError(
                "Temporary OCR storage quota exceeded.",
                details={
                    "used_gb": round(used_storage_bytes / (1024**3), 3),
                    "max_gb": self._settings.max_disk_usage_gb,
                },
            )
        return {
            "free_mb": round(free_mb, 2),
            "used_mb": round(used_storage_bytes / (1024 * 1024), 2),
            "quota_mb": round(max_quota_bytes / (1024 * 1024), 2),
        }

    def save_upload_streams(
        self,
        job_id: str,
        uploads: list[tuple[str, BinaryIO]],
    ) -> tuple[list[Path], str, int]:
        """Stream uploaded files to disk while enforcing MAX_UPLOAD_MB and computing SHA-256."""
        self.check_storage_capacity()
        paths = self.get_job_paths(job_id, create=True)
        saved_paths: list[Path] = []
        hasher = hashlib.sha256()
        total_bytes = 0
        max_bytes = self._settings.max_upload_bytes

        try:
            for idx, (orig_name, file_obj) in enumerate(uploads):
                safe_name = sanitize_filename(orig_name, fallback=f"input_{idx:04d}.bin")
                dest_path = (paths.input_dir / f"{idx:04d}_{safe_name}").resolve()
                if not str(dest_path).startswith(str(paths.input_dir)):
                    raise ValidationError("Invalid upload file path.", code="PATH_TRAVERSAL")

                file_obj.seek(0)
                with open(dest_path, "wb") as out_f:
                    while True:
                        chunk = file_obj.read(CHUNK_SIZE)
                        if not chunk:
                            break
                        total_bytes += len(chunk)
                        if total_bytes > max_bytes:
                            raise PayloadTooLargeError(
                                f"Total upload size exceeds limit of {self._settings.max_upload_mb} MB.",
                                details={"max_upload_mb": self._settings.max_upload_mb},
                            )
                        hasher.update(chunk)
                        out_f.write(chunk)

                if dest_path.stat().st_size == 0:
                    raise ValidationError(
                        f"Uploaded file {orig_name!r} is empty (0 bytes).",
                        code="EMPTY_FILE",
                    )
                saved_paths.append(dest_path)
        except Exception:
            self.purge_job_artifacts(job_id)
            raise

        return saved_paths, hasher.hexdigest(), total_bytes

    def cleanup_working_dir(self, job_id: str) -> None:
        """Remove intermediate working files after job completion while keeping output artifacts."""
        paths = self.get_job_paths(job_id, create=False)
        if paths.work_dir.exists():
            shutil.rmtree(paths.work_dir, ignore_errors=True)

    def purge_job_artifacts(self, job_id: str) -> bool:
        """Completely remove a job's directory (input, work, output) upon expiration or cancellation."""
        paths = self.get_job_paths(job_id, create=False)
        if paths.job_dir.exists():
            shutil.rmtree(paths.job_dir, ignore_errors=True)
            return True
        return False

    @staticmethod
    def compute_directory_size(directory: Path) -> int:
        if not directory.exists():
            return 0
        total = 0
        for p in directory.rglob("*"):
            try:
                if p.is_file():
                    total += p.stat().st_size
            except OSError:
                continue
        return total
