"""Package jobs: stream CDN files one by one into size-limited zip parts with a MANIFEST.

Disk use stays bounded by the zip parts themselves plus one file in flight: each file is
downloaded, stored into the open part (no compression; archive PDFs/TIFFs do not shrink)
and deleted before the next one is fetched. A file that cannot be fetched is skipped and
reported instead of failing the whole package, since one missing scan should not cost the
user thousands of good ones.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import time
import zipfile
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ocr_service.config import Settings
from ocr_service.domain.models import JobEntity, PackageEntry
from ocr_service.exceptions import (
    JobCancelledException,
    JobTimeoutError,
    OcrServiceError,
)
from ocr_service.infrastructure.cdn.cdn_client import CdnFetchError, InternalCdnClient
from ocr_service.infrastructure.db.repository import JobRepository
from ocr_service.infrastructure.storage.local_storage import LocalJobStorage
from ocr_service.logging_config import correlation_id_ctx, job_id_ctx

logger = logging.getLogger(__name__)

PACKAGE_ENTRIES_FILENAME = "package_entries.jsonl"
PART_FILENAME_RE = re.compile(r"^[A-Za-z0-9._-]{1,200}\.zip$")
MAX_REPORTED_ERRORS = 200
# Consecutive network-level failures (no HTTP answer at all) mean the CDN is down rather
# than a few files missing; the attempt is then retried as a whole instead of producing
# a package with every file marked as skipped.
MAX_CONSECUTIVE_NETWORK_FAILURES = 5
FILE_FETCH_ATTEMPTS = 2
FILE_RETRY_DELAY_SECONDS = 1.0
CAPACITY_CHECK_EVERY = 50
LEASE_RENEW_INTERVAL_SECONDS = 10.0
PROGRESS_SAVE_EVERY = 25

_UNSAFE_SEGMENT_CHARS_RE = re.compile(r'[\x00-\x1f\x7f<>:"|?*]')
_UNSAFE_PREFIX_CHARS_RE = re.compile(r"[^A-Za-z0-9._-]")


def sanitize_archive_name(name: str, fallback: str) -> str:
    """Make a caller-supplied zip entry name safe to extract on Windows and Linux.

    Folders are kept (``Fon/K00001_belge.pdf``) but absolute paths, drive letters and
    ``..`` segments are dropped so extraction can never write outside the target.
    """
    segments: list[str] = []
    for raw in (name or "").replace("\\", "/").split("/"):
        # Trailing dots are invalid on Windows; stripping them also drops "." and "..".
        seg = _UNSAFE_SEGMENT_CHARS_RE.sub("_", raw).strip().rstrip(".")
        if seg:
            segments.append(seg[:150])
    cleaned = "/".join(segments)[:400].strip("/")
    return cleaned or fallback


def sanitize_part_prefix(prefix: str | None) -> str:
    cleaned = _UNSAFE_PREFIX_CHARS_RE.sub("_", (prefix or "").strip())[:120].strip("._")
    return cleaned or "package"


def read_package_entries(path: Path) -> list[PackageEntry]:
    entries: list[PackageEntry] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                entries.append(PackageEntry.from_dict(json.loads(line)))
    return entries


def write_package_entries(path: Path, entries: list[PackageEntry]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for entry in entries:
            fh.write(json.dumps(entry.to_dict(), ensure_ascii=False) + "\n")
    tmp.replace(path)


def package_part_path(storage: LocalJobStorage, job_id: str, filename: str) -> Path:
    """Resolve a finished part's file, refusing anything but a plain zip name in output/."""
    if not PART_FILENAME_RE.match(filename):
        raise ValueError(f"Invalid package part filename: {filename!r}")
    return storage.get_job_paths(job_id, create=False).output_dir / filename


@dataclass
class _OpenPart:
    part_no: int
    filename: str
    path: Path
    archive: zipfile.ZipFile
    manifest: list[str] = field(default_factory=list)
    file_count: int = 0

    @property
    def size_bytes(self) -> int:
        return self.path.stat().st_size


class PackageJobHandler:
    """Executes ``package`` jobs claimed by a worker of the package pool."""

    def __init__(
        self,
        settings: Settings,
        repo: JobRepository,
        storage: LocalJobStorage,
        cdn_client: InternalCdnClient | None = None,
    ) -> None:
        self._settings = settings
        self._repo = repo
        self._storage = storage
        self._cdn_client = cdn_client or InternalCdnClient(settings)

    def execute_job(self, job: JobEntity, worker_id: str) -> JobEntity:
        token_job = job_id_ctx.set(job.job_id)
        token_corr = correlation_id_ctx.set(job.correlation_id)
        start = time.monotonic()
        try:
            return self._run(job, worker_id, start)
        except JobCancelledException:
            logger.info("Package job cancelled", extra={"worker_id": worker_id})
            self._storage.purge_job_artifacts(job.job_id)
            return self._repo.finalize_cancelled_job(job.job_id)
        except OcrServiceError as exc:
            logger.warning(
                "Package job failed",
                extra={"error_code": exc.code, "retryable": exc.retryable, "error_message": exc.message},
            )
            return self._fail(job.job_id, exc.code, exc.message, exc.retryable)
        except Exception as exc:
            logger.exception("Unexpected error during package job")
            return self._fail(job.job_id, "INTERNAL_WORKER_ERROR", str(exc), True)
        finally:
            job_id_ctx.reset(token_job)
            correlation_id_ctx.reset(token_corr)

    def _fail(self, job_id: str, code: str, message: str, retryable: bool) -> JobEntity:
        updated = self._repo.mark_failed_or_retry(
            job_id=job_id,
            error_code=code,
            error_message=message,
            retryable=retryable,
            failed_retention_hours=self._settings.failed_retention_hours,
        )
        if updated.status.value == "failed":
            self._clear_outputs(job_id)
        self._storage.cleanup_working_dir(job_id)
        return updated

    def _clear_outputs(self, job_id: str) -> None:
        output_dir = self._storage.get_job_paths(job_id, create=False).output_dir
        if output_dir.exists():
            shutil.rmtree(output_dir, ignore_errors=True)

    def _run(self, job: JobEntity, worker_id: str, start: float) -> JobEntity:
        paths = self._storage.get_job_paths(job.job_id, create=True)
        entries = read_package_entries(paths.input_dir / PACKAGE_ENTRIES_FILENAME)
        options = dict(job.metadata_json.get("package_request") or {})
        max_part_bytes = int(options.get("max_part_mb") or 250) * 1024 * 1024
        prefix = sanitize_part_prefix(options.get("part_name_prefix"))
        manifest_title = str(options.get("manifest_title") or "")[:300]
        max_file_bytes = self._settings.max_package_file_mb * 1024 * 1024
        timeout = self._settings.job_timeout_for(job.job_type.value)

        # A retry starts from scratch: half-written parts of a crashed attempt are dropped.
        self._clear_outputs(job.job_id)
        paths.output_dir.mkdir(parents=True, exist_ok=True)
        if paths.work_dir.exists():
            shutil.rmtree(paths.work_dir, ignore_errors=True)
        paths.work_dir.mkdir(parents=True, exist_ok=True)
        self._storage.check_storage_capacity()
        self._repo.set_job_progress(job.job_id, total=len(entries), completed=0)

        logger.info(
            "Starting package job",
            extra={"worker_id": worker_id, "entry_count": len(entries), "attempt": job.attempt_count},
        )

        last_renew = time.monotonic()

        def keep_alive(_written: int = 0) -> None:
            nonlocal last_renew
            if time.monotonic() - start > timeout:
                raise JobTimeoutError(f"Job {job.job_id} exceeded timeout of {timeout} seconds.")
            if time.monotonic() - last_renew >= LEASE_RENEW_INTERVAL_SECONDS:
                last_renew = time.monotonic()
                if self._repo.renew_job_lease(
                    job.job_id, worker_id, lease_seconds=self._settings.worker_lease_seconds
                ):
                    raise JobCancelledException(f"Job {job.job_id} cancelled by client request.")

        parts: list[dict[str, Any]] = []
        errors: list[dict[str, str]] = []
        error_count = 0
        ok_count = 0
        total_bytes = 0
        used_names: set[str] = {"manifest.txt"}  # reserved for the per-part manifest
        pending_manifest: list[str] = []
        current: _OpenPart | None = None
        consecutive_network_failures = 0

        def record_error(entry: PackageEntry, arcname: str, code: str, message: str) -> None:
            nonlocal error_count
            error_count += 1
            if len(errors) < MAX_REPORTED_ERRORS:
                errors.append(
                    {"ref": entry.ref, "name": arcname, "path": entry.path, "error_code": code, "message": message[:300]}
                )
            line = f"HATA;{entry.ref};{arcname};;{code}"
            (current.manifest if current else pending_manifest).append(line)

        try:
            for index, entry in enumerate(entries):
                keep_alive()
                if index % CAPACITY_CHECK_EVERY == 0 and index:
                    self._storage.check_storage_capacity()

                arcname = self._unique_name(
                    used_names, sanitize_archive_name(entry.name, fallback=f"dosya_{index + 1:06d}")
                )
                tmp_file = paths.work_dir / f"{index:06d}.bin"
                try:
                    size = self._fetch(entry.path, tmp_file, max_file_bytes, keep_alive)
                    consecutive_network_failures = 0
                except CdnFetchError as exc:
                    if "status_code" not in (exc.details or {}):
                        consecutive_network_failures += 1
                        if consecutive_network_failures >= MAX_CONSECUTIVE_NETWORK_FAILURES:
                            raise
                    record_error(entry, arcname, exc.code, exc.message)
                    continue
                except (JobCancelledException, JobTimeoutError):
                    raise
                except OcrServiceError as exc:
                    record_error(entry, arcname, exc.code, exc.message)
                    continue

                try:
                    # Close a full part only when another file needs room, so errors that
                    # come after the last file still land in the last part's MANIFEST.
                    if current is not None and current.size_bytes >= max_part_bytes:
                        parts.append(self._close_part(current, manifest_title, job.job_id))
                        current = None
                    if current is None:
                        current = self._open_part(paths.output_dir, prefix, len(parts) + 1)
                        current.manifest.extend(pending_manifest)
                        pending_manifest.clear()
                    current.archive.write(tmp_file, arcname=arcname)
                    current.file_count += 1
                    current.manifest.append(f"OK;{entry.ref};{arcname};{size}")
                    ok_count += 1
                    total_bytes += size
                finally:
                    tmp_file.unlink(missing_ok=True)

                if (index + 1) % PROGRESS_SAVE_EVERY == 0:
                    self._repo.set_job_progress(job.job_id, total=len(entries), completed=index + 1)

            if current is not None:
                parts.append(self._close_part(current, manifest_title, job.job_id))
                current = None
        finally:
            if current is not None:
                current.archive.close()

        duration_ms = int((time.monotonic() - start) * 1000)
        result = {
            "entry_count": len(entries),
            "ok_count": ok_count,
            "error_count": error_count,
            "total_bytes": total_bytes,
            "parts": parts,
            "errors": errors,
            "errors_truncated": error_count > len(errors),
        }
        self._repo.merge_job_metadata(job.job_id, {"package": result})
        self._repo.set_job_progress(job.job_id, total=len(entries), completed=len(entries))
        self._storage.cleanup_working_dir(job.job_id)

        if ok_count == 0:
            logger.warning("Package job produced no files", extra={"error_count": error_count})
            return self._fail(
                job.job_id,
                "PACKAGE_EMPTY",
                f"None of the {len(entries)} files could be fetched from the CDN.",
                False,
            )

        completed = self._repo.mark_completed(
            job_id=job.job_id,
            page_count=len(entries),
            processing_duration_ms=duration_ms,
            retention_hours=self._settings.retention_hours_for(job.job_type.value),
            pdfa_compliant=None,
            pdfa_validator=None,
        )
        logger.info(
            "Completed package job",
            extra={"ok_count": ok_count, "error_count": error_count, "part_count": len(parts), "duration_ms": duration_ms},
        )
        return completed

    def _fetch(self, cdn_path: str, dest: Path, max_bytes: int, keep_alive: Any) -> int:
        """Download one file, retrying a single time on transient CDN errors."""
        for attempt in range(1, FILE_FETCH_ATTEMPTS + 1):
            try:
                return self._cdn_client.download_to(
                    cdn_path, dest, max_bytes=max_bytes, on_progress=keep_alive
                )
            except CdnFetchError:
                if attempt >= FILE_FETCH_ATTEMPTS:
                    raise
                time.sleep(FILE_RETRY_DELAY_SECONDS)
        raise AssertionError("unreachable")

    @staticmethod
    def _unique_name(used: set[str], name: str) -> str:
        candidate = name
        counter = 1
        while candidate.lower() in used:
            counter += 1
            stem, dot, ext = name.rpartition(".")
            candidate = f"{stem}_{counter}.{ext}" if dot and "/" not in ext else f"{name}_{counter}"
        used.add(candidate.lower())
        return candidate

    @staticmethod
    def _open_part(output_dir: Path, prefix: str, part_no: int) -> _OpenPart:
        filename = f"{prefix}_paket-{part_no:02d}.zip"
        path = output_dir / filename
        archive = zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED, allowZip64=True)
        return _OpenPart(part_no=part_no, filename=filename, path=path, archive=archive)

    @staticmethod
    def _close_part(part: _OpenPart, manifest_title: str, job_id: str) -> dict[str, Any]:
        header = [
            f"# {manifest_title}" if manifest_title else f"# Package {job_id}",
            f"# Paket: {part.part_no}, Tarih: {datetime.now(UTC):%Y-%m-%d %H:%M} UTC",
            "# Durum;Ref;PaketIciAd;Boyut;HataKodu",
        ]
        text = "\n".join(header + part.manifest) + "\n"
        part.archive.writestr("MANIFEST.txt", text.encode("utf-8"))
        part.archive.close()
        return {
            "part_no": part.part_no,
            "filename": part.filename,
            "size_bytes": part.path.stat().st_size,
            "file_count": part.file_count,
        }
