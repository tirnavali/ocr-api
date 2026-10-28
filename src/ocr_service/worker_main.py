"""Background worker process, crash recovery supervisor, and retention cleanup janitor."""

from __future__ import annotations

import logging
import multiprocessing
import os
import signal
import socket
import time
import uuid
from types import FrameType
from typing import Protocol

import cv2

from ocr_service.config import Settings, get_settings
from ocr_service.domain.models import JobEntity, JobType
from ocr_service.infrastructure.callbacks.notifier import CallbackNotifier
from ocr_service.infrastructure.db.database import DatabaseSessionManager
from ocr_service.infrastructure.db.repository import JobRepository
from ocr_service.infrastructure.storage.local_storage import LocalJobStorage
from ocr_service.logging_config import configure_logging
from ocr_service.pipeline.orchestrator import OcrPipelineOrchestrator

logger = logging.getLogger(__name__)

CALLBACK_POLL_INTERVAL_SECONDS = 5.0


class JobHandler(Protocol):
    """Executes one claimed job of a given type and returns its updated state."""

    def execute_job(self, job: JobEntity, worker_id: str) -> JobEntity: ...


class RetentionAndRecoveryJanitor:
    """Recover stale worker leases and clean up expired job files safely (FR-02, FR-09)."""

    def __init__(
        self,
        settings: Settings,
        repo: JobRepository,
        storage: LocalJobStorage,
    ) -> None:
        self._settings = settings
        self._repo = repo
        self._storage = storage

    def run_maintenance_cycle(self) -> dict[str, int]:
        """Run one pass of crash recovery and expired artifact cleanup."""
        recovered_ids = self._repo.recover_stale_jobs(
            failed_retention_hours=self._settings.failed_retention_hours
        )
        if recovered_ids:
            logger.warning(
                "Recovered jobs with expired worker leases",
                extra={"recovered_job_ids": recovered_ids, "count": len(recovered_ids)},
            )

        expired_candidates = self._repo.find_jobs_due_for_expiration()
        expired_count = 0
        for job_id in expired_candidates:
            # Verify the job is not active before touching files (FR-09)
            marked = self._repo.mark_job_expired(
                job_id,
                metadata_retention_days=self._settings.metadata_retention_days,
            )
            if marked:
                self._storage.purge_job_artifacts(job_id)
                expired_count += 1
                logger.info("Purged expired job artifacts", extra={"expired_job_id": job_id})

        return {
            "recovered_jobs": len(recovered_ids),
            "expired_jobs": expired_count,
        }


class OcrWorker:
    """Single worker instance polling the durable job queue with graceful shutdown.

    A worker only claims the job types in ``WORKER_JOB_TYPES`` that it has a handler for,
    so OCR and other job types run in separate worker pools.
    """

    def __init__(
        self,
        settings: Settings,
        repo: JobRepository,
        storage: LocalJobStorage,
        orchestrator: OcrPipelineOrchestrator | None = None,
        worker_id: str | None = None,
        handlers: dict[str, JobHandler] | None = None,
        notifier: CallbackNotifier | None = None,
    ) -> None:
        self._settings = settings
        self._repo = repo
        self._storage = storage
        self._handlers: dict[str, JobHandler] = dict(handlers or {})
        if JobType.OCR.value in settings.worker_job_types and JobType.OCR.value not in self._handlers:
            self._handlers[JobType.OCR.value] = orchestrator or OcrPipelineOrchestrator(
                settings, repo, storage
            )
        self._job_types = [t for t in settings.worker_job_types if t in self._handlers]
        missing = [t for t in settings.worker_job_types if t not in self._handlers]
        if missing:
            logger.warning(
                "No handler registered for configured job types; they will not be claimed",
                extra={"job_types": missing},
            )
        self._max_concurrent_by_type = {
            t: settings.max_concurrent_jobs_for(t) for t in self._job_types
        }
        self._janitor = RetentionAndRecoveryJanitor(settings, repo, storage)
        self._notifier = notifier or CallbackNotifier(settings, repo)
        self.worker_id = (
            worker_id or f"worker-{socket.gethostname()}-{os.getpid()}-{uuid.uuid4().hex[:6]}"
        )
        self._stop_requested = False

    def request_stop(self, *_args: object) -> None:
        logger.info("Graceful shutdown signal received by worker", extra={"worker_id": self.worker_id})
        self._stop_requested = True

    def process_next_available_job(self) -> JobEntity | None:
        """Claim and execute a single queued job if available (useful for synchronous test runs)."""
        if not self._job_types:
            return None
        job = self._repo.claim_next_job(
            worker_id=self.worker_id,
            lease_seconds=self._settings.worker_lease_seconds,
            max_concurrent_jobs=self._settings.max_concurrent_jobs,
            job_types=self._job_types,
            max_concurrent_by_type=self._max_concurrent_by_type,
        )
        if job is None:
            return None

        self._repo.record_worker_heartbeat(
            self.worker_id, status="processing", active_job_id=job.job_id
        )
        result = self._handlers[job.job_type.value].execute_job(job, worker_id=self.worker_id)
        self._repo.record_worker_heartbeat(
            self.worker_id, status="idle", active_job_id=None, increment_processed=True
        )
        # Notify right away instead of waiting for the next callback poll.
        self.deliver_callbacks()
        return result

    def deliver_callbacks(self) -> int:
        try:
            return self._notifier.deliver_due()
        except Exception:
            logger.exception("Error while delivering job callbacks")
            return 0

    def run_forever(self) -> None:
        """Run the polling and maintenance loop until graceful shutdown is requested."""
        self._settings.enforce_cpu_thread_limits()
        cv2.setNumThreads(self._settings.ocr_threads_per_worker)

        signal.signal(signal.SIGTERM, self.request_stop)
        signal.signal(signal.SIGINT, self.request_stop)

        logger.info(
            "OCR worker started",
            extra={
                "worker_id": self.worker_id,
                "threads_per_worker": self._settings.ocr_threads_per_worker,
                "job_types": self._job_types,
            },
        )

        last_cleanup = 0.0
        last_heartbeat = 0.0
        last_callback_poll = 0.0

        while not self._stop_requested:
            now = time.monotonic()
            if now - last_heartbeat >= self._settings.worker_heartbeat_seconds:
                self._repo.record_worker_heartbeat(self.worker_id, status="idle")
                last_heartbeat = now

            if now - last_cleanup >= self._settings.cleanup_interval_seconds:
                try:
                    self._janitor.run_maintenance_cycle()
                except Exception:
                    logger.exception("Error during janitor maintenance cycle")
                last_cleanup = now

            # Retries of failed callbacks (and callbacks of jobs another process finished
            # or the janitor failed) are picked up here.
            if now - last_callback_poll >= CALLBACK_POLL_INTERVAL_SECONDS:
                self.deliver_callbacks()
                last_callback_poll = now

            try:
                processed = self.process_next_available_job()
            except Exception:
                logger.exception("Unhandled error in worker poll loop")
                processed = None

            if processed is None and not self._stop_requested:
                time.sleep(self._settings.worker_poll_interval_seconds)

        logger.info("OCR worker shut down cleanly", extra={"worker_id": self.worker_id})


def _worker_process_target(worker_index: int) -> None:
    settings = get_settings()
    settings.enforce_cpu_thread_limits()
    configure_logging(settings.log_level, settings.log_json)
    db = DatabaseSessionManager(settings)
    repo = JobRepository(db)
    storage = LocalJobStorage(settings)
    pool = "+".join(settings.worker_job_types)
    worker_id = f"worker-{pool}-{worker_index:02d}-{os.getpid()}"
    worker = OcrWorker(settings, repo, storage, worker_id=worker_id)
    try:
        worker.run_forever()
    finally:
        db.dispose()


def run_worker() -> None:
    """Entrypoint for starting 1..N OCR worker processes."""
    settings = get_settings()
    settings.enforce_cpu_thread_limits()
    configure_logging(settings.log_level, settings.log_json)

    count = settings.ocr_worker_count
    if count <= 1:
        _worker_process_target(1)
        return

    logger.info("Starting OCR worker pool supervisor", extra={"worker_count": count})
    ctx = multiprocessing.get_context("spawn")
    processes: list[multiprocessing.Process] = []

    stop_pool = False

    def _handle_sig(_signum: int, _frame: FrameType | None) -> None:
        nonlocal stop_pool
        stop_pool = True
        for p in processes:
            if p.is_alive() and p.pid:
                os.kill(p.pid, signal.SIGTERM)

    signal.signal(signal.SIGTERM, _handle_sig)
    signal.signal(signal.SIGINT, _handle_sig)

    for idx in range(1, count + 1):
        p = ctx.Process(target=_worker_process_target, args=(idx,), name=f"ocr-worker-{idx}")
        p.start()
        processes.append(p)

    try:
        while not stop_pool:
            for idx, p in enumerate(processes):
                if not p.is_alive() and not stop_pool:
                    logger.warning(
                        "Worker child process exited unexpectedly; respawning",
                        extra={"worker_index": idx + 1, "exitcode": p.exitcode},
                    )
                    new_p = ctx.Process(
                        target=_worker_process_target,
                        args=(idx + 1,),
                        name=f"ocr-worker-{idx + 1}",
                    )
                    new_p.start()
                    processes[idx] = new_p
            time.sleep(1.0)
    finally:
        for p in processes:
            p.join(timeout=15.0)
            if p.is_alive():
                p.terminate()
