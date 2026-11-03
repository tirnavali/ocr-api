"""Unit tests for the worker container health check (heartbeat based)."""

from __future__ import annotations

import socket
from datetime import timedelta

from ocr_service.config import Settings
from ocr_service.infrastructure.db.database import DatabaseSessionManager
from ocr_service.infrastructure.db.models import WorkerHeartbeatRecord, utcnow
from ocr_service.infrastructure.db.repository import JobRepository
from ocr_service.worker_health import check_worker_health


def _age_heartbeat(db: DatabaseSessionManager, worker_id: str, seconds: int) -> None:
    with db.session() as sess:
        rec = sess.get(WorkerHeartbeatRecord, worker_id)
        rec.last_heartbeat_at = utcnow() - timedelta(seconds=seconds)


def test_healthy_only_with_a_recent_heartbeat_on_this_host(test_settings: Settings) -> None:
    db = DatabaseSessionManager(test_settings)
    repo = JobRepository(db)
    host = socket.gethostname()

    assert check_worker_health(repo, host, 60) is False

    repo.record_worker_heartbeat("worker-ocr-01-1")
    assert check_worker_health(repo, host, 60) is True
    # Another container's workers do not make this one healthy.
    assert check_worker_health(repo, "another-container", 60) is False

    _age_heartbeat(db, "worker-ocr-01-1", 120)
    assert check_worker_health(repo, host, 60) is False
    db.dispose()


def test_restarted_container_reusing_a_worker_id_takes_over_the_heartbeat(
    test_settings: Settings,
) -> None:
    # worker_id embeds the PID, and PIDs repeat after a container restart.
    db = DatabaseSessionManager(test_settings)
    repo = JobRepository(db)
    repo.record_worker_heartbeat("worker-ocr-01-7")
    with db.session() as sess:
        sess.get(WorkerHeartbeatRecord, "worker-ocr-01-7").hostname = "old-container"

    repo.record_worker_heartbeat("worker-ocr-01-7")

    assert check_worker_health(repo, socket.gethostname(), 60) is True
    assert check_worker_health(repo, "old-container", 60) is False
    db.dispose()
