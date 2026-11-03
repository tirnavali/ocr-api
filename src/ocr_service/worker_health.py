"""Container health check for worker processes (``ocr-worker-health``).

Workers serve no HTTP port, so the image's default HEALTHCHECK (curl :8000) always fails
for them. A worker container is healthy when at least one worker process on this host
has written a heartbeat recently. Kept free of OpenCV/pipeline imports so it starts fast.
"""

from __future__ import annotations

import socket
import sys

from ocr_service.config import get_settings
from ocr_service.infrastructure.db.database import DatabaseSessionManager
from ocr_service.infrastructure.db.repository import JobRepository

# Heartbeats are written every WORKER_HEARTBEAT_SECONDS (default 15); allow a few misses
# so a worker busy on a long page is not reported unhealthy.
MIN_MAX_AGE_SECONDS = 60


def check_worker_health(repo: JobRepository, hostname: str, max_age_seconds: int) -> bool:
    return repo.count_live_workers_on_host(hostname, max_age_seconds) > 0


def main() -> None:
    settings = get_settings()
    max_age = max(MIN_MAX_AGE_SECONDS, settings.worker_heartbeat_seconds * 4)
    db = DatabaseSessionManager(settings)
    try:
        healthy = check_worker_health(JobRepository(db), socket.gethostname(), max_age)
    except Exception as exc:
        print(f"unhealthy: {exc}", file=sys.stderr)
        sys.exit(1)
    finally:
        db.dispose()
    if not healthy:
        print(f"unhealthy: no worker heartbeat in the last {max_age}s", file=sys.stderr)
        sys.exit(1)
    print("healthy")


if __name__ == "__main__":
    main()
