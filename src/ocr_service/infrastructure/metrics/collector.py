"""Collect runtime CPU, memory, disk, and job queue observability metrics (FR-18)."""

from __future__ import annotations

import os
from typing import Any

import psutil

from ocr_service.infrastructure.db.repository import JobRepository
from ocr_service.infrastructure.storage.local_storage import LocalJobStorage


class MetricsCollector:
    """Gather operational metrics across DB queue, storage volume, and host/process resources."""

    def __init__(self, repo: JobRepository, storage: LocalJobStorage) -> None:
        self._repo = repo
        self._storage = storage

    def collect(self) -> dict[str, Any]:
        queue_stats = self._repo.get_queue_and_system_stats()
        proc = psutil.Process(os.getpid())
        mem_info = proc.memory_info()
        vm = psutil.virtual_memory()

        used_storage_bytes = self._storage.compute_directory_size(self._storage.jobs_dir)

        return {
            **queue_stats,
            "system": {
                "cpu_count_logical": psutil.cpu_count(logical=True) or 1,
                "cpu_percent": psutil.cpu_percent(interval=None),
                "system_memory_total_mb": round(vm.total / (1024 * 1024), 1),
                "system_memory_available_mb": round(vm.available / (1024 * 1024), 1),
                "system_memory_percent": vm.percent,
                "process_rss_mb": round(mem_info.rss / (1024 * 1024), 2),
            },
            "storage": {
                "jobs_dir_used_mb": round(used_storage_bytes / (1024 * 1024), 2),
            },
        }
