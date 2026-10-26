"""Performance, memory, throughput, and Turkish CER/WER benchmark suite (Section 7)."""

from __future__ import annotations

import json
import multiprocessing
import os
import shutil
import tempfile
import time
from pathlib import Path

import psutil
from tests.conftest import render_turkish_archival_image, render_two_column_newspaper_image
from tests.validation.test_turkish_ocr_quality import compute_cer, compute_wer

from ocr_service.config import Settings
from ocr_service.domain.models import JobSubmissionOptions, LayoutMode, PdfAProfile
from ocr_service.infrastructure.db.database import DatabaseSessionManager
from ocr_service.infrastructure.db.repository import JobRepository
from ocr_service.infrastructure.storage.local_storage import LocalJobStorage
from ocr_service.worker_main import OcrWorker


def _worker_drain_queue(storage_dir_str: str, worker_idx: int) -> float:
    """Worker process target that drains queued jobs and returns peak RSS in MB."""
    storage_dir = Path(storage_dir_str)
    settings = Settings(
        APP_ENV="production",
        LOG_JSON=False,
        LOG_LEVEL="ERROR",
        TEMP_STORAGE_PATH=storage_dir,
        OCR_THREADS_PER_WORKER=1,
        MAX_CONCURRENT_JOBS=16,
    )
    settings.enforce_cpu_thread_limits()
    db = DatabaseSessionManager(settings)
    repo = JobRepository(db)
    storage = LocalJobStorage(settings)
    worker = OcrWorker(settings, repo, storage, worker_id=f"bench-worker-{worker_idx}")
    proc = psutil.Process(os.getpid())
    peak_rss_mb = 0.0
    try:
        while True:
            res = worker.process_next_available_job()
            rss = proc.memory_info().rss / (1024 * 1024)
            peak_rss_mb = max(peak_rss_mb, rss)
            if res is None:
                break
    finally:
        db.dispose()
    return peak_rss_mb


def run_benchmark_suite() -> dict[str, object]:
    tmp_root = Path(tempfile.mkdtemp(prefix="ocr_bench_"))
    try:
        samples_dir = tmp_root / "samples"
        samples_dir.mkdir(parents=True, exist_ok=True)

        # 1. Prepare Turkish archival and newspaper test pages
        archival_lines = [
            "TÜRKİYE BÜYÜK MİLLET MECLİSİ TUTANAK DERGİSİ",
            "Birinci Oturum: Cumhuriyet arşiv belgelerinin dijitalleştirilmesi.",
            "Özel Türkçe harfler ve işaretler: çığöşü ÇİĞÖŞÜ.",
            "Ankara, İstanbul, İzmir, Eskişehir, Diyarbakır, Gümüşhane, Şanlıurfa.",
            "Sayfa bütünlüğü, görüntü kalitesi ve metin hizalaması korunmaktadır.",
        ]
        sample_doc = render_turkish_archival_image(
            samples_dir / "archival_page.jpg",
            archival_lines,
            width_px=1800,
            height_px=2400,
            dpi=300,
            fmt="JPEG",
        )
        newspaper_path = render_two_column_newspaper_image(
            samples_dir / "newspaper_page.png",
            headline="CUMHURİYET ARŞİV GAZETESİ BAŞLIĞI",
            left_column_lines=[
                "Sol sütun birinci satır arşiv haberi.",
                "Sol sütun ikinci satır tarihsel kayıt.",
                "Sol sütun üçüncü satır meclis kararı.",
            ],
            right_column_lines=[
                "Sağ sütun birinci satır ekonomi haberi.",
                "Sağ sütun ikinci satır kültür sanat.",
                "Sağ sütun üçüncü satır sonuç bildirgesi.",
            ],
        )

        # Single-worker latency + CER/WER accuracy measurement
        single_dir = tmp_root / "single_run"
        settings = Settings(
            APP_ENV="production",
            LOG_JSON=False,
            LOG_LEVEL="ERROR",
            TEMP_STORAGE_PATH=single_dir,
            OCR_THREADS_PER_WORKER=1,
        )
        settings.enforce_cpu_thread_limits()
        db = DatabaseSessionManager(settings)
        repo = JobRepository(db)
        storage = LocalJobStorage(settings)
        worker = OcrWorker(settings, repo, storage, worker_id="bench-single")

        with open(sample_doc.path, "rb") as f1, open(newspaper_path, "rb") as f2:
            saved_paths, sha, _ = storage.save_upload_streams(
                "job-bench-single",
                [("archival.jpg", f1), ("newspaper.png", f2)],
            )
        repo.enqueue_job(
            job_id="job-bench-single",
            options=JobSubmissionOptions(
                language="tur",
                working_dpi=300,
                layout_mode=LayoutMode.AUTO,
                pdfa_profile=PdfAProfile.PDFA_2B,
            ),
            input_filenames=["archival.jpg", "newspaper.png"],
            input_paths=saved_paths,
            content_sha256=sha,
            idempotency_key=None,
            correlation_id="bench-1",
            max_retries=1,
            max_queue_depth=100,
        )

        t0 = time.perf_counter()
        completed = worker.process_next_available_job()
        elapsed_single = time.perf_counter() - t0
        assert completed is not None and completed.status.value == "completed"

        pages = repo.get_job_pages("job-bench-single", include_boxes=True)
        cer_page1 = compute_cer(sample_doc.ground_truth_text, pages[0].text)
        wer_page1 = compute_wer(sample_doc.ground_truth_text, pages[0].text)
        disk_used_mb = storage.compute_directory_size(storage.jobs_dir) / (1024 * 1024)
        db.dispose()

        # Concurrent throughput scaling across worker counts (1, 2, 4 workers, 8 jobs = 16 pages)
        concurrency_results: list[dict[str, object]] = []
        ctx = multiprocessing.get_context("spawn")

        for worker_count in (1, 2, 4):
            run_dir = tmp_root / f"pool_{worker_count}"
            s_pool = Settings(
                APP_ENV="production",
                LOG_JSON=False,
                LOG_LEVEL="ERROR",
                TEMP_STORAGE_PATH=run_dir,
                OCR_THREADS_PER_WORKER=1,
                MAX_CONCURRENT_JOBS=worker_count,
            )
            db_pool = DatabaseSessionManager(s_pool)
            repo_pool = JobRepository(db_pool)
            storage_pool = LocalJobStorage(s_pool)

            num_jobs = 8  # 8 jobs x 2 pages = 16 pages
            total_pages = num_jobs * 2
            for j in range(num_jobs):
                jid = f"job-pool-{worker_count}-{j:02d}"
                with open(sample_doc.path, "rb") as f1, open(newspaper_path, "rb") as f2:
                    spaths, jsha, _ = storage_pool.save_upload_streams(
                        jid, [("p1.jpg", f1), ("p2.png", f2)]
                    )
                repo_pool.enqueue_job(
                    job_id=jid,
                    options=JobSubmissionOptions(),
                    input_filenames=["p1.jpg", "p2.png"],
                    input_paths=spaths,
                    content_sha256=jsha,
                    idempotency_key=None,
                    correlation_id=f"corr-{j}",
                    max_retries=1,
                    max_queue_depth=200,
                )
            db_pool.dispose()

            t_start = time.perf_counter()
            with ctx.Pool(processes=worker_count) as pool:
                rss_list = pool.starmap(
                    _worker_drain_queue,
                    [(str(run_dir), idx + 1) for idx in range(worker_count)],
                )
            wall_seconds = time.perf_counter() - t_start
            pages_per_min = (total_pages / wall_seconds) * 60.0
            avg_latency_ms = (wall_seconds * 1000.0) / total_pages

            concurrency_results.append(
                {
                    "worker_count": worker_count,
                    "threads_per_worker": 1,
                    "total_pages": total_pages,
                    "wall_time_seconds": round(wall_seconds, 2),
                    "pages_per_minute": round(pages_per_min, 1),
                    "effective_ms_per_page": round(avg_latency_ms, 1),
                    "max_worker_rss_mb": round(max(rss_list), 1),
                    "total_workers_rss_mb": round(sum(rss_list), 1),
                }
            )

        summary = {
            "quality_metrics": {
                "turkish_archival_cer": round(cer_page1, 4),
                "turkish_archival_wer": round(wer_page1, 4),
                "page1_confidence_mean": round(pages[0].mean_confidence, 2),
                "page2_columns_detected": pages[1].column_count,
            },
            "single_job_2pages": {
                "total_duration_seconds": round(elapsed_single, 2),
                "page1_duration_ms": pages[0].processing_duration_ms,
                "page2_duration_ms": pages[1].processing_duration_ms,
                "output_disk_mb": round(disk_used_mb, 2),
            },
            "concurrency_scaling": concurrency_results,
        }
        return summary
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


if __name__ == "__main__":
    results = run_benchmark_suite()
    print(json.dumps(results, indent=2, ensure_ascii=False))
