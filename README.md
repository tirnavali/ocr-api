# Archival Turkish OCR & Searchable PDF/A-2b Service

Production-ready, Linux-compatible Python OCR API service for processing digitized newspapers, journals, historical archival documents, multipage TIFFs, and scanned PDFs into standards-compliant **searchable PDF/A-2b** (and **PDF/A-1b**) documents with sub-point-aligned invisible Unicode text layers.

---

## 1. Key Engineering Highlights & Legacy .NET Improvements

| Capability | Legacy .NET Service (`OcrMyImageAndPdf`) | Python Production Service (`ocr-service`) |
| :--- | :--- | :--- |
| **OS Compatibility** | Windows-only (`System.Drawing`, `gswin64c.exe`) | Cross-platform Linux AMD64 & macOS Apple Silicon ARM64 |
| **Execution Model** | Blocking HTTP calls returning raw PDF bytes in JSON | Durable asynchronous SQLite-WAL lease queue (`202 Accepted`), page-level checkpointing, crash recovery |
| **Visual Fidelity** | Downsampled images >3000px and re-encoded at JPEG Q60 / 265 DPI | **Dual-Track Pipeline**: lossless `/DCTDecode` passthrough for JPEGs & `/FlateDecode` for PNG/TIFF; working copy preprocessing is strictly ephemeral |
| **Rotation & Deskew Alignment** | Rotated OCR image while printing unrotated image in PDF (severe misalignment bug) | Exact inverse affine matrix ($M_{\text{deskew}}^{-1}$ + inverse cardinal rotation + inverse DPI scale) mapping working coordinates back to source image and PDF points |
| **Invisible Text Layer** | Visible `DirectContentUnder` with unembedded `CP1254` / `WINANSI` fonts | PDF `3 Tr` invisible text rendering mode + embedded Unicode `Type0` / `CIDFontType2` (`Identity-H` + `/ToUnicode` CMap) + per-word horizontal scaling (`Tz`) |
| **Newspaper Reading Order** | Raw Tesseract top-to-bottom scan order | Multi-column XY-cut / gutter projection analyzer separating spanning banner headlines from vertical columns without altering word geometry |

---

## 2. REST API Contract (`/api/v1`)

Interactive OpenAPI / Swagger UI is served at `/docs` and ReDoc at `/redoc`.

| Method | Endpoint | Status | Description |
| :--- | :--- | :--- | :--- |
| `POST` | `/api/v1/jobs` *(alias `/api/v1/ocr/jobs`)* | `202 Accepted` | Upload 1 PDF or 1..N images (`JPEG`, `PNG`, single/multipage `TIFF`). Supports `Idempotency-Key` header and optional `callback_url` form field. |
| `POST` | `/api/v1/jobs/from-cdn` | `202 Accepted` | JSON body with `sources: [{path, filename?}]` referencing files on the internal CDN; the worker downloads them itself (nothing is uploaded). Same options as upload plus `callback_url`. |
| `GET` | `/api/v1/jobs/{job_id}` | `200 OK` | Poll job status (`queued`, `processing`, `completed`, `failed`, `cancelled`, `expired`) and page progress. |
| `GET` | `/api/v1/jobs/{job_id}/result` | `200 OK` / `409` / `410` | Retrieve structured JSON result with per-page Turkish text, dimensions, DPI metadata, and optional word boxes (`?include_boxes=true`). |
| `GET` | `/api/v1/jobs/{job_id}/pages` | `200 OK` | Retrieve paginated per-page OCR text and word geometry (`?page_start=1&page_end=5&include_boxes=true`). |
| `GET` | `/api/v1/jobs/{job_id}/pdf` | `200 OK` / `409` / `410` | Stream completed searchable PDF/A binary (`application/pdf`). |
| `GET` | `/api/v1/jobs/{job_id}/metadata` | `200 OK` | Retrieve audit events, per-page timings, DPI sources, image preservation modes, and PDF/A validation report. |
| `POST` | `/api/v1/jobs/{job_id}/cancel` | `200 OK` | Cancel a `queued` job immediately or signal cooperative cancellation on a `processing` job. |
| `POST` | `/api/v1/jobs/package` | `202 Accepted` | JSON body with `entries: [{path, name, ref?}]` (CDN path, path inside the zip, caller's id), `max_part_mb`, `part_name_prefix`, `manifest_title`, `callback_url`. Builds size-limited zip parts. |
| `GET` | `/api/v1/jobs/{job_id}/packages/{part_no}` | `200 OK` / `206` / `409` / `410` | Download one zip part of a completed package job; `Range` requests are supported. |
| `DELETE` | `/api/v1/jobs/{job_id}` | `200 OK` | Purge a job's files now (cancels it first if it is still running). |
| `GET` | `/api/v1/health/live` | `200 OK` | Container liveness check. |
| `GET` | `/api/v1/health/ready` | `200 OK` / `503` | Readiness check verifying SQLite WAL write access, disk quota, and Tesseract `tur` language data. |
| `GET` | `/api/v1/metrics` | `200 OK` | Queue depth, active jobs, page latency averages, error counters, CPU/RAM & storage metrics. |

### 2.1 Job Types & Worker Pools

Every job has a `job_type` (`ocr`, `package`). A worker process only claims the types listed in
`WORKER_JOB_TYPES`, so each type runs in its own pool. Concurrency (`MAX_CONCURRENT_JOBS` for OCR,
`MAX_CONCURRENT_PACKAGE_JOBS` for packaging), queue backpressure and retention
(`RESULT_RETENTION_HOURS` / `PACKAGE_RETENTION_HOURS`) are all applied per type. Existing state
databases are upgraded in place on startup (new columns and indexes are added).

### 2.2 CDN Inputs

`POST /api/v1/jobs/from-cdn` requires `CDN_BASE_URL` and `CDN_VERIFICATION_ID`. Paths are
CDN-relative (`AOS\ab\cd\file.pdf`); absolute URLs and `..` segments are rejected. The worker sends
the same `UygulamaId` / `TokenTime` headers as the TBMM Framework client. A `404` from the CDN fails
the job immediately (`CDN_SOURCE_NOT_FOUND`, no retries); network errors and `5xx` are retried.
Note: the TBMM internal CDN currently answers a missing file with `500` and an empty body, so such
jobs end as `CDN_FETCH_FAILED` after `JOB_MAX_RETRIES` retries rather than failing on the first try.

### 2.3 Package Jobs (zip export)

A `package` job is claimed only by a worker with `WORKER_JOB_TYPES=package` (the `package-worker`
service in `docker-compose.yml`). The worker downloads one file at a time from the CDN, stores it
uncompressed into the open zip part and deletes the download, so disk use is the parts themselves
plus one file in flight. A part is closed when it has passed `max_part_mb` and another file needs
room; every part ends with a `MANIFEST.txt` (`Durum;Ref;PaketIciAd;Boyut;HataKodu`).

- A file that cannot be fetched (missing, CDN `5xx` after one retry, larger than
  `MAX_PACKAGE_FILE_MB`) is skipped and listed in the result instead of failing the package.
- If no file at all could be fetched the job fails with `PACKAGE_EMPTY`; five consecutive network
  failures (no HTTP answer) mean the CDN is down and the whole attempt is retried later.
- Zip entry names keep their folders, but absolute paths, drive letters, `..` segments and characters
  invalid on Windows are removed; duplicate names get a `_2`, `_3` suffix.
- The entry list is stored in the job directory, not in the job table, so large exports do not
  slow down status queries. Limits: `MAX_PACKAGE_ENTRIES`, `MAX_PACKAGE_PART_MB`,
  `PACKAGE_JOB_TIMEOUT_SECONDS`; parts are kept for `PACKAGE_RETENTION_HOURS`.
- `GET /api/v1/jobs/{job_id}` and the callback both carry `package`: counts, the part list and the
  first 200 skipped files.

### 2.4 Turkish NER: rule engine and BERTurk (ONNX INT8)

`nlp_mode=fast` runs the rule engine (dictionary + capitalisation patterns). `nlp_mode=onnx`
runs a BERTurk token-classification model for PER/ORG/LOC on CPU via ONNX Runtime; DATE and
LAW_REF still come from the rules (the model has no such labels). The page reports which
engine actually ran in `nlp.engine_used` (`hybrid_onnx_tr` / `hybrid_fast_tr`): without a model,
or if it fails to load, onnx mode falls back to the rules instead of failing the job.

Model setup (not in git, ~110 MB):

1. Download `model.onnx`, `config.json`, `tokenizer.json` of
   [akdeniz27/bert-base-turkish-cased-ner](https://huggingface.co/akdeniz27/bert-base-turkish-cased-ner)
   (MIT, fine-tuned from dbmdz BERTurk) into a folder, e.g. `models/ner/`.
2. `pip install -e .[ner,ner-tools]` and run `python scripts/prepare_ner_model.py models/ner`
   to write `model_int8.onnx` (dynamic INT8; tags identical to FP32 on the check sentence).
3. Set `NER_MODEL_HOST_DIR` (host folder) and `NER_MODEL_DIR=/models/ner` for docker compose.

Measured on two OCR'd archive pages (1959 newspaper, magazine; 47 hand-labelled entities,
type-level match), one CPU thread:

| Engine | Recall | Precision | F1 | Time (2 pages) |
| :--- | ---: | ---: | ---: | ---: |
| rule (`fast`) | 0.04 | 0.40 | 0.08 | 0.06 s |
| BERTurk FP32 | 0.64 | 0.64 | 0.64 | 5.5 s |
| BERTurk INT8 (`onnx`) | 0.64 | 0.64 | 0.64 | 3.8 s |

Most remaining BERTurk errors come from OCR (misread names, words hyphenated across lines).

### 2.5 Signed Completion Callbacks

If a job has a `callback_url` (must start with one of `CALLBACK_ALLOWED_PREFIXES`), the service POSTs
a JSON summary when the job becomes `completed`, `failed` or `cancelled`. Delivery uses a durable
outbox: failures are retried with exponential backoff up to `CALLBACK_MAX_ATTEMPTS`, and the
delivery state is visible as `callback_status` on `GET /api/v1/jobs/{job_id}`.

Receivers must verify `X-Ocr-Signature: sha256=<hex>` = HMAC-SHA256(`CALLBACK_HMAC_SECRET`,
`"<X-Ocr-Timestamp>." + raw_body`) and treat callbacks idempotently by `job_id` (a callback may
arrive more than once).

---

## 3. Development Instructions (macOS Apple Silicon M4 Pro)

### 3.1 Native Dependencies
```bash
brew install tesseract tesseract-lang poppler qpdf
# Optional independent ISO 19005 validator:
brew install --cask temurin
brew install verapdf
```

### 3.2 Environment Setup & Running Locally
```bash
cd /Users/sercan/Projects/ocr-api
uv sync --extra dev

# Terminal 1: Start FastAPI server
uv run ocr-api

# Terminal 2: Start background OCR worker(s)
OCR_WORKER_COUNT=2 OCR_THREADS_PER_WORKER=1 uv run ocr-worker
```

### 3.3 Running Automated Tests & Benchmarks
```bash
# Run full unit, integration, and validation test suite (40 tests)
uv run pytest

# Run performance, memory, and Turkish CER/WER benchmark suite
PYTHONPATH="src:." uv run python tests/benchmark/run_benchmark.py
```

---

## 4. Production Deployment Instructions (Linux AMD EPYC 16-Core / 64 GB RAM)

### 4.1 Containerized Deployment via Docker Compose
```bash
cp .env.example .env
# Edit .env to set AUTH_ENABLED=true, API_KEYS=<strong-random-key>, and worker parameters
docker compose up -d --build
```

### 4.2 Hardware Sizing & Concurrency Recommendations (16 Cores, 64 GB RAM)
Based on our empirical benchmark measurements (`tests/benchmark/run_benchmark.py`):

| Worker Processes (`OCR_WORKER_COUNT`) | Threads per Worker (`OCR_THREADS_PER_WORKER`) | Throughput (Pages / Min) | Effective Latency (ms / Page) | Peak RSS per Worker (MB) | Total Worker Pool RAM (GB) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| `1` | `1` | `209.7` | `286.1 ms` | `439.2 MB` | `0.43 GB` |
| `2` | `1` | `369.9` | `162.2 ms` | `378.3 MB` | `0.73 GB` |
| `4` | `1` | `567.8` | `105.7 ms` | `341.2 MB` | `1.28 GB` |

- **Turkish OCR Quality on Archival Benchmark:**
  - **CER (Character Error Rate):** `0.33%` (`0.0033`)
  - **WER (Word Error Rate):** `2.94%` (`0.0294`)
- **Recommended Production Settings for 16-Core AMD EPYC / 64 GB RAM:**
  - `OCR_WORKER_COUNT=10` (or `12` for dedicated batch servers)
  - `OCR_THREADS_PER_WORKER=1` (`OMP_THREAD_LIMIT=1` is automatically enforced in code so Tesseract/OpenCV never oversubscribe CPU cores)
  - `MAX_CONCURRENT_JOBS=10`
  - Leaves 4–6 cores and >48 GB RAM headroom for FastAPI streaming I/O, `pikepdf` PDF/A-2b assembly, OS page cache, and `veraPDF` validation.

---

## 5. Requirements-to-Tests Traceability Matrix

| Requirement | Description | Implementation Module(s) | Verification Test(s) |
| :--- | :--- | :--- | :--- |
| **FR-01** | Python 3.12+, FastAPI, Pydantic v2, OpenAPI, strict typing | `src/ocr_service/main.py`, `src/ocr_service/api/schemas.py` | `tests/integration/test_api_workflow.py` |
| **FR-02** | Durable async job queue, retries, crash recovery, cancellation, idempotency | `src/ocr_service/infrastructure/db/repository.py`, `src/ocr_service/worker_main.py` | `tests/unit/test_job_lifecycle.py`, `tests/integration/test_worker_recovery.py` |
| **FR-03** | JPEG, PNG, multipage TIFF, scanned PDF, multi-image array, existing text layer inspection (`force_ocr`) | `src/ocr_service/pipeline/inspector.py`, `src/ocr_service/pipeline/orchestrator.py` | `tests/integration/test_multipage_and_multiimage.py`, `tests/integration/test_pdf_text_inspection.py` |
| **FR-04** | Pluggable CPU OCR engine (`OcrEnginePort`) with Tesseract 5 Turkish LSTM | `src/ocr_service/domain/ports.py`, `src/ocr_service/pipeline/ocr_engine.py` | `tests/validation/test_turkish_ocr_quality.py` |
| **FR-05** | Searchable PDF/A-2b & PDF/A-1b generation and validation | `src/ocr_service/pipeline/pdfa_generator.py`, `src/ocr_service/pipeline/pdfa_validator.py` | `tests/validation/test_pdfa_conformance.py` |
| **FR-06** | Accurate invisible text layer (`3 Tr`) coordinate alignment, rotation & deskew inverse mapping | `src/ocr_service/pipeline/geometry.py`, `src/ocr_service/pipeline/pdfa_generator.py` | `tests/unit/test_geometry.py`, `tests/validation/test_text_layer_alignment.py` |
| **FR-07** | Structured OCR JSON output & separate binary PDF streaming | `src/ocr_service/api/v1/jobs.py` | `tests/integration/test_api_workflow.py` |
| **FR-08** | Versioned REST endpoints, HTTP 202/409/410 status semantics, health/ready probes | `src/ocr_service/api/v1/jobs.py`, `src/ocr_service/api/v1/health.py` | `tests/integration/test_api_workflow.py`, `tests/integration/test_worker_recovery.py` |
| **FR-09** | 24h configurable retention, active-job file safety, disk quota enforcement, post-expiry metadata | `src/ocr_service/infrastructure/storage/local_storage.py`, `src/ocr_service/worker_main.py` | `tests/unit/test_retention_cleanup.py`, `tests/integration/test_worker_recovery.py` |
| **FR-10** | Explicit job state machine (`queued`, `processing`, `completed`, `failed`, `cancelled`, `expired`) | `src/ocr_service/domain/models.py`, `src/ocr_service/infrastructure/db/repository.py` | `tests/unit/test_job_lifecycle.py` |
| **FR-11** | Multi-column newspaper layout segmentation & reading order without altering word geometry | `src/ocr_service/pipeline/layout.py` | `tests/unit/test_layout_reading_order.py`, `tests/validation/test_turkish_ocr_quality.py` |
| **FR-12** | Turkish character support (`ç, Ç, ğ, Ğ, ı, I, i, İ, ö, Ö, ş, Ş, ü, Ü`) in JSON and PDF/A `/ToUnicode` CMap | `src/ocr_service/pipeline/ocr_engine.py`, `src/ocr_service/pipeline/pdfa_generator.py` | `tests/validation/test_turkish_ocr_quality.py` |
| **FR-13** | Independent OCR working DPI vs. preserved source resolution | `src/ocr_service/pipeline/preprocessor.py`, `src/ocr_service/pipeline/pdfa_generator.py` | `tests/validation/test_text_layer_alignment.py` |
| **FR-14** | Lossless image preservation (`/DCTDecode` raw JPEG passthrough & `/FlateDecode`) | `src/ocr_service/pipeline/pdfa_generator.py` | `tests/validation/test_pdfa_conformance.py` |
| **FR-15** | CPU/RAM resource governance, `OMP_THREAD_LIMIT=1`, page-by-page streaming, queue backpressure | `src/ocr_service/config.py`, `src/ocr_service/worker_main.py` | `tests/unit/test_config.py`, `tests/benchmark/run_benchmark.py` |
| **FR-16** | Transient vs. permanent failure classification, encrypted/corrupted PDF handling | `src/ocr_service/exceptions.py`, `src/ocr_service/pipeline/inspector.py` | `tests/unit/test_file_validation.py`, `tests/integration/test_multipage_and_multiimage.py` |
| **FR-17** | API key authentication, magic-byte validation, path traversal prevention, non-root container | `src/ocr_service/api/dependencies.py`, `src/ocr_service/infrastructure/storage/local_storage.py`, `Dockerfile` | `tests/unit/test_file_validation.py`, `tests/integration/test_api_workflow.py` |
| **FR-18** | Structured JSON logging, correlation IDs, per-page timings, `/metrics` | `src/ocr_service/logging_config.py`, `src/ocr_service/infrastructure/metrics/collector.py` | `tests/integration/test_api_workflow.py` |
