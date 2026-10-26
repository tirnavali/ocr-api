"""Tesseract 5 CPU OCR engine adapter with Turkish Unicode NFC normalization (FR-04, FR-12)."""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import unicodedata
from pathlib import Path

from ocr_service.config import Settings
from ocr_service.domain.models import BBoxPx, WordBox
from ocr_service.domain.ports import OcrEnginePort, OcrRecognitionOutput
from ocr_service.exceptions import OcrEngineExecutionError, PermanentProcessingError

logger = logging.getLogger(__name__)

OSD_ROTATE_RE = re.compile(r"Rotate:\s*(\d+)")


class TesseractOcrEngine(OcrEnginePort):
    """Subprocess-isolated Tesseract 5 OCR adapter with strict thread and timeout controls."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._tesseract_cmd = settings.tesseract_cmd
        self._tessdata_dir = self._resolve_tessdata_dir(settings.tessdata_dir)

    @staticmethod
    def _resolve_tessdata_dir(configured_dir: Path | None) -> Path | None:
        if configured_dir and configured_dir.exists():
            return configured_dir.resolve()
        return None

    def _build_env(self) -> dict[str, str]:
        env = os.environ.copy()
        threads = str(self._settings.ocr_threads_per_worker)
        env["OMP_THREAD_LIMIT"] = threads
        env["OMP_NUM_THREADS"] = threads
        if self._tessdata_dir is not None:
            env["TESSDATA_PREFIX"] = str(self._tessdata_dir)
        return env

    def check_health(self, required_language: str = "tur") -> tuple[bool, str]:
        """Verify tesseract binary and required language pack availability."""
        bin_path = shutil.which(self._tesseract_cmd)
        if not bin_path:
            return False, f"Tesseract binary {self._tesseract_cmd!r} not found on PATH."
        try:
            proc = subprocess.run(
                [self._tesseract_cmd, "--list-langs"],
                env=self._build_env(),
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            output = (proc.stdout or "") + "\n" + (proc.stderr or "")
            available = {line.strip() for line in output.splitlines() if line.strip()}
            required_langs = [lang.strip() for lang in required_language.split("+") if lang.strip()]
            missing = [lang for lang in required_langs if lang not in available]
            if missing:
                return False, f"Missing Tesseract language packs: {missing}"
            return True, f"Tesseract ready ({bin_path}), languages verified: {required_langs}"
        except Exception as exc:
            return False, f"Tesseract health check failed: {exc}"

    def detect_orientation(
        self,
        working_image_path: Path,
        *,
        timeout_seconds: int = 30,
    ) -> int:
        """Detect cardinal orientation (0, 90, 180, 270) using Tesseract OSD (--psm 0)."""
        cmd = [
            self._tesseract_cmd,
            str(working_image_path),
            "stdout",
            "--psm",
            "0",
        ]
        try:
            proc = subprocess.run(
                cmd,
                env=self._build_env(),
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                check=False,
            )
            if proc.returncode != 0:
                return 0
            match = OSD_ROTATE_RE.search(proc.stdout or "")
            if match:
                deg = int(match.group(1)) % 360
                if deg in (0, 90, 180, 270):
                    return deg
        except subprocess.TimeoutExpired:
            logger.warning("Tesseract OSD timed out on %s; assuming 0 deg.", working_image_path.name)
        except Exception as exc:
            logger.debug("OSD orientation detection skipped: %s", exc)
        return 0

    def recognize_page(
        self,
        working_image_path: Path,
        *,
        language: str = "tur",
        dpi: int = 300,
        psm: int = 3,
        timeout_seconds: int = 180,
    ) -> OcrRecognitionOutput:
        """Run Tesseract LSTM OCR and parse TSV word geometry and confidences."""
        cmd = [
            self._tesseract_cmd,
            str(working_image_path),
            "stdout",
            "-l",
            language,
            "--oem",
            "1",
            "--psm",
            str(psm),
            "--dpi",
            str(max(72, dpi)),
            "-c",
            "preserve_interword_spaces=1",
            "tsv",
        ]

        try:
            proc = subprocess.run(
                cmd,
                env=self._build_env(),
                capture_output=True,
                timeout=timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise OcrEngineExecutionError(
                f"Tesseract OCR timed out after {timeout_seconds}s on {working_image_path.name}."
            ) from exc
        except OSError as exc:
            raise OcrEngineExecutionError(f"Failed to invoke Tesseract: {exc}") from exc

        if proc.returncode != 0:
            stderr_msg = proc.stderr.decode("utf-8", errors="replace").strip()
            if "Failed loading language" in stderr_msg or "Error opening data file" in stderr_msg:
                raise PermanentProcessingError(
                    f"Unsupported or missing OCR language {language!r}: {stderr_msg}",
                    code="INVALID_OCR_LANGUAGE",
                )
            raise OcrEngineExecutionError(
                f"Tesseract exited with code {proc.returncode}: {stderr_msg}"
            )

        tsv_text = proc.stdout.decode("utf-8", errors="replace")
        return self._parse_tsv_output(tsv_text)

    @staticmethod
    def _parse_tsv_output(tsv_text: str) -> OcrRecognitionOutput:
        words: list[WordBox] = []
        confidences: list[float] = []

        lines = tsv_text.splitlines()
        if len(lines) <= 1:
            return OcrRecognitionOutput(words=[], raw_text="", mean_confidence=0.0)

        for row in lines[1:]:
            parts = row.split("\t")
            if len(parts) < 12:
                continue
            try:
                level = int(parts[0])
                if level != 5:  # Level 5 is word level in Tesseract TSV
                    continue
                block_num = int(parts[2])
                par_num = int(parts[3])
                line_num = int(parts[4])
                word_num = int(parts[5])
                left = float(parts[6])
                top = float(parts[7])
                width = float(parts[8])
                height = float(parts[9])
                conf = float(parts[10])
                raw_word = parts[11].strip()
            except ValueError:
                continue

            if not raw_word or conf < 0:
                continue

            # Normalize Turkish characters to canonical NFC form (FR-12)
            normalized_word = unicodedata.normalize("NFC", raw_word)
            if width <= 0 or height <= 0:
                continue

            bbox = BBoxPx(left=left, top=top, width=width, height=height)
            words.append(
                WordBox(
                    text=normalized_word,
                    confidence=conf,
                    bbox_working_px=bbox,
                    bbox_source_px=bbox,
                    block_num=block_num,
                    par_num=par_num,
                    line_num=line_num,
                    word_num=word_num,
                )
            )
            confidences.append(conf)

        mean_conf = float(sum(confidences) / len(confidences)) if confidences else 0.0
        raw_text = " ".join(w.text for w in words)
        return OcrRecognitionOutput(
            words=words,
            raw_text=raw_text,
            detected_orientation_deg=0,
            mean_confidence=mean_conf,
        )
