"""BERTurk token-classification NER (PER/ORG/LOC) on CPU via ONNX Runtime.

The model directory holds a Hugging Face ``tokenizer.json``, ``config.json`` (id2label)
and an ONNX export of a BertForTokenClassification model, normally the dynamic INT8
quantization produced by ``scripts/prepare_ner_model.py``. Torch is not needed.
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# BERT position limit is 512 incl. [CLS]/[SEP]; leave headroom for long OCR words.
MAX_SUBWORDS_PER_CHUNK = 448


@dataclass(frozen=True, slots=True)
class WordTag:
    """BIO tag of one input word (from its first sub-word) and that tag's probability."""

    tag: str
    prob: float


class OnnxBertNer:
    """Word-level BIO tagging with a BERT token-classification ONNX model."""

    def __init__(self, model_dir: Path, model_file: str = "model_int8.onnx", threads: int = 1) -> None:
        import onnxruntime as ort
        from tokenizers import Tokenizer

        model_path = model_dir / model_file
        if not model_path.exists():
            raise FileNotFoundError(f"NER model not found: {model_path}")
        config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
        self._id2label = {int(k): v for k, v in config["id2label"].items()}
        self._tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        self._tokenizer.no_truncation()
        self._tokenizer.no_padding()

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = max(1, threads)
        opts.inter_op_num_threads = 1
        self._session = ort.InferenceSession(str(model_path), opts, providers=["CPUExecutionProvider"])
        self._input_names = {i.name for i in self._session.get_inputs()}
        self.model_name = f"{model_dir.name}/{model_file}"

    def tag_words(self, words: list[str]) -> list[WordTag]:
        """Tag each word; words are fed pre-split so tags map 1:1 back to OCR word boxes."""
        tags: list[WordTag] = []
        for start, end in self._chunks(words):
            tags.extend(self._tag_chunk(words[start:end]))
        return tags

    def _chunks(self, words: list[str]) -> list[tuple[int, int]]:
        # Greedy chunks below the sub-word limit; per-word sub-word counts come from one
        # batched encode so chunking stays cheap on 1000+ word newspaper pages.
        counts = [len(e.ids) - 2 for e in self._tokenizer.encode_batch(words)] if words else []
        chunks: list[tuple[int, int]] = []
        start, used = 0, 0
        for i, c in enumerate(counts):
            c = max(1, c)
            if used + c > MAX_SUBWORDS_PER_CHUNK and i > start:
                chunks.append((start, i))
                start, used = i, 0
            used += c
        if start < len(words):
            chunks.append((start, len(words)))
        return chunks

    def _tag_chunk(self, words: list[str]) -> list[WordTag]:
        enc = self._tokenizer.encode(words, is_pretokenized=True)
        ids = enc.ids[: MAX_SUBWORDS_PER_CHUNK + 2]
        n = len(ids)
        feed = {
            "input_ids": np.array([ids], dtype=np.int64),
            "attention_mask": np.ones((1, n), dtype=np.int64),
            "token_type_ids": np.zeros((1, n), dtype=np.int64),
        }
        logits = self._session.run(None, {k: v for k, v in feed.items() if k in self._input_names})[0][0]
        shifted = logits - logits.max(axis=-1, keepdims=True)
        probs = np.exp(shifted) / np.exp(shifted).sum(axis=-1, keepdims=True)

        result = [WordTag("O", 1.0)] * len(words)
        seen: set[int] = set()
        for pos, word_id in enumerate(enc.word_ids[:n]):
            if word_id is None or word_id in seen:
                continue
            seen.add(word_id)  # first sub-word decides the word's tag
            label_id = int(probs[pos].argmax())
            result[word_id] = WordTag(self._id2label[label_id], float(probs[pos][label_id]))
        return result


class LazyOnnxBertNer:
    """Loads the model on first use; a missing or broken model disables it once, loudly."""

    def __init__(self, model_dir: Path | None, model_file: str, threads: int) -> None:
        self._model_dir = model_dir
        self._model_file = model_file
        self._threads = threads
        self._model: OnnxBertNer | None = None
        self._failed = model_dir is None
        self._lock = threading.Lock()

    @property
    def configured(self) -> bool:
        return self._model_dir is not None

    def get(self) -> OnnxBertNer | None:
        if self._model is not None or self._failed:
            return self._model
        with self._lock:
            if self._model is None and not self._failed:
                try:
                    assert self._model_dir is not None
                    self._model = OnnxBertNer(self._model_dir, self._model_file, self._threads)
                    logger.info("Loaded ONNX NER model", extra={"model": self._model.model_name})
                except Exception:
                    self._failed = True
                    logger.exception(
                        "ONNX NER model could not be loaded; falling back to rule-based NER",
                        extra={"model_dir": str(self._model_dir)},
                    )
        return self._model
