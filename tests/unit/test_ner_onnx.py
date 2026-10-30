"""nlp_mode=onnx: BERTurk entities merged with rule DATE/LAW_REF, honest engine label, fallback."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from ocr_service.domain.models import EntityLabel, NlpMode
from ocr_service.pipeline.ner_onnx import LazyOnnxBertNer, WordTag
from ocr_service.pipeline.turkish_nlp import TurkishNlpAnalyzer

TEXT = "Hasan Pulur 23 Nisan 1920 günü Bandırma Belediyesi önünde Ankara Kanunu tartıştı"


class _FakeModel:
    """Tags words by a fixed table, like the real model does word by word."""

    def __init__(self, tags: dict[str, str]) -> None:
        self._tags = tags

    def tag_words(self, words: list[str]) -> list[WordTag]:
        return [WordTag(self._tags.get(w, "O"), 0.9) for w in words]


class _FakeLazy:
    def __init__(self, model: _FakeModel | None) -> None:
        self._model = model

    def get(self) -> _FakeModel | None:
        return self._model


def _entities(analyzer: TurkishNlpAnalyzer, mode: NlpMode):
    res = analyzer.analyze_page(TEXT, [], mode=mode)
    return res, {(e.label, e.text) for e in res.entities}


def test_onnx_mode_uses_model_for_names_and_keeps_rule_dates() -> None:
    model = _FakeModel(
        {
            "Hasan": "B-PER",
            "Pulur": "I-PER",
            "Bandırma": "B-ORG",
            "Belediyesi": "I-ORG",
            "Ankara": "B-LOC",
            "1920": "B-LOC",  # the model's guess on a date token must not override the rule DATE
        }
    )
    res, ents = _entities(TurkishNlpAnalyzer(onnx_ner=_FakeLazy(model)), NlpMode.ONNX)

    assert res.engine_used == "hybrid_onnx_tr"
    assert (EntityLabel.PER, "Hasan Pulur") in ents
    assert (EntityLabel.ORG, "Bandırma Belediyesi") in ents
    assert (EntityLabel.LOC, "Ankara") in ents
    assert any(label == EntityLabel.DATE and "1920" in text for label, text in ents)
    assert not any(label == EntityLabel.LOC and "1920" in text for label, text in ents)
    by_text = {e.text: e for e in res.entities}
    assert by_text["Hasan Pulur"].source == "onnx_berturk"
    assert by_text["Hasan Pulur"].confidence == pytest.approx(0.9)


def test_stray_inside_tag_starts_a_new_entity() -> None:
    model = _FakeModel({"Hasan": "B-PER", "Pulur": "I-PER", "Ankara": "I-LOC"})
    _, ents = _entities(TurkishNlpAnalyzer(onnx_ner=_FakeLazy(model)), NlpMode.ONNX)
    assert (EntityLabel.LOC, "Ankara") in ents


def test_fast_mode_never_calls_the_model() -> None:
    class _Exploding:
        def get(self):  # noqa: ANN202
            raise AssertionError("model must not be loaded in fast mode")

    res, _ = _entities(TurkishNlpAnalyzer(onnx_ner=_Exploding()), NlpMode.FAST)
    assert res.engine_used == "hybrid_fast_tr"


def test_onnx_mode_without_model_reports_rule_engine(tmp_path: Path) -> None:
    # A configured but missing model directory must not break OCR jobs.
    lazy = LazyOnnxBertNer(tmp_path / "yok", "model_int8.onnx", threads=1)
    res, _ = _entities(TurkishNlpAnalyzer(onnx_ner=lazy), NlpMode.ONNX)
    assert res.engine_used == "hybrid_fast_tr"
    assert lazy.get() is None

    res_none, _ = _entities(TurkishNlpAnalyzer(), NlpMode.ONNX)
    assert res_none.engine_used == "hybrid_fast_tr"


@pytest.mark.skipif(
    not os.environ.get("NER_MODEL_DIR"), reason="set NER_MODEL_DIR to run against the real BERTurk model"
)
def test_real_model_tags_long_pages_word_by_word() -> None:
    lazy = LazyOnnxBertNer(Path(os.environ["NER_MODEL_DIR"]), "model_int8.onnx", threads=1)
    model = lazy.get()
    assert model is not None
    words = ("Hasan Pulur Bandırma'da Milliyet Gazetesi için yazdı . " * 120).split()
    tags = model.tag_words(words)  # > 1 chunk of 448 sub-words
    assert len(tags) == len(words)
    assert tags[0].tag == "B-PER" and tags[1].tag == "I-PER"
    assert tags[-8].tag == "B-PER"
