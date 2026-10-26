"""Unit tests for CPU-based Turkish Topic Assignment, Keyphrases, and NER with WordBox alignment."""

from __future__ import annotations

from ocr_service.domain.models import (
    BBoxPt,
    BBoxPx,
    DpiSource,
    EntityLabel,
    NlpMode,
    PageOcrResult,
    WordBox,
)
from ocr_service.pipeline.turkish_nlp import TurkishMorphNormalizer, TurkishNlpAnalyzer


def _make_word_boxes(words_text: list[str]) -> list[WordBox]:
    boxes: list[WordBox] = []
    x_cursor = 100.0
    for idx, w in enumerate(words_text):
        width_px = float(max(30, len(w) * 12))
        px = BBoxPx(left=x_cursor, top=150.0, width=width_px, height=32.0)
        pt = BBoxPt(
            x=round(x_cursor * 0.24, 2),
            y=700.0,
            width=round(width_px * 0.24, 2),
            height=8.0,
        )
        boxes.append(
            WordBox(
                text=w,
                confidence=95.0,
                bbox_working_px=px,
                bbox_source_px=px,
                bbox_pdf_pt=pt,
                line_num=1,
                word_num=idx + 1,
                reading_order_index=idx,
            )
        )
        x_cursor += width_px + 14.0
    return boxes


def test_turkish_morph_normalizer_case_and_stemming() -> None:
    """Verify Turkish dotted/dotless I case folding, circumflex normalization, and suffix stripping."""
    assert TurkishMorphNormalizer.to_lower_tr("İSTANBUL") == "istanbul"
    assert TurkishMorphNormalizer.to_lower_tr("ISPARTA") == "ısparta"
    assert TurkishMorphNormalizer.to_lower_tr("Vekâleti") == "vekaleti"

    assert TurkishMorphNormalizer.stem("Ankara'da") == "ankara"
    assert TurkishMorphNormalizer.stem("Meclislerimizin") == "meclis"
    assert TurkishMorphNormalizer.stem("Kanunlarından") == "kanun"
    assert TurkishMorphNormalizer.stem("Bakanlığının") == "bakan"


def test_turkish_parliament_and_law_topic_classification_and_ner_bbox_linking() -> None:
    """Verify topic assignment, keyphrases, and NER with union WordBox coordinates."""
    sentence = (
        "Türkiye Büyük Millet Meclisi Riyaseti 23 Nisan 1920 tarihinde "
        "Ankara Vilayeti merkezinde Gazi Mustafa Kemal Paşa başkanlığında "
        "toplanarak 123 sayılı Kanun ve Madde 4 hükümlerini müzakere etti."
    )
    tokens = sentence.split()
    word_boxes = _make_word_boxes(tokens)
    analyzer = TurkishNlpAnalyzer()

    analysis = analyzer.analyze_page(
        text=sentence,
        words=word_boxes,
        page_height_px=2000,
        mode=NlpMode.FAST,
        custom_topics=["Milli Egemenlik", "Demiryolu"],
    )

    assert analysis.engine_used == "hybrid_fast_tr"
    assert len(analysis.topics) >= 2
    topic_ids = {t.category_id for t in analysis.topics}
    assert "politics_parliament" in topic_ids
    assert "law_legislation" in topic_ids

    assert len(analysis.keywords) >= 3

    entities_by_label: dict[EntityLabel, list] = {}
    for ent in analysis.entities:
        entities_by_label.setdefault(ent.label, []).append(ent)

    # Check ORG ("Türkiye Büyük Millet Meclisi")
    assert EntityLabel.ORG in entities_by_label
    orgs = entities_by_label[EntityLabel.ORG]
    tbmm_ent = next(e for e in orgs if "Meclis" in e.text)
    assert tbmm_ent.bbox_source_px is not None
    assert tbmm_ent.bbox_pdf_pt is not None
    assert tbmm_ent.bbox_source_px.width > 100.0

    # Check DATE ("23 Nisan 1920" -> normalized "1920-04-23")
    assert EntityLabel.DATE in entities_by_label
    date_ent = entities_by_label[EntityLabel.DATE][0]
    assert date_ent.text == "23 Nisan 1920"
    assert date_ent.normalized_text == "1920-04-23"
    assert date_ent.bbox_source_px is not None

    # Check LOC ("Ankara Vilayeti")
    assert EntityLabel.LOC in entities_by_label
    loc_ent = entities_by_label[EntityLabel.LOC][0]
    assert "Ankara" in loc_ent.text
    assert loc_ent.bbox_source_px is not None

    # Check PER ("Gazi Mustafa Kemal Paşa")
    assert EntityLabel.PER in entities_by_label
    per_ent = entities_by_label[EntityLabel.PER][0]
    assert "Mustafa Kemal Paşa" in per_ent.text
    assert per_ent.bbox_source_px is not None
    assert len(per_ent.word_indices) >= 3

    # Check LAW_REF ("123 sayılı Kanun", "Madde 4")
    assert EntityLabel.LAW_REF in entities_by_label
    law_texts = [e.text for e in entities_by_label[EntityLabel.LAW_REF]]
    assert any("123" in t for t in law_texts)
    assert any("Madde 4" in t for t in law_texts)


def test_economy_topic_and_document_level_nlp_summary() -> None:
    """Verify economy topic classification and multi-page document summary aggregation."""
    analyzer = TurkishNlpAnalyzer()
    econ_text = (
        "Maliye Vekâleti bütçe komisyonu gümrük vergileri, ihracat, ithalat ve "
        "Ziraat Bankası sermaye kredileri hakkında İstanbul Ticaret Odası raporunu inceledi."
    )
    nlp_page1 = analyzer.analyze_page(
        text=econ_text,
        words=_make_word_boxes(econ_text.split()),
        page_height_px=1800,
    )
    assert nlp_page1.topics
    assert nlp_page1.topics[0].category_id == "economy_finance"

    page_res = PageOcrResult(
        page_number=1,
        width_px=1200,
        height_px=1800,
        width_pt=595.0,
        height_pt=842.0,
        dpi_x=300.0,
        dpi_y=300.0,
        dpi_source=DpiSource.DECLARED,
        rotation_deg=0,
        deskew_angle_deg=0.0,
        ocr_applied=True,
        mean_confidence=94.0,
        column_count=1,
        text=econ_text,
        words=_make_word_boxes(econ_text.split()),
        nlp=nlp_page1,
    )
    summary = analyzer.summarize_document([page_res])
    assert summary["analyzed_page_count"] == 1
    assert summary["document_topics"][0]["category_id"] == "economy_finance"
    assert len(summary["top_entities"]) >= 2
