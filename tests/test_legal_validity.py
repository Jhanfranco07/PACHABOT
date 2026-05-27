from pathlib import Path

from app.core.logger import setup_logging
from app.documents.cleaner import clean_legal_document
from app.documents.consolidator import LegalConsolidator, article_status_map


def _consolidate_real_sources(tmp_path: Path) -> dict:
    raw_dir = Path(__file__).resolve().parent.parent / "data" / "raw"
    cleaned = {
        path.stem: clean_legal_document(path.read_text(encoding="utf-8-sig"))
        for path in raw_dir.glob("ordenanza_*.txt")
    }
    return LegalConsolidator(setup_logging("INFO")).consolidate(
        cleaned,
        output_dir=tmp_path / "consolidated",
    )


def test_articulos_modificados_tienen_trazabilidad() -> None:
    statuses = article_status_map("ordenanza_108_2012")

    for label in ("2", "5", "6", "7", "30", "36", "38", "52", "54"):
        assert statuses[label]["vigencia"] == "historico"
        assert statuses[label]["modificado_por"] != ""


def test_consolidador_registra_corpus_incompleto(tmp_path: Path) -> None:
    payload = _consolidate_real_sources(tmp_path)

    assert payload["validation"]["corpus_complete"] is False
    assert any(article["article_label"] == "57" for article in payload["unverified_articles"])
    assert any("posterior" in item.lower() for item in payload["validation"]["unverified_scope"])


def test_norma_consolidada_no_contiene_articulos_historicos(tmp_path: Path) -> None:
    payload = _consolidate_real_sources(tmp_path)

    assert payload["articles"]
    assert all(
        article["vigencia"] in {"vigente", "vigente_con_observacion"}
        for article in payload["articles"]
    )
    assert any(
        article["article_label"] == "23"
        and article["vigencia"] == "vigente_con_observacion"
        and article["requires_review"] is True
        for article in payload["articles"]
    )


def test_no_existen_chunks_con_vigencia_contradictoria(tmp_path: Path) -> None:
    payload = _consolidate_real_sources(tmp_path)
    status_by_label: dict[str, set[str]] = {}
    for article in payload["articles"]:
        status_by_label.setdefault(article["article_label"], set()).add(article["vigencia"])

    assert all(len(statuses) == 1 for statuses in status_by_label.values())
    assert status_by_label["23"] == {"vigente_con_observacion"}
    assert status_by_label["39"] == {"vigente_con_observacion"}


def test_articulo_17_consolidado_reemplaza_distancia_antigua(tmp_path: Path) -> None:
    payload = _consolidate_real_sources(tmp_path)
    article_17 = next(article for article in payload["articles"] if article["article_label"] == "17")

    assert "20.0 m" in article_17["text"]
    assert "25 m" not in article_17["text"]


def test_articulo_57_esta_excluido_por_truncamiento(tmp_path: Path) -> None:
    payload = _consolidate_real_sources(tmp_path)
    article_57 = next(article for article in payload["unverified_articles"] if article["article_label"] == "57")

    assert article_57["exclude_from_retrieval"] is True
    assert article_57["requires_review"] is True
