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
    assert all(article["vigencia"] == "vigente" for article in payload["articles"])


def test_no_existen_chunks_con_vigencia_contradictoria(tmp_path: Path) -> None:
    payload = _consolidate_real_sources(tmp_path)
    status_by_label: dict[str, set[str]] = {}
    for article in payload["articles"]:
        status_by_label.setdefault(article["article_label"], set()).add(article["vigencia"])

    assert all(statuses == {"vigente"} for statuses in status_by_label.values())
