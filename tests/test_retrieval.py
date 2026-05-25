from pathlib import Path

from app.config import get_settings
from app.core.logger import setup_logging
from app.models.schemas import DocumentChunk
from app.services.retrieval_service import RetrievalService


def _isolated_settings(tmp_path: Path):
    settings = get_settings()
    settings.vectorstore_dir = tmp_path / "vectorstore"
    settings.processed_chunks_file = tmp_path / "processed" / "chunks.json"
    settings.vectorizer_file = tmp_path / "vectorstore" / "tfidf_vectorizer.joblib"
    settings.matrix_file = tmp_path / "vectorstore" / "tfidf_matrix.joblib"
    return settings


def test_retrieval_returns_relevant_chunk(tmp_path: Path) -> None:
    settings = _isolated_settings(tmp_path)
    logger = setup_logging("INFO")
    service = RetrievalService(settings, logger)
    chunks = [
        DocumentChunk(
            chunk_id="doc-001",
            document_id="ordenanza_108_2012",
            source_title="Ordenanza 108-2012-MDP/C",
            text="La autorización municipal exige requisitos y documentación básica.",
            section_title="TÍTULO III | DE LA AUTORIZACION MUNICIPAL",
            article_label="6",
            metadata={},
        ),
        DocumentChunk(
            chunk_id="doc-002",
            document_id="ordenanza_227_2019",
            source_title="Ordenanza 227-2019-MDP/C",
            text="Las zonas rígidas prohíben el comercio ambulatorio en vías determinadas.",
            section_title="TÍTULO IV | REGULACION DEL COMERCIO INFORMAL EN LA VIA PUBLICA",
            article_label="13",
            metadata={},
        ),
    ]
    service.build_index(chunks)
    results = service.search("que requisitos necesito para la autorizacion", top_k=2)

    assert results
    assert results[0].document_id == "ordenanza_108_2012"


def test_retrieval_prioritizes_exact_article_match(tmp_path: Path) -> None:
    settings = _isolated_settings(tmp_path)
    logger = setup_logging("INFO")
    service = RetrievalService(settings, logger)
    chunks = [
        DocumentChunk(
            chunk_id="doc-001",
            document_id="ordenanza_108_2012",
            source_title="Ordenanza 108-2012-MDP/C",
            text="Artículo 6°. Para ejercer el comercio informal se requiere autorización.",
            section_title="TÍTULO III | DE LA AUTORIZACION MUNICIPAL",
            article_label="6",
            metadata={},
        ),
        DocumentChunk(
            chunk_id="doc-002",
            document_id="ordenanza_108_2012",
            source_title="Ordenanza 108-2012-MDP/C",
            text="Artículo 7°. La autorización municipal es personal e intransferible.",
            section_title="TÍTULO III | DE LA AUTORIZACION MUNICIPAL",
            article_label="7",
            metadata={},
        ),
    ]
    service.build_index(chunks)
    results = service.search("que dice el articulo 7", top_k=2)

    assert results
    assert results[0].article_label == "7"


def test_retrieval_prefers_definition_over_preamble(tmp_path: Path) -> None:
    settings = _isolated_settings(tmp_path)
    logger = setup_logging("INFO")
    service = RetrievalService(settings, logger)
    chunks = [
        DocumentChunk(
            chunk_id="doc-001",
            document_id="ordenanza_108_2012",
            source_title="Ordenanza 108-2012-MDP/C",
            text="Es política de la gestión municipal regular el comercio ambulatorio en el distrito.",
            section_title="PREAMBULO",
            article_label="",
            metadata={},
        ),
        DocumentChunk(
            chunk_id="doc-002",
            document_id="ordenanza_108_2012",
            source_title="Ordenanza 108-2012-MDP/C",
            text="Artículo 2°.- Se entiende por comercio en la vía pública la actividad económica autorizada y temporal.",
            section_title="TÍTULO I | DEFINICIONES",
            article_label="2",
            metadata={},
        ),
    ]
    service.build_index(chunks)
    results = service.search("que es el comercio ambulatorio", top_k=2)

    assert results
    assert results[0].section_title.startswith("TÍTULO I")


def test_chunks_prioridad_cero_no_aparecen_en_resultados(tmp_path: Path) -> None:
    settings = _isolated_settings(tmp_path)
    service = RetrievalService(settings, setup_logging("INFO"))
    service.build_index(
        [
            DocumentChunk(
                chunk_id="noise",
                document_id="doc",
                source_title="Ordenanza",
                text="CONSIDERANDO requisitos documentos comercio ambulatorio.",
                tipo_contenido="considerando",
                prioridad_retrieval=0,
            ),
            DocumentChunk(
                chunk_id="valid",
                document_id="doc",
                source_title="Ordenanza",
                text="Artículo 30. Presentar documentos para obtener autorización.",
                article_label="30",
                tipo_contenido="requisito",
                vigencia="vigente",
                prioridad_retrieval=3,
            ),
        ]
    )

    results = service.search("Qué documentos necesito para comercio ambulatorio", top_k=2)

    assert results
    assert all(result.prioridad_retrieval > 0 for result in results)
    assert results[0].chunk_id == "valid"


def test_consulta_vigente_no_devuelve_historicos_como_primero(tmp_path: Path) -> None:
    service = RetrievalService(_isolated_settings(tmp_path), setup_logging("INFO"))
    service.build_index(
        [
            DocumentChunk(
                chunk_id="old",
                document_id="ordenanza_108_2012",
                source_title="Ordenanza 108-2012-MDP/C",
                text="Artículo 5. La autorización tiene una vigencia anterior.",
                article_label="5",
                tipo_contenido="procedimiento",
                vigencia="historico",
                prioridad_retrieval=1,
            ),
            DocumentChunk(
                chunk_id="current",
                document_id="ordenanza_227_2019",
                source_title="Ordenanza 227-2019-MDP/C",
                text="Artículo 5. El plazo de vigencia de la autorización es un año.",
                article_label="5",
                tipo_contenido="procedimiento",
                vigencia="vigente",
                prioridad_retrieval=3,
            ),
        ]
    )

    results = service.search("Cuánto dura la autorización", top_k=2)

    assert results[0].chunk_id == "current"
    assert all(result.vigencia != "historico" for result in results)


def test_consulta_zonas_devuelve_articulo_con_zona(tmp_path: Path) -> None:
    service = RetrievalService(_isolated_settings(tmp_path), setup_logging("INFO"))
    service.build_index(
        [
            DocumentChunk(
                chunk_id="zone",
                document_id="ordenanza_108_2012",
                source_title="Ordenanza 108-2012-MDP/C",
                text="Artículo 13. Son zonas rígidas aquellas donde no se autoriza vender.",
                article_label="13",
                tipo_contenido="zona",
                vigencia="vigente",
                prioridad_retrieval=3,
            ),
        ]
    )

    results = service.search("Dónde no puedo vender", top_k=1)

    assert results[0].tipo_contenido in {"zona", "prohibicion"}
