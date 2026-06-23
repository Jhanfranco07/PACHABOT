from pathlib import Path

from app.config import get_settings
from app.core.logger import setup_logging
from app.models.schemas import DocumentChunk
from app.services.retrieval_service import RetrievalService
from app.utils.helpers import write_json


def _isolated_settings(tmp_path: Path):
    settings = get_settings()
    settings.vectorstore_dir = tmp_path / "vectorstore"
    settings.tramites_data_dir = tmp_path / "tramites"
    settings.faq_data_dir = tmp_path / "faq"
    settings.consolidated_norm_file = tmp_path / "consolidated" / "norma_consolidada.json"
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


def test_retrieval_prioritizes_short_art_alias_match(tmp_path: Path) -> None:
    settings = _isolated_settings(tmp_path)
    logger = setup_logging("INFO")
    service = RetrievalService(settings, logger)
    chunks = [
        DocumentChunk(
            chunk_id="renewal",
            document_id="requisitos",
            source_title="Ficha de renovacion",
            text="La renovacion requiere voucher y DNI.",
            tipo_contenido="requisito",
            prioridad_retrieval=4,
        ),
        DocumentChunk(
            chunk_id="article-36",
            document_id="ordenanza_227_2019",
            source_title="Ordenanza 227-2019-MDP/C",
            text=(
                "Articulo 36. El comerciante informal autorizado esta obligado al pago "
                "por concepto de SISA al valor de S/1.00 diario."
            ),
            section_title="TITULO VI | PAGO SISA",
            article_label="36",
            tipo_contenido="costo",
            vigencia="vigente",
            prioridad_retrieval=3,
            fuente="Ordenanza 227-2019-MDP/C, Articulo 36",
        ),
    ]
    service.build_index(chunks)

    results = service.search("y que dice el art. 36?", top_k=2, history=[
        # La historia no debe contaminar una consulta de articulo exacto.
    ])

    assert results
    assert results[0].article_label == "36"
    assert "SISA" in results[0].text


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


def test_retrieval_loads_structured_restricted_zones(tmp_path: Path) -> None:
    settings = _isolated_settings(tmp_path)
    settings.tramites_data_dir.mkdir(parents=True)
    write_json(
        settings.tramites_data_dir / "zonas_restringidas_comercio_ambulatorio.json",
        {
            "id": "zonas_restringidas_comercio_ambulatorio",
            "nombre": "Zonas restringidas",
            "zonas": [
                {
                    "id": "miguel_grau",
                    "ubicacion": "Jr. Miguel Grau entre Av. Victor Malasquez y Av. Manchay",
                    "restriccion": "Zona rigida para comercio ambulatorio.",
                    "articulo": "17.4",
                    "fuente": "Ordenanza 227-2019-MDP/C, Articulo 17.4",
                    "vigencia": "vigente",
                    "requires_review": False,
                }
            ],
        },
    )
    service = RetrievalService(settings, setup_logging("INFO"))
    service.build_index(service.compose_knowledge_index([]))

    results = service.search("Puedo vender en Jr Miguel Grau", top_k=2)

    assert results
    assert results[0].knowledge_layer == "zonas"
    assert results[0].article_label == "17.4"


def test_unverified_articles_57_to_64_are_not_indexed_as_evidence(tmp_path: Path) -> None:
    settings = _isolated_settings(tmp_path)
    settings.tramites_data_dir.mkdir(parents=True)
    write_json(
        settings.tramites_data_dir / "ordenanza_227_articulos_57_64.json",
        {
            "id": "ordenanza_227_articulos_57_64",
            "nombre": "Articulos no verificables",
            "articulos": [
                {
                    "articulo": "64",
                    "observacion": "No hay texto verificable cargado.",
                    "fuente": "Ordenanza 227-2019-MDP/C",
                }
            ],
        },
    )
    service = RetrievalService(settings, setup_logging("INFO"))
    service.build_index(
        service.compose_knowledge_index(
            [
                DocumentChunk(
                    chunk_id="valid",
                    document_id="ordenanza_227_2019",
                    source_title="Ordenanza 227-2019-MDP/C",
                    text="Articulo 30. Presentar solicitud para autorizacion municipal.",
                    article_label="30",
                    tipo_contenido="requisito",
                    vigencia="vigente",
                    prioridad_retrieval=3,
                )
            ]
        )
    )

    results = service.search("Que dice el articulo 64", top_k=3)

    assert all(result.article_label != "64" for result in results)


def test_retrieval_loads_structured_rubros_from_tramite(tmp_path: Path) -> None:
    settings = _isolated_settings(tmp_path)
    settings.tramites_data_dir.mkdir(parents=True)
    write_json(
        settings.tramites_data_dir / "comercio_ambulatorio.json",
        {
            "id": "tramite_comercio_ambulatorio",
            "nombre_tramite": "Autorizacion municipal para comercio ambulatorio",
            "requiere_validacion_humana": True,
            "rubros_permitidos": [
                {
                    "rubro": "Rubro 3",
                    "nombre": "Venta de productos preparados al dia",
                    "giros": [
                        {"codigo": "G004", "descripcion": "Bebidas saludables: emoliente, quinua, maca, soya."},
                        {"codigo": "G007", "descripcion": "Sandwiches."},
                    ],
                    "fuente": "Ordenanza 227-2019-MDP/C, Articulo 21",
                }
            ],
        },
    )
    service = RetrievalService(settings, setup_logging("INFO"))
    service.build_index(service.compose_knowledge_index([]))

    results = service.search("Cuales son los giros permitidos", top_k=2)

    assert results
    assert results[0].tipo_contenido == "rubro"
    assert "emoliente" in results[0].text.lower()


def test_retrieval_loads_complete_project_giros_with_count(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    settings = _isolated_settings(tmp_path)
    settings.tramites_data_dir = project_root / "data" / "tramites"

    service = RetrievalService(settings, setup_logging("INFO"))
    chunks = service.compose_knowledge_index([])
    rubro_chunk = next(
        chunk
        for chunk in chunks
        if chunk.chunk_id == "tramite-tramite_comercio_ambulatorio-rubros"
    )

    assert rubro_chunk.article_label == "21"
    assert "5 rubros y 20 giros permitidos" in rubro_chunk.text
    assert "G001" in rubro_chunk.text
    assert "G020" in rubro_chunk.text
    assert "galletas" in rubro_chunk.text.lower()


def test_retrieval_relates_common_product_to_giro(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    settings = _isolated_settings(tmp_path)
    settings.tramites_data_dir = project_root / "data" / "tramites"

    service = RetrievalService(settings, setup_logging("INFO"))
    service.build_index(service.compose_knowledge_index([]))

    results = service.search("te refieres a galletas?", top_k=3)

    assert results
    assert results[0].tipo_contenido == "rubro"
    assert "G001" in results[0].text
    assert "Golosinas" in results[0].text


def test_retrieval_loads_differentiated_requirement_cases(tmp_path: Path) -> None:
    settings = _isolated_settings(tmp_path)
    settings.tramites_data_dir.mkdir(parents=True)
    write_json(
        settings.tramites_data_dir / "requisitos_comercio_ambulatorio.json",
        {
            "tramite": "Comercio ambulatorio",
            "area_responsable": "Subgerencia",
            "fuentes": [{"nombre": "Ficha interna de requisitos de comercio ambulatorio"}],
            "tipos_tramite": [
                {
                    "id": "nuevo_ingreso_padron",
                    "nombre": "Tramite nuevo / ingreso al padron municipal",
                    "cuando_aplica": ["Cuando solicita permiso por primera vez."],
                    "frases_usuario": ["quiero vender en la calle"],
                    "requisitos": ["Foto panoramica del lugar donde desea vender."],
                    "explicacion_ciudadana": "Presenta solicitud, DNI y foto panoramica.",
                    "explicacion_simple": "Lleva DNI y foto del lugar.",
                },
                {
                    "id": "renovacion",
                    "nombre": "Renovacion",
                    "cuando_aplica": ["Cuando ya tiene autorizacion."],
                    "frases_usuario": ["tengo mi voucher"],
                    "requisitos": ["Dos fotos tamano carne.", "Copia del ultimo voucher."],
                    "explicacion_ciudadana": "Presenta formato, fotos y voucher.",
                    "explicacion_simple": "Lleva fotos y voucher.",
                },
            ],
        },
    )
    service = RetrievalService(settings, setup_logging("INFO"))
    service.build_index(service.compose_knowledge_index([]))

    nuevo = service.search("Que necesito para vender en la calle?", top_k=2)
    renovacion = service.search("Tengo mi voucher, que mas llevo?", top_k=2)

    assert nuevo[0].metadata["tipo_tramite"] == "nuevo_ingreso_padron"
    assert "foto panoramica" in nuevo[0].text.lower()
    assert renovacion[0].metadata["tipo_tramite"] == "renovacion"
    assert "voucher" in renovacion[0].text.lower()


def test_retrieval_expands_plain_language_removal_of_stand(tmp_path: Path) -> None:
    service = RetrievalService(_isolated_settings(tmp_path), setup_logging("INFO"))
    service.build_index(
        [
            DocumentChunk(
                chunk_id="revocation",
                document_id="ordenanza_108_2012",
                source_title="Ordenanza 108-2012-MDP/C",
                text="Articulo 50. Es causal de revocatoria ejercer comercio en zona rigida o incumplir condiciones de la autorizacion.",
                article_label="50",
                tipo_contenido="sancion",
                vigencia="vigente",
                prioridad_retrieval=3,
            ),
            DocumentChunk(
                chunk_id="module-definition",
                document_id="ordenanza_227_2019",
                source_title="Ordenanza 227-2019-MDP/C",
                text="Modulo es el mobiliario desmontable destinado a desarrollar la actividad comercial.",
                article_label="2",
                tipo_contenido="definicion",
                vigencia="vigente",
                prioridad_retrieval=3,
            ),
        ]
    )

    results = service.search("Me pueden quitar mi puesto?", top_k=2)

    assert results
    assert results[0].tipo_contenido == "sancion"


def test_retrieval_finds_feria_schedule_and_hygiene_with_expansion(tmp_path: Path) -> None:
    service = RetrievalService(_isolated_settings(tmp_path), setup_logging("INFO"))
    service.build_index(
        [
            DocumentChunk(
                chunk_id="feria-horario",
                document_id="ordenanza_227_2019",
                source_title="Ordenanza 227-2019-MDP/C",
                text="Articulo 61. El horario de funcionamiento de la feria sera autorizado por la Municipalidad.",
                article_label="61",
                tipo_contenido="horario",
                vigencia="vigente",
                prioridad_retrieval=3,
            ),
            DocumentChunk(
                chunk_id="feria-banos",
                document_id="ordenanza_227_2019",
                source_title="Ordenanza 227-2019-MDP/C",
                text="Articulo 62. La feria debe contar con servicios higienicos para comerciantes y usuarios.",
                article_label="62",
                tipo_contenido="requisito",
                vigencia="vigente",
                prioridad_retrieval=3,
            ),
        ]
    )

    horario = service.search("Que horario tiene una feria?", top_k=1)
    banos = service.search("Tiene que haber baños en la feria?", top_k=1)

    assert horario[0].article_label == "61"
    assert banos[0].article_label == "62"
