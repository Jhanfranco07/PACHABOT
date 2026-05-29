import json
from pathlib import Path

from app.channels.schemas import IncomingChatMessage
from app.config import Settings
from app.core.logger import setup_logging
from app.models.domain import QueryIntent
from app.models.schemas import AnswerPayload, KnowledgeBundle, RetrievedChunk, RoutedQuery
from app.services.evidence_service import EvidenceService


def _knowledge(*chunks: RetrievedChunk) -> KnowledgeBundle:
    return KnowledgeBundle(
        original_query="Que requisitos necesito?",
        effective_query="Que requisitos necesito para comercio ambulatorio?",
        chunks=list(chunks),
        confidence=chunks[0].score if chunks else 0.0,
        sources=[chunk.fuente for chunk in chunks],
        search_queries=["requisitos comercio ambulatorio"],
    )


def test_evidence_service_marks_citizen_record_as_high_confidence() -> None:
    service = EvidenceService(Settings(), setup_logging("INFO"))
    assessment = service.assess(
        _knowledge(
            RetrievedChunk(
                chunk_id="tramite-1",
                document_id="tramite_comercio_ambulatorio",
                source_title="Ficha de tramite",
                text="Presentar solicitud y dos fotografias tamano carne.",
                score=0.72,
                fuente="Ordenanza 227-2019-MDP/C, Articulo 30",
                knowledge_layer="tramites",
                tipo_contenido="requisito",
            )
        )
    )

    assert assessment.sufficient is True
    assert assessment.confidence_level == "high"
    assert assessment.items[0].source_type == "tramite"
    assert assessment.items[0].score == 0.72
    assert "Presentar solicitud" in assessment.items[0].excerpt


def test_evidence_service_blocks_unverified_or_weak_support() -> None:
    settings = Settings()
    settings.retrieval_min_score = 0.35
    service = EvidenceService(settings, setup_logging("INFO"))
    assessment = service.assess(
        _knowledge(
            RetrievedChunk(
                chunk_id="article-57",
                document_id="ordenanza_227_2019",
                source_title="Ordenanza 227-2019-MDP/C",
                text="Texto truncado.",
                score=0.90,
                vigencia="no_verificable",
                exclude_from_retrieval=True,
                knowledge_layer="chunks",
            )
        )
    )

    assert assessment.sufficient is False
    assert assessment.confidence_level == "low"
    assert "coincidencias" in assessment.warning


def test_evidence_service_writes_optional_debug_trace(tmp_path: Path) -> None:
    settings = Settings()
    settings.rag_debug_trace = True
    settings.runtime_debug_dir = tmp_path / "runtime" / "debug"
    service = EvidenceService(settings, setup_logging("INFO"))
    knowledge = _knowledge(
        RetrievedChunk(
            chunk_id="faq-1",
            document_id="faq_comercio_ambulatorio",
            source_title="FAQ",
            text="El costo debe verificarse en el TUPA vigente.",
            score=0.64,
            fuente="Ordenanza 227-2019-MDP/C, Articulo 30",
            knowledge_layer="faq",
        )
    )
    assessment = service.assess(knowledge)
    payload = AnswerPayload(
        answer="El costo debe verificarse en el TUPA vigente.",
        sources=["Ordenanza 227-2019-MDP/C, Articulo 30"],
        intent=QueryIntent.PAGOS_SISA,
        in_domain=True,
        confidence=0.64,
        used_llm=False,
        evidence=assessment.items,
        confidence_level=assessment.confidence_level,
    )
    routed = RoutedQuery(
        original_query="Cuanto cuesta?",
        normalized_query="cuanto cuesta",
        intent=QueryIntent.PAGOS_SISA,
        in_domain=True,
    )

    service.write_trace(
        IncomingChatMessage("web", "trace-demo", "u1", "Cuanto cuesta?"),
        routed,
        knowledge,
        assessment,
        payload,
    )

    trace = json.loads((settings.runtime_debug_dir / "web_trace-demo.jsonl").read_text(encoding="utf-8"))
    assert trace["sufficient"] is True
    assert trace["confidence_level"] == "high"
    assert trace["evidence"][0]["source_type"] == "faq"
