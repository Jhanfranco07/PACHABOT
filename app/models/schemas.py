from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from app.models.domain import QueryIntent


@dataclass(slots=True)
class DocumentChunk:
    chunk_id: str
    document_id: str
    source_title: str
    text: str
    section_title: str = ""
    article_label: str = ""
    # CAMBIO FASE 7.1 — Extender la evidencia con metadatos normativos recuperables.
    # Motivo: impedir que texto historico o de bajo valor compita con norma vigente.
    # Riesgo mitigado: los valores por defecto mantienen compatibles los chunks existentes.
    normalized_text: str = ""
    tipo_contenido: str = "disposicion"
    user_intents: list[str] = field(default_factory=lambda: ["ninguno"])
    vigencia: str = "vigente"
    modificado_por: str = ""
    prioridad_retrieval: int = 2
    fuente: str = ""
    tramite_relacionado: str = ""
    knowledge_layer: str = "normativa"
    exclude_from_retrieval: bool = False
    requires_review: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RetrievedChunk:
    chunk_id: str
    document_id: str
    source_title: str
    text: str
    score: float
    section_title: str = ""
    article_label: str = ""
    normalized_text: str = ""
    tipo_contenido: str = "disposicion"
    user_intents: list[str] = field(default_factory=lambda: ["ninguno"])
    vigencia: str = "vigente"
    modificado_por: str = ""
    prioridad_retrieval: int = 2
    fuente: str = ""
    tramite_relacionado: str = ""
    knowledge_layer: str = "normativa"
    exclude_from_retrieval: bool = False
    requires_review: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RoutedQuery:
    original_query: str
    normalized_query: str
    intent: QueryIntent
    in_domain: bool
    matched_keywords: list[str] = field(default_factory=list)


@dataclass(slots=True)
class AnswerPayload:
    answer: str
    sources: list[str]
    intent: QueryIntent
    in_domain: bool
    confidence: float
    used_llm: bool
    response_origin: str = "fallback"
    evidence: list["EvidenceItem"] = field(default_factory=list)
    evidence_warning: str = ""
    confidence_level: str = "none"


@dataclass(slots=True)
class EvidenceItem:
    source: str
    source_type: str
    score: float
    excerpt: str
    article_label: str = ""
    requires_review: bool = False


@dataclass(slots=True)
class EvidenceAssessment:
    sufficient: bool
    confidence_level: str
    warning: str = ""
    items: list[EvidenceItem] = field(default_factory=list)


@dataclass(slots=True)
class ConversationTurn:
    role: str
    text: str
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class KnowledgeBundle:
    original_query: str
    effective_query: str
    chunks: list[RetrievedChunk]
    confidence: float
    sources: list[str]
    notes: list[str] = field(default_factory=list)
    search_queries: list[str] = field(default_factory=list)
    tools_used: list[str] = field(default_factory=list)
