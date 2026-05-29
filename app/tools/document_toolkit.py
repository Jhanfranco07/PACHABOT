from __future__ import annotations

import re

from app.config import Settings
from app.models.domain import QueryIntent
from app.models.schemas import ConversationTurn, KnowledgeBundle, RetrievedChunk, RoutedQuery
from app.services.query_rewriter import QueryRewriter
from app.services.retrieval_service import RetrievalService
from app.utils.query_expansion import expansion_terms, related_search_queries
from app.utils.text_cleaner import normalize_for_search


ARTICLE_PATTERN = re.compile(r"art[ií]culo\s+([0-9]+(?:\.[0-9]+)?[A-Z]?)", re.IGNORECASE)
COMMON_TYPO_HINTS = {
    "ke": "que",
    "qe": "que",
    "q": "que",
    "kuanto": "cuanto",
    "kanto": "cuanto",
    "komo": "como",
    "dnd": "donde",
    "sona": "zona",
    "sissa": "sisa",
    "modlo": "modulo",
}

class DocumentToolkit:
    """Prepare queries, recover document evidence and package it for answer generation."""

    def __init__(
        self,
        settings: Settings,
        retrieval_service: RetrievalService,
        query_rewriter: QueryRewriter,
        logger,
    ) -> None:
        self.settings = settings
        self.retrieval_service = retrieval_service
        self.query_rewriter = query_rewriter
        self.logger = logger.getChild("document_toolkit")

    def prepare_query(self, question: str) -> tuple[str, list[str]]:
        """Normalize only the most obvious typos before retrieval."""

        corrected_question = self._correct_spelling(question)
        notes: list[str] = []
        if normalize_for_search(corrected_question) != normalize_for_search(question):
            notes.append(
                f"Se interpreto la consulta corrigiendo posibles errores de escritura: {corrected_question}"
            )
        return corrected_question, notes

    def gather_knowledge(
        self,
        question: str,
        routed_query: RoutedQuery,
        history: list[ConversationTurn],
        *,
        original_question: str | None = None,
        preparation_notes: list[str] | None = None,
    ) -> KnowledgeBundle:
        """Collect document evidence using a rewritten query plus lightweight multi-search."""

        original_question = original_question or question
        preparation_notes = preparation_notes or []
        effective_query = self.query_rewriter.rewrite(question, history)
        search_queries = self._build_search_queries(question, effective_query, routed_query)

        merged: dict[str, RetrievedChunk] = {}
        for search_query in search_queries:
            results = self.retrieval_service.search(
                search_query,
                top_k=self.settings.retrieval_max_results,
                history=history,
            )
            for item in results:
                previous = merged.get(item.chunk_id)
                if previous is None or item.score > previous.score:
                    merged[item.chunk_id] = item

        if routed_query.intent == QueryIntent.NORMATIVA:
            for item in self._normative_identity_evidence():
                merged[item.chunk_id] = item

        final_chunks = self._finalize_chunks(
            question,
            effective_query,
            list(merged.values()),
            routed_query.intent,
        )
        confidence = final_chunks[0].score if final_chunks else 0.0
        sources = list(dict.fromkeys(self._format_source(chunk) for chunk in final_chunks))

        notes = list(preparation_notes)
        if normalize_for_search(effective_query) != normalize_for_search(question):
            notes.append("La consulta se reformulo usando el historial reciente.")
        if any(chunk.requires_review for chunk in final_chunks):
            notes.append("La evidencia recuperada contiene informacion pendiente de validacion humana.")
        expanded_terms = expansion_terms(effective_query, routed_query.intent)
        if expanded_terms:
            notes.append(f"Terminos expandidos: {', '.join(expanded_terms[:12])}")

        return KnowledgeBundle(
            original_query=original_question,
            effective_query=effective_query,
            chunks=final_chunks,
            confidence=confidence,
            sources=sources,
            notes=notes,
            search_queries=search_queries,
            tools_used=["query_rewrite", "document_search"],
        )

    def _build_search_queries(
        self,
        question: str,
        effective_query: str,
        routed_query: RoutedQuery,
    ) -> list[str]:
        """Create a small set of search queries without overfitting to hand-written rules."""

        queries = [question, effective_query]
        normalized_question = normalize_for_search(question)

        article_match = ARTICLE_PATTERN.search(question)
        if article_match:
            queries.append(f"Articulo {article_match.group(1)} comercio ambulatorio")

        if any(pattern in normalized_question for pattern in ("que es", "defin", "se entiende", "concepto")):
            queries.append(f"{effective_query} definiciones")

        queries.extend(related_search_queries(question, routed_query.intent))
        queries.extend(related_search_queries(effective_query, routed_query.intent))

        return list(dict.fromkeys(query.strip() for query in queries if query.strip()))

    def _finalize_chunks(
        self,
        question: str,
        effective_query: str,
        chunks: list[RetrievedChunk],
        intent: QueryIntent,
    ) -> list[RetrievedChunk]:
        """Stabilize retrieved results with a light final rerank."""

        normalized_question = normalize_for_search(question)
        normalized_effective_query = normalize_for_search(effective_query)
        article_match = ARTICLE_PATTERN.search(effective_query)
        article_label = article_match.group(1).upper() if article_match else ""
        asks_definition = any(
            pattern in normalized_question
            for pattern in ("que es", "defin", "se entiende", "concepto")
        )
        strong_topic_markers = {
            "sisa": "sisa",
            "modulo": "modulo",
            "feria": "feria",
            "zona rigida": "zona rigida",
            "autoriz": "autoriz",
            "giro": "giro",
            "giros": "giro",
            "rubro": "rubro",
            "rubros": "rubro",
        }

        reranked: list[RetrievedChunk] = []
        if article_label:
            chunks = [
                chunk
                for chunk in chunks
                if chunk.article_label.upper() == article_label
                and not chunk.exclude_from_retrieval
            ]
        for chunk in chunks:
            bonus = 0.0
            normalized_text = normalize_for_search(chunk.text)
            normalized_section = normalize_for_search(chunk.section_title)

            if article_label and chunk.article_label and chunk.article_label.upper() == article_label:
                bonus += 0.45
            if asks_definition and (
                "definiciones" in normalized_section or "se entiende por" in normalized_text
            ):
                bonus += 0.20

            for marker, text_marker in strong_topic_markers.items():
                if marker in normalized_effective_query:
                    if text_marker in normalized_text or text_marker in normalized_section:
                        bonus += 0.90
                    else:
                        bonus -= 0.60

            reranked.append(
                RetrievedChunk(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    source_title=chunk.source_title,
                    text=chunk.text,
                    score=chunk.score + bonus,
                    section_title=chunk.section_title,
                    article_label=chunk.article_label,
                    normalized_text=chunk.normalized_text,
                    tipo_contenido=chunk.tipo_contenido,
                    user_intents=chunk.user_intents,
                    vigencia=chunk.vigencia,
                    modificado_por=chunk.modificado_por,
                    prioridad_retrieval=chunk.prioridad_retrieval,
                    fuente=chunk.fuente,
                    tramite_relacionado=chunk.tramite_relacionado,
                    knowledge_layer=chunk.knowledge_layer,
                    exclude_from_retrieval=chunk.exclude_from_retrieval,
                    requires_review=chunk.requires_review,
                    metadata=chunk.metadata,
                )
            )

        reranked.sort(key=lambda item: item.score, reverse=True)

        if intent == QueryIntent.DEFINICION or asks_definition:
            current_definitions = [
                chunk
                for chunk in reranked
                if chunk.tipo_contenido == "definicion" and chunk.vigencia == "vigente"
            ]
            if current_definitions:
                return current_definitions[: self.settings.retrieval_top_k]

        asks_cost_and_requirements = any(
            term in normalized_effective_query or term in normalized_question
            for term in ("cuanto cuesta", "costo", "monto", "tasa", "pago")
        ) and any(
            term in normalized_effective_query or term in normalized_question
            for term in ("requisit", "documento", "que necesito")
        )
        if intent in {
            QueryIntent.REQUISITOS_NUEVO,
            QueryIntent.REQUISITOS_RENOVACION,
            QueryIntent.REQUISITOS_AMBIGUO,
        }:
            requirement_case_chunks = self._requirement_case_chunks(
                reranked,
                intent,
                include_cost=asks_cost_and_requirements,
            )
            if requirement_case_chunks:
                return requirement_case_chunks[: self.settings.retrieval_top_k]
            if intent == QueryIntent.REQUISITOS_AMBIGUO:
                return []

        if asks_cost_and_requirements:
            useful = [
                chunk
                for chunk in reranked
                if chunk.tipo_contenido in {"requisito", "costo"}
                or chunk.knowledge_layer in {"tramites", "faq"}
            ]
            if useful:
                return useful[: self.settings.retrieval_top_k]

        # CAMBIO FASE OLLAMA 5 — Exigir evidencia monetaria para responder costos.
        # Motivo: una coincidencia con la palabra tramite no acredita un precio vigente.
        # Riesgo mitigado: las consultas explicitas de SISA conservan sus chunks de monto.
        if intent == QueryIntent.PAGOS_SISA:
            topic_text = normalize_for_search(f"{question} {effective_query}")
            if "sisa" not in topic_text and not any(
                topic in topic_text for topic in ("autoriz", "permiso", "feria", "modulo")
            ):
                return []

            cost_chunks = [
                chunk
                for chunk in reranked
                if chunk.tipo_contenido == "costo"
                or (
                    any(
                        marker in normalize_for_search(chunk.text)
                        for marker in ("sisa", "monto", "tasa", "costo", "s/.", "s/")
                    )
                )
            ]
            cost_chunks.sort(
                key=lambda chunk: (chunk.knowledge_layer == "tramites", chunk.score),
                reverse=True,
            )
            if "sisa" not in topic_text:
                cost_chunks = [
                    chunk
                    for chunk in cost_chunks
                    if (
                        chunk.knowledge_layer == "tramites"
                        and chunk.tramite_relacionado == "tramite_comercio_ambulatorio"
                    )
                    or (
                        chunk.knowledge_layer == "faq"
                        and "costo_permiso" in chunk.chunk_id
                    )
                ]
            return cost_chunks[: self.settings.retrieval_top_k]

        if intent in {
            QueryIntent.REQUISITOS,
            QueryIntent.REQUISITOS_NUEVO,
            QueryIntent.REQUISITOS_RENOVACION,
        }:
            current_authorization_requirements = [
                chunk
                for chunk in reranked
                if chunk.tipo_contenido == "requisito"
                and chunk.vigencia in {"vigente", "vigente_con_observacion"}
            ]
            if current_authorization_requirements:
                # CAMBIO FASE OLLAMA 7 — Acotar requisitos a la norma modificatoria recuperada.
                # Motivo: evitar que el LLM mezcle requisitos con ubicaciones o tramites secundarios.
                # Riesgo mitigado: solo se activa cuando ya existe el Articulo 30 vigente.
                citizen_guidance = [
                    chunk
                    for chunk in current_authorization_requirements
                    if chunk.knowledge_layer == "tramites"
                    or (
                        chunk.knowledge_layer == "faq"
                        and "requisit" in normalize_for_search(chunk.chunk_id)
                    )
                ]
                if citizen_guidance:
                    procedure_guidance = [
                        chunk
                        for chunk in reranked
                        if chunk.knowledge_layer == "tramites"
                        and chunk.tipo_contenido == "procedimiento"
                        and any(
                            marker in normalize_for_search(chunk.chunk_id + " " + chunk.section_title)
                            for marker in ("pasos", "procedimiento")
                        )
                    ]
                    if not procedure_guidance:
                        procedure_guidance = [
                            self._retrieved_from_index_chunk(chunk, citizen_guidance[0].score - 0.05)
                            for chunk in self.retrieval_service.chunks
                            if chunk.knowledge_layer == "tramites"
                            and chunk.tipo_contenido == "procedimiento"
                            and "pasos" in normalize_for_search(chunk.chunk_id + " " + chunk.section_title)
                        ]
                    return [*citizen_guidance, *procedure_guidance][
                        : self.settings.retrieval_top_k
                    ]
                article_30 = [
                    chunk for chunk in current_authorization_requirements
                    if chunk.article_label == "30"
                ]
                return article_30[:1] or current_authorization_requirements[:1]

        if intent == QueryIntent.MODULOS:
            module_chunks = [
                chunk
                for chunk in reranked
                if (
                    chunk.tipo_contenido in {"modulo", "definicion"}
                    and (
                        "modulo" in normalize_for_search(chunk.text)
                        or "mobiliario" in normalize_for_search(chunk.text)
                    )
                )
                or "especificaciones tecnicas" in normalize_for_search(chunk.text)
                or "parametros tecnicos" in normalize_for_search(chunk.text)
            ]
            if module_chunks:
                return module_chunks[: self.settings.retrieval_top_k]

        if intent == QueryIntent.ZONAS_RIGIDAS:
            # CAMBIO FASE 7.2 - Encabezar el contexto con la restriccion vigente explicita.
            # Motivo: una lista de vias rigidas no basta para responder si se puede vender.
            # Riesgo mitigado: se mantiene el contexto de zonas como evidencia secundaria.
            explicit_restrictions = [
                chunk
                for chunk in reranked
                if "no se autoriza" in normalize_for_search(chunk.text)
                and "zona" in normalize_for_search(chunk.text)
                and chunk.vigencia in {"vigente", "vigente_con_observacion"}
            ]
            if explicit_restrictions:
                named_location = any(
                    marker in normalized_effective_query for marker in ("miguel grau", "manchay")
                )
                supporting_zones = [
                    chunk
                    for chunk in reranked
                    if chunk.chunk_id != explicit_restrictions[0].chunk_id
                    and chunk.tipo_contenido in {"zona", "prohibicion"}
                ]
                if named_location:
                    supporting_zones.sort(
                        key=lambda chunk: (chunk.article_label == "17.4", chunk.score),
                        reverse=True,
                    )
                return [explicit_restrictions[0], *supporting_zones][
                    : self.settings.retrieval_top_k
                ]

        if intent == QueryIntent.RUBROS:
            rubro_chunks = [
                chunk
                for chunk in reranked
                if chunk.tipo_contenido == "rubro"
                or chunk.article_label == "21"
                or "rubros permitidos" in normalize_for_search(chunk.section_title)
                or "giros permitidos" in normalize_for_search(chunk.text)
            ]
            rubro_chunks.sort(
                key=lambda chunk: (
                    chunk.knowledge_layer == "tramites",
                    chunk.article_label == "21",
                    chunk.score,
                ),
                reverse=True,
            )
            if rubro_chunks:
                return rubro_chunks[: self.settings.retrieval_top_k]

        if intent == QueryIntent.AUTORIZACIONES:
            authorization_chunks = [
                chunk
                for chunk in reranked
                if (
                    "autoriz" in normalize_for_search(chunk.text)
                    or chunk.article_label in {"6", "7", "7A", "8A", "30", "50"}
                )
                and chunk.tipo_contenido not in {"zona", "prohibicion"}
            ]
            if authorization_chunks:
                return authorization_chunks[: self.settings.retrieval_top_k]

        if intent in {QueryIntent.OBLIGACIONES, QueryIntent.SANCIONES, QueryIntent.REVOCACION}:
            related_types = {"obligacion", "sancion", "prohibicion"}
            related_articles = {"38", "50", "57", "63", "64"}
            obligation_or_sanction_chunks = [
                chunk
                for chunk in reranked
                if chunk.tipo_contenido in related_types
                or chunk.article_label in related_articles
                or any(
                    marker in normalize_for_search(chunk.text)
                    for marker in ("incumpl", "revoc", "retiro", "sancion", "obligacion")
                )
            ]
            if intent in {QueryIntent.SANCIONES, QueryIntent.REVOCACION}:
                obligation_or_sanction_chunks.sort(
                    key=lambda chunk: (
                        chunk.tipo_contenido == "sancion",
                        chunk.article_label in {"38", "50", "63", "64"},
                        chunk.score,
                    ),
                    reverse=True,
                )
            if obligation_or_sanction_chunks:
                return obligation_or_sanction_chunks[: self.settings.retrieval_top_k]

        if intent == QueryIntent.HORARIO:
            horario_chunks = [
                chunk
                for chunk in reranked
                if chunk.tipo_contenido == "horario"
                or chunk.article_label == "61"
                or "horario" in normalize_for_search(chunk.text)
            ]
            if horario_chunks:
                return horario_chunks[: self.settings.retrieval_top_k]

        if intent == QueryIntent.FERIAS:
            normalized_all = f"{normalized_question} {normalized_effective_query}"
            if any(term in normalized_all for term in ("bano", "baño", "servicios higienicos", "servicio higienico")):
                hygiene_chunks = [
                    chunk
                    for chunk in reranked
                    if chunk.article_label == "62"
                    or "servicios higienicos" in normalize_for_search(chunk.text)
                    or "bano" in normalize_for_search(chunk.text)
                    or "baño" in normalize_for_search(chunk.text)
                ]
                if hygiene_chunks:
                    return hygiene_chunks[: self.settings.retrieval_top_k]

        if intent == QueryIntent.NORMATIVA:
            identity_chunks = self._normative_identity_evidence()
            if identity_chunks:
                # CAMBIO FASE SIMULADOR 2 - Responder la norma aplicable con evidencia de identidad.
                # Motivo: articulos tematicos no acreditan cual ordenanza regula o modifica.
                # Riesgo mitigado: solo se seleccionan encabezados oficiales del corpus cargado.
                return identity_chunks[: self.settings.retrieval_top_k]

        if intent == QueryIntent.DEFINICION or asks_definition:
            current_definitions = [
                chunk
                for chunk in reranked
                if chunk.tipo_contenido == "definicion" and chunk.vigencia == "vigente"
            ]
            if current_definitions:
                # CAMBIO FASE OLLAMA 8 — Responder definiciones solo con definiciones vigentes.
                # Motivo: una definicion ciudadana no debe derivarse de requisitos secundarios.
                # Riesgo mitigado: solo se filtra cuando ya se recupero evidencia definitoria.
                return current_definitions[: self.settings.retrieval_top_k]

        return reranked[: self.settings.retrieval_top_k]

    def _requirement_case_chunks(
        self,
        chunks: list[RetrievedChunk],
        intent: QueryIntent,
        *,
        include_cost: bool,
    ) -> list[RetrievedChunk]:
        """Select the maintained requirement case without mixing it with the other case."""

        if intent == QueryIntent.REQUISITOS_NUEVO:
            selected_case = "nuevo_ingreso_padron"
        elif intent == QueryIntent.REQUISITOS_RENOVACION:
            selected_case = "renovacion"
        else:
            return []

        selected = [
            chunk
            for chunk in chunks
            if (
                chunk.document_id == "requisitos_comercio_ambulatorio"
                and chunk.metadata.get("tipo_tramite") == selected_case
            )
        ]
        if not selected:
            selected = [
                self._retrieved_from_index_chunk(chunk, 5.0)
                for chunk in self.retrieval_service.chunks
                if (
                    chunk.document_id == "requisitos_comercio_ambulatorio"
                    and chunk.metadata.get("tipo_tramite") == selected_case
                )
            ]
        selected.sort(key=lambda chunk: chunk.score, reverse=True)

        cost_chunks: list[RetrievedChunk] = []
        if include_cost:
            cost_chunks = [
                chunk
                for chunk in chunks
                if chunk.tipo_contenido == "costo"
                and (
                    chunk.knowledge_layer == "tramites"
                    or "tupa" in normalize_for_search(chunk.text)
                )
            ]
        return [*selected[:1], *cost_chunks[:1]]

    def _normative_identity_evidence(self) -> list[RetrievedChunk]:
        """Recover document-title evidence needed for questions about the governing norm."""

        specifications = (
            (
                "ordenanza_108_2012",
                "ordenanza que reglamenta el comercio ambulatorio",
                "disposicion",
            ),
            (
                "ordenanza_227_2019",
                "ordenanza modificatoria de la ordenanza",
                "base_legal",
            ),
        )
        evidence: list[RetrievedChunk] = []
        for document_id, marker, preferred_type in specifications:
            candidates = [
                chunk
                for chunk in self.retrieval_service.chunks
                if chunk.document_id == document_id
                and marker in normalize_for_search(chunk.text)
            ]
            if not candidates:
                continue
            chunk = min(
                candidates,
                key=lambda item: (
                    item.tipo_contenido != preferred_type,
                    len(item.text),
                ),
            )
            evidence.append(
                RetrievedChunk(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    source_title=chunk.source_title,
                    text=chunk.text,
                    score=4.0,
                    section_title="IDENTIFICACION NORMATIVA",
                    article_label=chunk.article_label,
                    normalized_text=chunk.normalized_text,
                    tipo_contenido=chunk.tipo_contenido,
                    user_intents=chunk.user_intents,
                    vigencia=chunk.vigencia,
                    modificado_por=chunk.modificado_por,
                    prioridad_retrieval=chunk.prioridad_retrieval,
                    fuente=chunk.fuente,
                    tramite_relacionado=chunk.tramite_relacionado,
                    knowledge_layer=chunk.knowledge_layer,
                    exclude_from_retrieval=chunk.exclude_from_retrieval,
                    requires_review=chunk.requires_review,
                    metadata=chunk.metadata,
                )
            )
        return evidence

    @staticmethod
    def _retrieved_from_index_chunk(chunk, score: float) -> RetrievedChunk:
        """Promote a known indexed guidance record into the current evidence set."""

        return RetrievedChunk(
            chunk_id=chunk.chunk_id,
            document_id=chunk.document_id,
            source_title=chunk.source_title,
            text=chunk.text,
            score=score,
            section_title=chunk.section_title,
            article_label=chunk.article_label,
            normalized_text=chunk.normalized_text,
            tipo_contenido=chunk.tipo_contenido,
            user_intents=chunk.user_intents,
            vigencia=chunk.vigencia,
            modificado_por=chunk.modificado_por,
            prioridad_retrieval=chunk.prioridad_retrieval,
            fuente=chunk.fuente,
            tramite_relacionado=chunk.tramite_relacionado,
            knowledge_layer=chunk.knowledge_layer,
            exclude_from_retrieval=chunk.exclude_from_retrieval,
            requires_review=chunk.requires_review,
            metadata=chunk.metadata,
        )

    def _format_source(self, chunk: RetrievedChunk) -> str:
        """Create a concise source string."""

        if chunk.fuente:
            return chunk.fuente
        parts = [chunk.source_title]
        if chunk.article_label:
            parts.append(f"Art. {chunk.article_label}")
        elif chunk.section_title:
            parts.append(chunk.section_title)
        return " | ".join(parts)

    def _correct_spelling(self, question: str) -> str:
        """Correct only a small whitelist of obvious typo forms."""

        normalized_question = normalize_for_search(question)
        tokens = re.findall(r"[a-záéíóúñ0-9]+", normalized_question)
        corrected_tokens = [COMMON_TYPO_HINTS.get(token, token) for token in tokens]
        return " ".join(corrected_tokens)
