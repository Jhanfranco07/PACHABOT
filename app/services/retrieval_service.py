from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.config import Settings
from app.models.schemas import ConversationTurn, DocumentChunk, RetrievedChunk
from app.utils.helpers import ensure_directory, read_json
from app.utils.text_cleaner import normalize_for_search


ARTICLE_QUERY_PATTERN = re.compile(r"art[ií]culo\s+([0-9]+(?:\.[0-9]+)?[A-Z]?)", re.IGNORECASE)

FOLLOW_UP_HINTS = (
    "y ",
    "tambien",
    "también",
    "osea",
    "o sea",
    "entonces",
    "eso",
    "esa",
    "ese",
    "otra cosa",
    "otra pregunta",
    "otra consulta",
)

TOPIC_HINTS = (
    "sisa",
    "modulo",
    "autoriz",
    "zona",
    "feria",
    "articulo",
    "vender",
    "via publica",
)


class RetrievalService:
    """Local retrieval with TF-IDF plus light conversational expansion."""

    def __init__(self, settings: Settings, logger: logging.Logger) -> None:
        self.settings = settings
        self.logger = logger.getChild("retrieval_service")
        self.word_vectorizer: TfidfVectorizer | None = None
        self.char_vectorizer: TfidfVectorizer | None = None
        self.word_matrix = None
        self.char_matrix = None
        self.chunks: list[DocumentChunk] = []

    def build_index(self, chunks: list[DocumentChunk]) -> None:
        """Create and persist the local vector index."""

        ensure_directory(self.settings.vectorstore_dir)
        texts = [self._build_index_text(chunk) for chunk in chunks]
        normalized_texts = [normalize_for_search(text) for text in texts]

        self.word_vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words=None)
        self.char_vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))
        self.word_matrix = self.word_vectorizer.fit_transform(texts)
        self.char_matrix = self.char_vectorizer.fit_transform(normalized_texts)
        self.chunks = chunks

        joblib.dump(
            {
                "word": self.word_vectorizer,
                "char": self.char_vectorizer,
            },
            self.settings.vectorizer_file,
        )
        joblib.dump(
            {
                "word": self.word_matrix,
                "char": self.char_matrix,
            },
            self.settings.matrix_file,
        )
        self.logger.info("Vectorstore local actualizado en %s", self.settings.vectorstore_dir)

    def load_index(self) -> None:
        """Load RAG chunks and citizen-oriented knowledge sources."""

        if not self.settings.processed_chunks_file.exists():
            self.logger.warning(
                "No existe %s. Ejecuta primero scripts/ingest_documents.py",
                self.settings.processed_chunks_file,
            )
            return

        raw_chunks = read_json(self.settings.processed_chunks_file)
        processed_chunks = [DocumentChunk(**item) for item in raw_chunks]
        self.chunks = self.compose_knowledge_index(processed_chunks)

        if self.settings.vectorizer_file.exists() and self.settings.matrix_file.exists():
            vectorizers = joblib.load(self.settings.vectorizer_file)
            matrices = joblib.load(self.settings.matrix_file)
            if not isinstance(vectorizers, dict) or not isinstance(matrices, dict):
                self.logger.warning("El vectorstore persistido es de una version anterior. Se reconstruira.")
                self.build_index(self.chunks)
                return

            self.word_vectorizer = vectorizers.get("word")
            self.char_vectorizer = vectorizers.get("char")
            self.word_matrix = matrices.get("word")
            self.char_matrix = matrices.get("char")

            matrix_rows = getattr(self.word_matrix, "shape", (0, 0))[0]
            if matrix_rows != len(self.chunks):
                self.logger.warning(
                    "El vectorstore persistido no coincide con los chunks procesados (%s vs %s). Se reconstruira.",
                    matrix_rows,
                    len(self.chunks),
                )
                self.build_index(self.chunks)
            else:
                self.logger.info("Vectorstore local cargado correctamente")
        else:
            self.logger.warning("No se encontro el vectorstore persistido; se usara indexacion temporal")
            self.build_index(self.chunks)

    def search(
        self,
        query: str,
        top_k: int = 4,
        *,
        history: list[ConversationTurn] | None = None,
    ) -> list[RetrievedChunk]:
        """Retrieve the most relevant chunks for a query, optionally enriched with history."""

        if not self.chunks:
            self.load_index()

        if not self.chunks or self.word_vectorizer is None or self.char_vectorizer is None:
            return []
        if self.word_matrix is None or self.char_matrix is None:
            return []

        effective_query = self._expand_query_with_history(query, history or [])
        normalized_query = normalize_for_search(effective_query)
        historical_query = any(
            marker in normalized_query
            for marker in ("historico", "version anterior", "antes de la modificacion", "texto anterior")
        )
        query_tokens = self._extract_query_tokens(normalized_query)
        article_match = ARTICLE_QUERY_PATTERN.search(query)
        article_label = article_match.group(1).upper() if article_match else ""

        word_query_vector = self.word_vectorizer.transform([effective_query])
        char_query_vector = self.char_vectorizer.transform([normalized_query])
        word_scores = cosine_similarity(word_query_vector, self.word_matrix).flatten()
        char_scores = cosine_similarity(char_query_vector, self.char_matrix).flatten()
        combined_scores = (word_scores * 0.72) + (char_scores * 0.28)

        # CAMBIO FASE 7.2 — Excluir ruido documental e historicos de la respuesta ordinaria.
        # Motivo: los considerandos y textos reemplazados no deben orientar al ciudadano.
        # Riesgo mitigado: una consulta explicita sobre version historica aun puede recuperarlos.
        top_k = min(top_k, self.settings.retrieval_max_results)
        ranked_indices = combined_scores.argsort()[::-1]
        candidates: list[RetrievedChunk] = []
        for index in ranked_indices:
            chunk = self.chunks[index]
            if chunk.exclude_from_retrieval or chunk.prioridad_retrieval <= 0:
                continue
            if not historical_query and chunk.vigencia in {"historico", "modificado", "derogado"}:
                continue
            if chunk.vigencia in {"no_verificable", "vigencia_no_verificable"}:
                continue
            score = float(combined_scores[index]) + self._metadata_bonus(
                chunk,
                query_tokens=query_tokens,
                article_label=article_label,
                normalized_query=normalized_query,
            )
            if score < self.settings.retrieval_min_score:
                continue

            candidates.append(
                RetrievedChunk(
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
            )

        results: list[RetrievedChunk] = []
        seen_keys: set[tuple[str, str]] = set()
        for item in sorted(candidates, key=lambda candidate: candidate.score, reverse=True):
            key = (item.document_id, item.article_label or item.chunk_id)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            results.append(item)
            if len(results) >= top_k:
                break

        return results

    def reset(self) -> None:
        """Delete persisted vectorstore artifacts."""

        for target in [self.settings.vectorizer_file, self.settings.matrix_file]:
            if Path(target).exists():
                Path(target).unlink()
                self.logger.info("Archivo eliminado: %s", target)

    def _build_index_text(self, chunk: DocumentChunk) -> str:
        """Expand the indexed text using legal metadata."""

        parts = [
            chunk.source_title,
            chunk.section_title,
            f"Articulo {chunk.article_label}" if chunk.article_label else "",
            chunk.tipo_contenido,
            chunk.knowledge_layer,
            " ".join(chunk.user_intents),
            chunk.text,
        ]
        return "\n".join(part for part in parts if part)

    def _expand_query_with_history(
        self,
        query: str,
        history: list[ConversationTurn],
    ) -> str:
        """Expand short follow-ups with the previous user message."""

        if not history:
            return query

        normalized_query = normalize_for_search(query)
        tokens = normalized_query.split()
        explicit_follow_up = any(normalized_query.startswith(marker) for marker in FOLLOW_UP_HINTS)
        has_topic_hint = any(hint in normalized_query for hint in TOPIC_HINTS)
        if not explicit_follow_up and (len(tokens) > 5 or has_topic_hint):
            return query

        previous_user_turn = next(
            (
                turn.text
                for turn in reversed(history)
                if turn.role == "user"
                and normalize_for_search(turn.text) != normalized_query
            ),
            "",
        )
        if not previous_user_turn:
            return query

        return f"{query} {previous_user_turn}"

    def _metadata_bonus(
        self,
        chunk: DocumentChunk,
        *,
        query_tokens: set[str],
        article_label: str,
        normalized_query: str,
    ) -> float:
        """Apply a light metadata-aware bonus to stabilize relevant results."""

        normalized_text = normalize_for_search(chunk.text)
        normalized_section = normalize_for_search(chunk.section_title)
        bonus = 0.0

        if chunk.vigencia in {"vigente", "vigente_con_observacion"}:
            bonus += 0.55
        bonus += max(0, chunk.prioridad_retrieval - 1) * 0.18
        bonus += {
            "tramites": 1.20,
            "faq": 1.00,
            "norma_consolidada": 0.45,
            "chunks": 0.0,
        }.get(chunk.knowledge_layer, 0.0)

        if any(term in normalized_query for term in ("documento", "requisit", "que necesito")):
            if chunk.tipo_contenido == "requisito":
                bonus += 0.85
            if chunk.article_label == "30" and chunk.vigencia == "vigente":
                bonus += 1.10
        if any(term in normalized_query for term in ("cuanto cuesta", "costo", "monto", "tasa", "pago")):
            if chunk.tipo_contenido == "costo":
                bonus += 2.00
            if chunk.knowledge_layer == "tramites" and chunk.tipo_contenido == "costo":
                bonus += 0.75
        if "cuanto dura" in normalized_query or "vigencia" in normalized_query or "renovacion" in normalized_query:
            if "vigencia" in normalized_text or "plazo" in normalized_text:
                bonus += 0.75
            if chunk.article_label == "5" and chunk.vigencia == "vigente":
                bonus += 0.90
        if "zona" in normalized_query or ("donde" in normalized_query and "vender" in normalized_query):
            if chunk.tipo_contenido in {"zona", "prohibicion"}:
                bonus += 0.90
            if "zona" in normalized_text and "no se autoriza" in normalized_text:
                # CAMBIO FASE 7.2 - Priorizar prohibiciones directas sobre zonas rigidas.
                # Motivo: entregar al LLM la evidencia que responde la autorizacion solicitada.
                # Riesgo mitigado: el bono solo aplica a texto documental explicito.
                bonus += 2.25
            if chunk.article_label in {"13", "17.4"} and chunk.vigencia == "vigente":
                bonus += 0.90
        if "miguel grau" in normalized_query or "manchay" in normalized_query:
            if "miguel grau" in normalized_text or "manchay" in normalized_text:
                bonus += 3.00
            if chunk.article_label == "17.4" and chunk.vigencia == "vigente":
                bonus += 1.00
        if "sisa" in normalized_query and any(
            marker in normalized_query for marker in ("no pago", "no pagar", "incumpl", "deuda")
        ):
            if chunk.article_label == "38" and chunk.vigencia == "vigente":
                bonus += 3.00
            if chunk.tipo_contenido == "sancion":
                bonus += 1.00

        if article_label and chunk.article_label and chunk.article_label.upper() == article_label:
            bonus += 2.00

        if "que es" in normalized_query or "defin" in normalized_query:
            if "definiciones" in normalized_section or "se entiende por" in normalized_text:
                bonus += 0.90

        overlap_in_section = sum(1 for token in query_tokens if token in normalized_section)
        overlap_in_text = sum(1 for token in query_tokens if token in normalized_text)
        bonus += min(0.20, overlap_in_section * 0.08)
        bonus += min(0.25, overlap_in_text * 0.03)

        if chunk.section_title.startswith("PREAMBULO"):
            bonus -= 0.45

        return bonus

    def _eligible_index_chunks(self, chunks: list[DocumentChunk]) -> list[DocumentChunk]:
        """Exclude known unusable records before they can influence any vector score."""

        return [
            chunk
            for chunk in chunks
            if not chunk.exclude_from_retrieval
            and chunk.vigencia not in {"no_verificable", "vigencia_no_verificable"}
        ]

    def compose_knowledge_index(self, processed_chunks: list[DocumentChunk]) -> list[DocumentChunk]:
        """Combine processed RAG chunks with maintained citizen guidance sources."""

        return self._eligible_index_chunks(
            [*processed_chunks, *self._load_additional_knowledge()]
        )

    def _load_additional_knowledge(self) -> list[DocumentChunk]:
        """Load citizen guidance and consolidated law after the processed RAG corpus."""

        chunks: list[DocumentChunk] = []
        tramite_file = self.settings.tramites_data_dir / "comercio_ambulatorio.json"
        faq_file = self.settings.faq_data_dir / "comercio_ambulatorio_faq.json"
        if tramite_file.exists():
            chunks.extend(self._chunks_from_tramite(read_json(tramite_file)))
        if faq_file.exists():
            chunks.extend(self._chunks_from_faq(read_json(faq_file)))
        if self.settings.consolidated_norm_file.exists():
            chunks.extend(self._chunks_from_consolidated(read_json(self.settings.consolidated_norm_file)))
        return chunks

    def _chunks_from_tramite(self, payload: dict[str, Any]) -> list[DocumentChunk]:
        title = f"Ficha de tramite: {payload.get('nombre_tramite', 'Comercio ambulatorio')}"
        draft = bool(payload.get("requiere_validacion_humana", True))
        requirements = payload.get("requisitos", [])
        requirement_text = "\n".join(
            f"- {item.get('descripcion', '')} Fuente: {item.get('fuente', '')}"
            for item in requirements
        )
        records = [
            (
                "requisitos",
                "REQUISITOS",
                f"{payload.get('descripcion', '')}\nRequisitos:\n{requirement_text}",
                "requisito",
                "Ordenanza 227-2019-MDP/C, Articulo 30",
            ),
            (
                "costo",
                "COSTO",
                str(payload.get("costo", {}).get("nota", "")),
                "costo",
                str(payload.get("costo", {}).get("fuente", "")),
            ),
            (
                "vigencia",
                "VIGENCIA Y RENOVACION",
                str(payload.get("vigencia", {}).get("descripcion", "")),
                "procedimiento",
                str(payload.get("vigencia", {}).get("fuente", "")),
            ),
            (
                "restricciones",
                "RESTRICCIONES",
                "\n".join(str(item) for item in payload.get("restricciones", [])),
                "zona",
                "Ordenanza 227-2019-MDP/C, Articulos 2 y 17.4",
            ),
        ]
        return [
            DocumentChunk(
                chunk_id=f"tramite-{payload.get('id', 'comercio_ambulatorio')}-{key}",
                document_id=str(payload.get("id", "tramite_comercio_ambulatorio")),
                source_title=title,
                text=text,
                section_title=section,
                normalized_text=normalize_for_search(text),
                tipo_contenido=content_type,
                user_intents=[f"consulta_{key}"],
                vigencia="vigente_con_observacion" if draft else "vigente",
                prioridad_retrieval=3,
                fuente=fuente,
                tramite_relacionado=str(payload.get("id", "")),
                knowledge_layer="tramites",
                requires_review=draft,
            )
            for key, section, text, content_type, fuente in records
            if text.strip()
        ]

    def _chunks_from_faq(self, payload: list[dict[str, Any]]) -> list[DocumentChunk]:
        chunks: list[DocumentChunk] = []
        for item in payload:
            faq_id = str(item.get("faq_id", "faq"))
            text = "\n".join(
                [
                    str(item.get("pregunta", "")),
                    *[str(value) for value in item.get("variantes", [])],
                    str(item.get("respuesta_orientativa", "")),
                ]
            )
            content_type = self._infer_content_type(faq_id, text)
            chunks.append(
                DocumentChunk(
                    chunk_id=faq_id,
                    document_id="faq_comercio_ambulatorio",
                    source_title="FAQ orientativa de comercio ambulatorio",
                    text=text,
                    section_title=str(item.get("pregunta", "")),
                    normalized_text=normalize_for_search(text),
                    tipo_contenido=content_type,
                    user_intents=[f"consulta_{content_type}"],
                    vigencia="vigente_con_observacion" if item.get("requiere_actualizacion") else "vigente",
                    prioridad_retrieval=3,
                    fuente="; ".join(str(source) for source in item.get("fuentes", [])),
                    tramite_relacionado=str(item.get("tramite_relacionado", "")),
                    knowledge_layer="faq",
                    requires_review=bool(item.get("requiere_actualizacion")),
                )
            )
        return chunks

    def _chunks_from_consolidated(self, payload: dict[str, Any]) -> list[DocumentChunk]:
        chunks: list[DocumentChunk] = []
        for article in [*payload.get("articles", []), *payload.get("unverified_articles", [])]:
            label = str(article.get("article_label", ""))
            text = str(article.get("text", ""))
            chunks.append(
                DocumentChunk(
                    chunk_id=f"consolidada-{label}",
                    document_id="norma_consolidada",
                    source_title=str(article.get("source_title", "Norma consolidada")),
                    text=text,
                    section_title=str(article.get("section_title", "")),
                    article_label=label,
                    normalized_text=normalize_for_search(text),
                    tipo_contenido=self._infer_content_type(label, text),
                    user_intents=["consulta_normativa"],
                    vigencia=str(article.get("vigencia", "vigente")),
                    modificado_por=str(article.get("modificado_por", "")),
                    prioridad_retrieval=3,
                    fuente="; ".join(str(source) for source in article.get("source_trace", [])),
                    tramite_relacionado="tramite_comercio_ambulatorio",
                    knowledge_layer="norma_consolidada",
                    exclude_from_retrieval=bool(article.get("exclude_from_retrieval", False)),
                    requires_review=bool(article.get("requires_review", False)),
                    metadata={"validation_notes": article.get("validation_notes", [])},
                )
            )
        return chunks

    @staticmethod
    def _infer_content_type(label: str, text: str) -> str:
        normalized = normalize_for_search(f"{label} {text}")
        if label == "5":
            return "procedimiento"
        if label in {"2", "17", "17.4"}:
            return "zona"
        if label == "38":
            return "sancion"
        if label == "36":
            return "costo"
        if label == "30" or "requisit" in normalized:
            return "requisito"
        if "sisa" in normalized and "pago" in normalized:
            return "costo"
        if "revoc" in normalized or "sancion" in normalized:
            return "sancion"
        if "zona rigida" in normalized:
            return "zona"
        if "vigencia" in normalized or "renov" in normalized:
            return "procedimiento"
        return "disposicion"

    def _extract_query_tokens(self, normalized_query: str) -> set[str]:
        """Extract coarse semantic tokens from the query."""

        return {
            token
            for token in re.findall(r"[a-záéíóúñ0-9]+", normalized_query)
            if len(token) > 2
        }
