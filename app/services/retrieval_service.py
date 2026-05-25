from __future__ import annotations

import logging
import re
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.config import Settings
from app.models.schemas import ConversationTurn, DocumentChunk, RetrievedChunk
from app.utils.helpers import ensure_directory, read_json
from app.utils.text_cleaner import normalize_for_search


ARTICLE_QUERY_PATTERN = re.compile(r"art[ií]culo\s+([0-9]+[A-Z]?)", re.IGNORECASE)

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
        """Load persisted chunks and vectors from disk if available."""

        if not self.settings.processed_chunks_file.exists():
            self.logger.warning(
                "No existe %s. Ejecuta primero scripts/ingest_documents.py",
                self.settings.processed_chunks_file,
            )
            return

        raw_chunks = read_json(self.settings.processed_chunks_file)
        self.chunks = [DocumentChunk(**item) for item in raw_chunks]

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
        ranked_indices = combined_scores.argsort()[::-1]
        candidates: list[RetrievedChunk] = []
        for index in ranked_indices:
            chunk = self.chunks[index]
            if chunk.prioridad_retrieval <= 0:
                continue
            if not historical_query and chunk.vigencia in {"historico", "modificado", "derogado"}:
                continue
            score = float(combined_scores[index]) + self._metadata_bonus(
                chunk,
                query_tokens=query_tokens,
                article_label=article_label,
                normalized_query=normalized_query,
            )
            if score <= 0:
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

        if chunk.vigencia == "vigente":
            bonus += 0.55
        bonus += max(0, chunk.prioridad_retrieval - 1) * 0.18

        if any(term in normalized_query for term in ("documento", "requisit", "que necesito")):
            if chunk.tipo_contenido == "requisito":
                bonus += 0.85
            if chunk.article_label == "30" and chunk.vigencia == "vigente":
                bonus += 1.10
        if "cuanto dura" in normalized_query or "vigencia" in normalized_query or "renovacion" in normalized_query:
            if "vigencia" in normalized_text or "plazo" in normalized_text:
                bonus += 0.75
            if chunk.article_label == "5" and chunk.vigencia == "vigente":
                bonus += 0.90
        if "zona" in normalized_query or ("donde" in normalized_query and "vender" in normalized_query):
            if chunk.tipo_contenido in {"zona", "prohibicion"}:
                bonus += 0.90
            if chunk.article_label in {"13", "17.4"} and chunk.vigencia == "vigente":
                bonus += 0.90

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

    def _extract_query_tokens(self, normalized_query: str) -> set[str]:
        """Extract coarse semantic tokens from the query."""

        return {
            token
            for token in re.findall(r"[a-záéíóúñ0-9]+", normalized_query)
            if len(token) > 2
        }
