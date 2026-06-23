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
from app.utils.query_expansion import expand_query
from app.utils.text_cleaner import normalize_for_search


ARTICLE_QUERY_PATTERN = re.compile(r"\b(?:art[ií]culo|art\.?)\s+([0-9]+(?:\.[0-9]+)?[A-Z]?)", re.IGNORECASE)

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
    "en que articulo",
    "que articulo",
    "donde dice",
)

TOPIC_HINTS = (
    "sisa",
    "modulo",
    "autoriz",
    "zona",
    "feria",
    "articulo",
    "vender",
    "giro",
    "rubro",
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
        expanded_query = expand_query(effective_query)
        normalized_query = normalize_for_search(expanded_query)
        historical_query = any(
            marker in normalized_query
            for marker in ("historico", "version anterior", "antes de la modificacion", "texto anterior")
        )
        query_tokens = self._extract_query_tokens(normalized_query)
        article_match = ARTICLE_QUERY_PATTERN.search(query)
        article_label = article_match.group(1).upper() if article_match else ""

        word_query_vector = self.word_vectorizer.transform([expanded_query])
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
        if ARTICLE_QUERY_PATTERN.search(query):
            return query
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

        if any(
            term in normalized_query
            for term in (
                "documento",
                "requisit",
                "que necesito",
                "sacar permiso",
                "sacar mi permiso",
                "obtener permiso",
                "obtener autorizacion",
                "como puedo sacar",
                "como saco",
                "como tramito",
            )
        ):
            if chunk.tipo_contenido == "requisito":
                bonus += 1.35
            if chunk.knowledge_layer == "tramites" and chunk.tipo_contenido == "procedimiento":
                bonus += 0.65
            if chunk.article_label == "30" and chunk.vigencia == "vigente":
                bonus += 1.10
            tipo_tramite = str(chunk.metadata.get("tipo_tramite", ""))
            if _looks_like_new_requirement_query(normalized_query):
                if tipo_tramite == "nuevo_ingreso_padron":
                    bonus += 4.00
                elif tipo_tramite == "renovacion":
                    bonus -= 3.50
            if _looks_like_renewal_requirement_query(normalized_query):
                if tipo_tramite == "renovacion":
                    bonus += 4.00
                elif tipo_tramite == "nuevo_ingreso_padron":
                    bonus -= 3.50
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
        if any(term in normalized_query for term in ("giro", "giros", "rubro", "rubros", "que puedo vender")):
            if chunk.tipo_contenido == "rubro":
                bonus += 2.50
            if chunk.knowledge_layer == "tramites" and chunk.tipo_contenido == "rubro":
                bonus += 1.00
            if chunk.article_label == "21":
                bonus += 1.00
        if any(
            term in normalized_query
            for term in (
                "cuantos giros",
                "cuantos rubros",
                "numero de giros",
                "cantidad de giros",
                "giros disponibles",
                "giros permitidos",
                "rubros disponibles",
                "rubros permitidos",
                "codigos de giro",
                "listado de giros",
                "lista de giros",
            )
        ):
            if chunk.tipo_contenido == "rubro":
                bonus += 5.00
            if chunk.article_label == "21":
                bonus += 2.00
            if chunk.tipo_contenido == "definicion":
                bonus -= 2.00
        if any(term in normalized_query for term in ("modulo", "puesto", "stand", "mobiliario")) and not any(
            term in normalized_query for term in ("quitar", "retiro", "retirar", "sancion", "revoc")
        ):
            if chunk.tipo_contenido in {"modulo", "definicion"} and (
                "modulo" in normalized_text or "mobiliario" in normalized_text
            ):
                bonus += 2.20
            if "especificaciones tecnicas" in normalized_text or "parametros tecnicos" in normalized_text:
                bonus += 1.20
            if chunk.tipo_contenido in {"zona", "prohibicion"}:
                bonus -= 0.90
        if any(
            term in normalized_query
            for term in (
                "obligacion",
                "obligaciones",
                "deberes",
                "cumplir",
                "no cumple",
                "no cumplo",
                "incumpl",
                "sancion",
                "revocacion",
                "retiro",
                "quitar",
                "fiscalizacion",
            )
        ):
            if chunk.tipo_contenido in {"obligacion", "sancion", "prohibicion"}:
                bonus += 1.60
            if chunk.article_label in {"38", "50", "57", "63", "64"}:
                bonus += 1.45
            if any(marker in normalized_text for marker in ("incumpl", "revoc", "retiro", "sancion")):
                bonus += 0.70
        if "feria" in normalized_query and ("horario" in normalized_query or "articulo 61" in normalized_query):
            if chunk.article_label == "61" or chunk.tipo_contenido == "horario":
                bonus += 2.50
        if "feria" in normalized_query and any(
            term in normalized_query for term in ("servicios higienicos", "bano", "baño", "articulo 62")
        ):
            if chunk.article_label == "62":
                bonus += 2.75
            if "servicios higienicos" in normalized_text or "bano" in normalized_text or "baño" in normalized_text:
                bonus += 1.25
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
            concept = str(chunk.metadata.get("concepto", ""))
            if chunk.tipo_contenido == "definicion":
                bonus += 1.50
            if concept and _definition_concept_matches(normalized_query, concept):
                bonus += 3.50
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
        glossary_file = self.settings.tramites_data_dir / "glosario_comercio_ambulatorio.json"
        requirements_file = self.settings.tramites_data_dir / "requisitos_comercio_ambulatorio.json"
        tramite_file = self.settings.tramites_data_dir / "comercio_ambulatorio.json"
        zonas_file = self.settings.tramites_data_dir / "zonas_restringidas_comercio_ambulatorio.json"
        unverified_articles_file = self.settings.tramites_data_dir / "ordenanza_227_articulos_57_64.json"
        faq_file = self.settings.faq_data_dir / "comercio_ambulatorio_faq.json"
        if glossary_file.exists():
            chunks.extend(self._chunks_from_glossary(read_json(glossary_file)))
        if requirements_file.exists():
            chunks.extend(self._chunks_from_requirement_cases(read_json(requirements_file)))
        if tramite_file.exists():
            chunks.extend(self._chunks_from_tramite(read_json(tramite_file)))
        if zonas_file.exists():
            chunks.extend(self._chunks_from_zones(read_json(zonas_file)))
        if unverified_articles_file.exists():
            chunks.extend(self._chunks_from_unverified_articles(read_json(unverified_articles_file)))
        if faq_file.exists():
            chunks.extend(self._chunks_from_faq(read_json(faq_file)))
        if self.settings.consolidated_norm_file.exists():
            chunks.extend(self._chunks_from_consolidated(read_json(self.settings.consolidated_norm_file)))
        return chunks

    def _chunks_from_glossary(self, payload: dict[str, Any]) -> list[DocumentChunk]:
        """Load concise citizen definitions as first-step guidance."""

        title = str(payload.get("nombre", "Glosario ciudadano de comercio ambulatorio"))
        chunks: list[DocumentChunk] = []
        for item in payload.get("conceptos", []):
            concept_id = str(item.get("id", "")).strip()
            term = str(item.get("termino", "")).strip()
            definition = str(item.get("definicion_ciudadana", "")).strip()
            follow_up = str(item.get("pregunta_orientadora", "")).strip()
            variants = " | ".join(
                str(value).strip()
                for value in item.get("variantes", [])
                if str(value).strip()
            )
            if not concept_id or not definition:
                continue
            text = "\n".join(
                part
                for part in (
                    f"Concepto: {term}",
                    f"Variantes: {variants}" if variants else "",
                    f"Definicion ciudadana: {definition}",
                    f"Pregunta orientadora: {follow_up}" if follow_up else "",
                )
                if part
            )
            chunks.append(
                DocumentChunk(
                    chunk_id=f"glosario-{concept_id}",
                    document_id=str(payload.get("id", "glosario_comercio_ambulatorio")),
                    source_title=title,
                    text=text,
                    section_title=term,
                    normalized_text=normalize_for_search(text),
                    tipo_contenido="definicion",
                    user_intents=["consulta_definicion"],
                    vigencia="vigente",
                    prioridad_retrieval=5,
                    fuente=str(item.get("fuente", title)),
                    tramite_relacionado="tramite_comercio_ambulatorio",
                    knowledge_layer="tramites",
                    requires_review=False,
                    metadata={
                        "concepto": concept_id,
                        "termino": term,
                        "pregunta_orientadora": follow_up,
                    },
                )
            )
        return chunks

    def _chunks_from_tramite(self, payload: dict[str, Any]) -> list[DocumentChunk]:
        title = f"Ficha de tramite: {payload.get('nombre_tramite', 'Comercio ambulatorio')}"
        draft = bool(payload.get("requiere_validacion_humana", True))
        requirements = payload.get("requisitos", [])
        requirement_text = "\n".join(
            f"- {item.get('descripcion', '')} Fuente: {item.get('fuente', '')}"
            for item in requirements
        )
        rubros = payload.get("rubros_permitidos", [])
        rubros_text = _format_rubros_permitidos(rubros)
        records = []
        if not (self.settings.tramites_data_dir / "requisitos_comercio_ambulatorio.json").exists():
            records.append(
                (
                    "requisitos",
                    "REQUISITOS",
                    f"{payload.get('descripcion', '')}\nRequisitos:\n{requirement_text}",
                    "requisito",
                    "Ordenanza 227-2019-MDP/C, Articulo 30",
                )
            )
        records.extend([
            (
                "pasos",
                "PASOS DEL TRAMITE",
                "\n".join(
                    _format_tramite_step(item)
                    for item in (
                        payload.get("pasos")
                        or payload.get("pasos_orientativos")
                        or []
                    )
                ),
                "procedimiento",
                "Ficha de tramite municipal basada en Ordenanza 227-2019-MDP/C",
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
                "rubros",
                "RUBROS Y GIROS PERMITIDOS",
                rubros_text,
                "rubro",
                "Ordenanza 227-2019-MDP/C, Articulo 21",
            ),
            (
                "restricciones",
                "RESTRICCIONES",
                "\n".join(str(item) for item in payload.get("restricciones", [])),
                "zona",
                "Ordenanza 227-2019-MDP/C, Articulos 2 y 17.4",
            ),
        ])
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
                article_label="21" if key == "rubros" else "",
                requires_review=draft,
            )
            for key, section, text, content_type, fuente in records
            if text.strip()
        ]

    def _chunks_from_requirement_cases(self, payload: dict[str, Any]) -> list[DocumentChunk]:
        """Load maintained citizen-facing requirements split by case."""

        title = "Ficha interna de requisitos de comercio ambulatorio"
        source_names = [
            str(item.get("nombre", "")).strip()
            for item in payload.get("fuentes", [])
            if str(item.get("nombre", "")).strip()
        ]
        general_source = "; ".join(source_names) or title
        chunks: list[DocumentChunk] = []
        for item in payload.get("tipos_tramite", []):
            case_id = str(item.get("id", "")).strip()
            if not case_id:
                continue
            requisitos = "\n".join(
                f"- {str(value).strip()}"
                for value in item.get("requisitos", [])
                if str(value).strip()
            )
            cuando_aplica = "\n".join(
                f"- {str(value).strip()}"
                for value in item.get("cuando_aplica", [])
                if str(value).strip()
            )
            frases = " | ".join(
                str(value).strip()
                for value in item.get("frases_usuario", [])
                if str(value).strip()
            )
            no_confundir = "\n".join(
                f"- {str(value).strip()}"
                for value in item.get("no_confundir_con", [])
                if str(value).strip()
            )
            text = "\n".join(
                part
                for part in (
                    str(item.get("nombre", "")).strip(),
                    "Cuando aplica:",
                    cuando_aplica,
                    "Formas comunes de preguntar:",
                    frases,
                    "Requisitos:",
                    requisitos,
                    "Explicacion ciudadana:",
                    str(item.get("explicacion_ciudadana", "")).strip(),
                    "Version simple:",
                    str(item.get("explicacion_simple", "")).strip(),
                    "No confundir:",
                    no_confundir,
                    "Observaciones generales:",
                    "\n".join(f"- {value}" for value in payload.get("observaciones_generales", [])),
                )
                if part
            )
            if not text.strip():
                continue
            intent = (
                "consulta_requisitos_renovacion"
                if case_id == "renovacion"
                else "consulta_requisitos_nuevo"
            )
            chunks.append(
                DocumentChunk(
                    chunk_id=f"requisitos-comercio-ambulatorio-{case_id}",
                    document_id="requisitos_comercio_ambulatorio",
                    source_title=title,
                    text=text,
                    section_title=str(item.get("nombre", "")).strip(),
                    normalized_text=normalize_for_search(text),
                    tipo_contenido="requisito",
                    user_intents=[intent, "consulta_requisitos"],
                    vigencia="vigente",
                    prioridad_retrieval=5,
                    fuente=general_source,
                    tramite_relacionado="tramite_comercio_ambulatorio",
                    knowledge_layer="tramites",
                    requires_review=False,
                    metadata={
                        "tipo_tramite": case_id,
                        "area_responsable": str(payload.get("area_responsable", "")),
                        "explicacion_simple": str(item.get("explicacion_simple", "")),
                    },
                )
            )
        return chunks

    def _chunks_from_zones(self, payload: dict[str, Any]) -> list[DocumentChunk]:
        title = str(payload.get("nombre", "Zonas restringidas de comercio ambulatorio"))
        chunks: list[DocumentChunk] = []
        for item in payload.get("zonas", []):
            zone_id = str(item.get("id", "zona_restringida"))
            location = str(item.get("ubicacion", ""))
            restriction = str(item.get("restriccion", ""))
            source = str(item.get("fuente", ""))
            text = "\n".join(part for part in (location, restriction, str(item.get("nota", ""))) if part)
            if not text.strip():
                continue
            chunks.append(
                DocumentChunk(
                    chunk_id=f"zonas-{zone_id}",
                    document_id=str(payload.get("id", "zonas_restringidas_comercio_ambulatorio")),
                    source_title=title,
                    text=text,
                    section_title="ZONAS RESTRINGIDAS",
                    article_label=str(item.get("articulo", "")),
                    normalized_text=normalize_for_search(text),
                    tipo_contenido="zona",
                    user_intents=["consulta_zona_restringida", "consulta_ubicacion"],
                    vigencia=str(item.get("vigencia", "vigente_con_observacion")),
                    prioridad_retrieval=3,
                    fuente=source,
                    tramite_relacionado="tramite_comercio_ambulatorio",
                    knowledge_layer="zonas",
                    requires_review=bool(item.get("requires_review", True)),
                    metadata={"ubicacion": location},
                )
            )
        return chunks

    def _chunks_from_unverified_articles(self, payload: dict[str, Any]) -> list[DocumentChunk]:
        title = str(payload.get("nombre", "Articulos no verificables de Ordenanza 227-2019"))
        chunks: list[DocumentChunk] = []
        for item in payload.get("articulos", []):
            label = str(item.get("articulo", ""))
            text = str(item.get("observacion", "Texto no disponible en la fuente cargada."))
            chunks.append(
                DocumentChunk(
                    chunk_id=f"ordenanza-227-articulo-{label}-no-verificable",
                    document_id=str(payload.get("id", "ordenanza_227_articulos_57_64")),
                    source_title=title,
                    text=text,
                    section_title="ARTICULOS NO VERIFICABLES",
                    article_label=label,
                    normalized_text=normalize_for_search(text),
                    tipo_contenido="disposicion",
                    user_intents=["consulta_normativa"],
                    vigencia="no_verificable",
                    prioridad_retrieval=0,
                    fuente=str(item.get("fuente", "Ordenanza 227-2019-MDP/C")),
                    tramite_relacionado="tramite_comercio_ambulatorio",
                    knowledge_layer="norma_no_verificable",
                    exclude_from_retrieval=True,
                    requires_review=True,
                    metadata={"observacion": text},
                )
            )
        return chunks

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
        if label == "57":
            return "obligacion"
        if label in {"50", "63", "64"}:
            return "sancion"
        if label == "61":
            return "horario"
        if label == "62":
            return "requisito"
        if label == "36":
            return "costo"
        if label == "21" or "rubro" in normalized or "giro" in normalized:
            return "rubro"
        if label == "30" or "requisit" in normalized:
            return "requisito"
        if "sisa" in normalized and "pago" in normalized:
            return "costo"
        if "obligacion" in normalized or "debe " in normalized or "deber" in normalized:
            return "obligacion"
        if "revoc" in normalized or "sancion" in normalized or "retiro" in normalized:
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


def _format_tramite_step(item: Any) -> str:
    if isinstance(item, dict):
        order = item.get("orden")
        prefix = f"{order}. " if order else "- "
        description = str(item.get("descripcion", "")).strip()
        source = str(item.get("fuente", "")).strip()
        return f"{prefix}{description}" + (f" Fuente: {source}" if source else "")
    return f"- {item}"


def _format_rubro_permitido(item: Any) -> str:
    if not isinstance(item, dict):
        return f"- {item}"

    rubro = str(item.get("rubro", "")).strip()
    nombre = str(item.get("nombre", "")).strip()
    fuente = str(item.get("fuente", "")).strip()
    header = " - ".join(part for part in (rubro, nombre) if part)
    giros = item.get("giros", [])
    lines = [header] if header else []
    for giro in giros:
        if isinstance(giro, dict):
            codigo = str(giro.get("codigo", "")).strip()
            descripcion = str(giro.get("descripcion", "")).strip()
            label = f"{codigo}: {descripcion}" if codigo else descripcion
            if label:
                lines.append(f"  - {label}")
        elif str(giro).strip():
            lines.append(f"  - {giro}")
    if fuente:
        lines.append(f"  Fuente: {fuente}")
    return "\n".join(lines)


def _format_rubros_permitidos(items: Any) -> str:
    if not isinstance(items, list) or not items:
        return ""

    total_rubros = 0
    total_giros = 0
    blocks: list[str] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        total_rubros += 1
        giros = item.get("giros", [])
        if isinstance(giros, list):
            total_giros += sum(1 for giro in giros if giro)
        formatted = _format_rubro_permitido(item)
        if formatted:
            blocks.append(formatted)

    summary = (
        "Articulo 21 - Rubros y giros permitidos para comercio ambulatorio.\n"
        f"Cantidad total registrada: {total_rubros} rubros y {total_giros} giros permitidos.\n"
        "La autorizacion municipal debe corresponder a un giro especifico."
    )
    return "\n\n".join([summary, *blocks])


def _looks_like_new_requirement_query(normalized_query: str) -> bool:
    return any(
        marker in normalized_query
        for marker in (
            "sacar permiso",
            "sacar mi permiso",
            "como saco permiso",
            "como saco mi permiso",
            "como puedo sacar",
            "primera vez",
            "soy nuevo",
            "quiero vender",
            "vender en la calle",
            "vender en la via publica",
            "vender en via publica",
            "permiso para vender",
            "necesito permiso para vender",
            "ingresar al padron",
            "ingreso al padron",
            "inscribirme",
            "que necesito para vender",
            "que necesito y cuanto cuesta",
            "pa vender",
        )
    )


def _looks_like_renewal_requirement_query(normalized_query: str) -> bool:
    return any(
        marker in normalized_query
        for marker in (
            "renovar",
            "renovacion",
            "como renuevo",
            "que necesito para renovar",
            "ya tengo permiso",
            "ya tengo autorizacion",
            "permiso esta por vencer",
            "permiso se vence",
            "autorizacion vence",
            "seguir vendiendo",
            "tengo mi voucher",
            "voucher",
        )
    )


def _definition_concept_matches(normalized_query: str, concept: str) -> bool:
    aliases = {
        "comercio_ambulatorio": ("comercio ambulatorio", "ambulatorio", "venta ambulante"),
        "padron_municipal": ("padron", "padron municipal", "registro municipal"),
        "tupa": ("tupa",),
        "sisa": ("sisa",),
        "modulo": ("modulo", "puesto", "stand"),
        "giro": ("giro", "rubro", "actividad", "tipo de venta"),
    }
    return any(marker in normalized_query for marker in aliases.get(concept, (concept,)))
