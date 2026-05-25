from __future__ import annotations

from typing import TYPE_CHECKING

from app.config import Settings
from app.models.schemas import ConversationTurn
from app.utils.text_cleaner import normalize_for_search

if TYPE_CHECKING:
    from app.services.llm_service import LLMService


FOLLOW_UP_MARKERS = (
    "y ",
    "tambien",
    "también",
    "eso",
    "esa",
    "ese",
    "esta",
    "este",
    "osea",
    "o sea",
    "entonces",
    "en resumen",
    "eso significa",
    "eso quiere decir",
    "nono",
    "no no",
)

TOPIC_ANCHORS = (
    "modulo",
    "sisa",
    "zona",
    "autoriz",
    "requis",
    "permiso",
    "articulo",
    "feria",
    "comercio",
    "ambulat",
    "tributo",
    "vender",
    "venta",
    "calle",
)

MEMORY_MARKERS = (
    "que te pregunte",
    "que te dije",
    "que respondiste",
    "primero",
    "anterior",
)


class QueryRewriter:
    """Rewrite follow-up questions so retrieval gets a clearer search query."""

    def __init__(self, settings: Settings, llm_service: "LLMService", logger) -> None:
        self.settings = settings
        self.llm_service = llm_service
        self.logger = logger.getChild("query_rewriter")

    def rewrite(self, question: str, history: list[ConversationTurn]) -> str:
        """Return a clearer search question using history when necessary."""

        heuristic = self._heuristic_rewrite(question, history)
        if normalize_for_search(heuristic) == normalize_for_search(question):
            # CAMBIO FASE OLLAMA 9 — Omitir reescritura LLM para consultas autonomas.
            # Motivo: Ollama local solo aporta valor al resolver referencias de seguimiento.
            # Riesgo mitigado: una repregunta detectada cambia el texto heuristico y sigue esta ruta.
            return heuristic

        if self.llm_service.client is None or self._should_skip_external_rewrite():
            return heuristic

        try:
            rewritten = self.llm_service.rewrite_query(question, history=history)
        except Exception as exc:  # pragma: no cover
            self.logger.warning("Fallo la reescritura con el proveedor externo: %s", exc)
            return heuristic

        rewritten = rewritten.strip()
        if not rewritten:
            return heuristic

        normalized_original = normalize_for_search(question)
        normalized_rewritten = normalize_for_search(rewritten)
        if normalized_rewritten == normalized_original:
            return heuristic

        return rewritten

    def _should_skip_external_rewrite(self) -> bool:
        """Avoid spending scarce free-tier calls on query rewriting."""

        if self.settings.llm_provider.lower().strip() != "openrouter":
            return False

        if self.settings.chat_model.endswith(":free"):
            return True

        return any(model.endswith(":free") for model in self.settings.chat_model_fallbacks)

    def _heuristic_rewrite(self, question: str, history: list[ConversationTurn]) -> str:
        """Use light conversational heuristics when no query-rewriter model is available."""

        if not history:
            return question

        normalized_question = normalize_for_search(question)
        tokens = normalized_question.split()
        asks_about_memory = any(marker in normalized_question for marker in MEMORY_MARKERS)
        if asks_about_memory:
            return question

        explicit_follow_up = any(normalized_question.startswith(marker) for marker in FOLLOW_UP_MARKERS)
        has_topic_anchor = any(anchor in normalized_question for anchor in TOPIC_ANCHORS)
        short_question = len(tokens) <= 5

        if not explicit_follow_up and not short_question:
            return question
        if has_topic_anchor and not explicit_follow_up:
            return question

        recent_user_turn = next(
            (
                turn.text
                for turn in reversed(history)
                if turn.role == "user"
                and normalize_for_search(turn.text) != normalized_question
            ),
            "",
        )
        if not recent_user_turn:
            return question

        return f"{recent_user_turn}. Seguimiento: {question}"
