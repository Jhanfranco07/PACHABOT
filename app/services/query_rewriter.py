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
    "y cuanto",
    "y que",
    "y en",
    "si es en",
    "si fuera en",
    "tambien aplica",
    "tambien para",
    "aplica para",
    "que pasa si",
    "que pasa si no",
    "en resumen",
    "eso significa",
    "eso quiere decir",
    "explicamelo",
    "explicame mas simple",
    "mas simple",
    "no entiendo",
    "no entendi",
    "nono",
    "no no",
    "en que articulo",
    "que articulo",
    "donde dice",
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
    "manchay",
    "miguel grau",
    "zona rigida",
    "incumpl",
    "cumple",
    "cumplo",
    "revoc",
    "retiro",
    "quitar",
    "sancion",
    "costo",
    "comercio",
    "ambulat",
    "tributo",
    "vender",
    "venta",
    "calle",
    "renov",
    "voucher",
    "padron",
    "nuevo",
    "dni",
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

        pending_query = _query_from_pending_options(normalized_question, history)
        if pending_query:
            return pending_query

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

        if any(
            marker in normalized_question
            for marker in ("explicamelo", "explicame mas simple", "mas simple", "no entiendo", "no entendi")
        ):
            return (
                f"{recent_user_turn}. Seguimiento: explicar en lenguaje mas sencillo "
                f"la respuesta anterior: {question}"
            )

        recent_context = normalize_for_search(recent_user_turn)
        if any(marker in normalized_question for marker in ("que llevo", "que papeles", "papeles llevo", "documentos llevo")):
            if _looks_like_renewal_context(recent_context):
                return "Que documentos debe llevar para renovar una autorizacion de comercio ambulatorio?"
            if _looks_like_new_context(recent_context):
                return (
                    "Que requisitos debe presentar una persona nueva para solicitar "
                    "ingreso al padron municipal de comercio ambulatorio?"
                )
        if any(marker in normalized_question for marker in ("cuanto cuesta", "costo", "monto")):
            if _looks_like_renewal_context(recent_context):
                return "Cual es el costo de renovacion de comercio ambulatorio segun el TUPA vigente?"
            if _looks_like_new_context(recent_context):
                return "Cual es el costo del tramite nuevo de comercio ambulatorio segun el TUPA vigente?"

        return f"{recent_user_turn}. Seguimiento: {question}"


def _looks_like_new_context(normalized_text: str) -> bool:
    return any(
        marker in normalized_text
        for marker in (
            "primera vez",
            "soy nuevo",
            "sacar permiso",
            "como saco",
            "quiero vender",
            "vender en la calle",
            "vender en la via publica",
            "ingreso al padron",
            "inscribirme",
        )
    )


def _looks_like_renewal_context(normalized_text: str) -> bool:
    return any(
        marker in normalized_text
        for marker in (
            "renovar",
            "renovacion",
            "ya tengo permiso",
            "ya tengo autorizacion",
            "permiso vence",
            "permiso esta por vencer",
            "seguir vendiendo",
            "voucher",
        )
    )


def _query_from_pending_options(
    normalized_question: str,
    history: list[ConversationTurn],
) -> str:
    last_assistant = next((turn for turn in reversed(history) if turn.role == "assistant"), None)
    if last_assistant is None:
        return ""
    pending_options = last_assistant.metadata.get("pending_options", {})
    if not isinstance(pending_options, dict):
        return ""

    normalized = normalized_question.strip()
    for key, option in pending_options.items():
        if not isinstance(option, dict):
            continue
        query = str(option.get("query", "")).strip()
        if not query:
            continue
        if normalized == normalize_for_search(str(key)):
            return query
        aliases = option.get("aliases", [])
        if any(
            normalized == normalize_for_search(str(alias))
            or normalize_for_search(str(alias)) in normalized
            for alias in aliases
        ):
            return query
    return ""
