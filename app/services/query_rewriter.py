from __future__ import annotations

import re
from typing import TYPE_CHECKING

from app.config import Settings
from app.models.schemas import ConversationTurn
from app.utils.text_cleaner import normalize_for_search

if TYPE_CHECKING:
    from app.services.llm_service import LLMService


ARTICLE_PATTERN = re.compile(r"\b(?:art[ií]culo|art\.?)\s+([0-9]+(?:\.[0-9]+)?[A-Z]?)", re.IGNORECASE)

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

        article_match = ARTICLE_PATTERN.search(question)
        if article_match:
            return f"Que dice el Articulo {article_match.group(1)} sobre comercio ambulatorio?"

        heuristic = self._heuristic_rewrite(question, history)
        if _is_focused_follow_up(normalize_for_search(question)):
            return heuristic
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

        recent_context = normalize_for_search(recent_user_turn)
        conversation_context = normalize_for_search(" ".join(turn.text for turn in history[-6:]))
        focused_rewrite = _rewrite_focused_follow_up(question, normalized_question, history, conversation_context)
        if focused_rewrite:
            return focused_rewrite
        if any(
            marker in normalized_question
            for marker in ("explicamelo", "explicame mas simple", "mas simple", "no entiendo", "no entendi")
        ):
            return (
                f"{recent_user_turn}. Seguimiento: explicar en lenguaje mas sencillo "
                f"la respuesta anterior: {question}"
            )
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


def _is_focused_follow_up(normalized_question: str) -> bool:
    return bool(_ordinal_numbers_from_text(normalized_question)) or any(
        marker in normalized_question
        for marker in (
            "esa parte",
            "ese punto",
            "ese requisito",
            "esa condicion",
            "eso de",
            "lo de",
            "que significa",
            "que quiere decir",
            "a que se refiere",
        )
    )


def _rewrite_focused_follow_up(
    question: str,
    normalized_question: str,
    history: list[ConversationTurn],
    conversation_context: str,
) -> str:
    if not _is_focused_follow_up(normalized_question):
        return ""

    last_assistant = next((turn.text for turn in reversed(history) if turn.role == "assistant"), "")
    last_user = next((turn.text for turn in reversed(history) if turn.role == "user"), "")
    focused_items = _focused_items_from_last_answer(normalized_question, last_assistant)
    focus_phrase = _extract_focus_phrase(question)
    topic_hint = _conversation_topic_hint(conversation_context)

    if focused_items:
        return (
            f"Explicar en lenguaje ciudadano solo esta parte de la respuesta anterior sobre {topic_hint}: "
            f"{'; '.join(focused_items)}. No repetir toda la lista ni reiniciar el tramite."
        )

    if focus_phrase:
        return (
            f"Explicar en lenguaje ciudadano solo la parte '{focus_phrase}' relacionada con {topic_hint}. "
            "Usar la evidencia documental recuperada y no repetir toda la respuesta anterior."
        )

    return (
        f"{last_user}. Seguimiento enfocado: explicar solo la parte especifica que el ciudadano senala "
        f"en '{question}', sin repetir toda la respuesta anterior."
    )


def _ordinal_numbers_from_text(normalized_question: str) -> list[int]:
    numbers: list[int] = []
    ordinal_map = {
        1: ("1", "1ro", "1ero", "primer", "primero", "primera"),
        2: ("2", "2do", "2 do", "segundo", "segunda"),
        3: ("3", "3er", "3 er", "3ro", "tercer", "tercero", "tercera"),
        4: ("4", "4to", "cuarto", "cuarta"),
        5: ("5", "5to", "quinto", "quinta"),
        6: ("6", "6to", "sexto", "sexta"),
    }
    for number, markers in ordinal_map.items():
        if any(re.search(rf"\b{re.escape(marker)}\b", normalized_question) for marker in markers):
            numbers.append(number)
    return numbers


def _focused_items_from_last_answer(normalized_question: str, last_assistant: str) -> list[str]:
    requested_numbers = _ordinal_numbers_from_text(normalized_question)
    if not requested_numbers or not last_assistant:
        return []

    items = _extract_list_items(last_assistant)
    if not items:
        return []
    return [
        items[number - 1]
        for number in requested_numbers
        if 0 < number <= len(items)
    ]


def _extract_list_items(text: str) -> list[str]:
    items: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        match = re.match(r"^(?:[-*•]\s+|\d+[.)]\s+)(.+)$", stripped)
        if match:
            items.append(match.group(1).strip())
    return items


def _extract_focus_phrase(question: str) -> str:
    normalized = " ".join(question.strip().strip("¿?").split())
    patterns = (
        r"(?i)\blo de\s+(.+)$",
        r"(?i)\beso de\s+(.+)$",
        r"(?i)\bque significa\s+(.+)$",
        r"(?i)\bqué significa\s+(.+)$",
        r"(?i)\bque quiere decir\s+(.+)$",
        r"(?i)\ba que se refiere\s+(.+)$",
    )
    for pattern in patterns:
        match = re.search(pattern, normalized)
        if match:
            return match.group(1).strip(" ?.")
    return ""


def _conversation_topic_hint(conversation_context: str) -> str:
    if "feria" in conversation_context:
        return "ferias municipales"
    if "zona" in conversation_context or "ubicacion" in conversation_context:
        return "zonas o ubicacion de comercio ambulatorio"
    if "sisa" in conversation_context or "pago" in conversation_context or "costo" in conversation_context:
        return "pagos o costos municipales"
    if "renov" in conversation_context or "voucher" in conversation_context:
        return "renovacion de autorizacion de comercio ambulatorio"
    if "giro" in conversation_context or "rubro" in conversation_context:
        return "giros o rubros de comercio ambulatorio"
    if "obligacion" in conversation_context or "sancion" in conversation_context or "incumpl" in conversation_context:
        return "obligaciones o consecuencias del comercio ambulatorio"
    if "requisito" in conversation_context or "padron" in conversation_context or "primera vez" in conversation_context:
        return "tramite nuevo de comercio ambulatorio"
    return "comercio ambulatorio"


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
