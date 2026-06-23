from __future__ import annotations

from dataclasses import dataclass

from app.models.domain import QueryIntent
from app.models.schemas import ConversationTurn, RoutedQuery
from app.services.llm_service import LLMService
from app.services.query_router import QueryRouter
from app.utils.text_cleaner import normalize_for_search


@dataclass(slots=True)
class IntentInterpretation:
    routed_query: RoutedQuery
    used_llm: bool = False


class IntentInterpreterService:
    """Resolve uncertain routing with history, clarification and optional LLM help."""

    def __init__(self, router: QueryRouter, llm_service: LLMService, logger) -> None:
        self.router = router
        self.llm_service = llm_service
        self.logger = logger.getChild("intent_interpreter")

    def interpret(
        self,
        question: str,
        routed_query: RoutedQuery,
        history: list[ConversationTurn],
    ) -> IntentInterpretation:
        normalized = normalize_for_search(question)

        clarification = self._heuristic_clarification(normalized, routed_query, history)
        if clarification:
            return IntentInterpretation(
                _with_clarification(routed_query, clarification),
                used_llm=False,
            )

        if self._router_is_confident(routed_query):
            return IntentInterpretation(routed_query, used_llm=False)

        llm_result = self._interpret_with_llm(question, routed_query, history)
        if llm_result:
            return llm_result

        return IntentInterpretation(routed_query, used_llm=False)

    def _router_is_confident(self, routed_query: RoutedQuery) -> bool:
        if routed_query.needs_clarification:
            return False
        if routed_query.intent in {
            QueryIntent.NORMATIVA,
            QueryIntent.DEFINICION,
            QueryIntent.REQUISITOS_NUEVO,
            QueryIntent.REQUISITOS_RENOVACION,
            QueryIntent.REQUISITOS_AMBIGUO,
            QueryIntent.ZONAS_RIGIDAS,
            QueryIntent.RUBROS,
            QueryIntent.SANCIONES,
            QueryIntent.OBLIGACIONES,
        }:
            return routed_query.confidence >= 0.75
        return routed_query.confidence >= 0.82

    def _heuristic_clarification(
        self,
        normalized_question: str,
        routed_query: RoutedQuery,
        history: list[ConversationTurn],
    ) -> str:
        if routed_query.intent == QueryIntent.REQUISITOS_AMBIGUO:
            return (
                "¿Es la primera vez que vas a solicitar el permiso o ya tienes "
                "autorización y quieres renovarla? Los requisitos cambian según el caso."
            )

        if _looks_like_ambiguous_money_question(normalized_question):
            context = normalize_for_search(" ".join(turn.text for turn in history[-6:]))
            has_tramite_context = any(
                marker in context
                for marker in ("renov", "permiso", "autorizacion", "tramite", "tupa", "voucher")
            )
            has_sisa_context = "sisa" in context or "pago" in context
            if has_tramite_context and has_sisa_context:
                return (
                    "¿Te refieres al costo del trámite según el TUPA o al pago de SISA? "
                    "Son pagos distintos y conviene revisarlos por separado."
                )

        if _looks_like_vague_follow_up(normalized_question) and history:
            return (
                "Para orientarte mejor, ¿te refieres a requisitos, costo, zona, giro "
                "o a algún artículo de la ordenanza?"
            )

        return ""

    def _interpret_with_llm(
        self,
        question: str,
        routed_query: RoutedQuery,
        history: list[ConversationTurn],
    ) -> IntentInterpretation | None:
        router_hint = (
            f"intent={routed_query.intent.value}; confidence={routed_query.confidence:.2f}; "
            f"in_domain={routed_query.in_domain}; keywords={', '.join(routed_query.matched_keywords[:8])}"
        )
        result = self.llm_service.interpret_intent(
            question,
            history=history,
            router_hint=router_hint,
        )
        if not result:
            return None

        try:
            intent = QueryIntent(str(result.get("intent", routed_query.intent.value)))
        except ValueError:
            intent = routed_query.intent
        confidence = _coerce_confidence(result.get("confidence"), routed_query.confidence)
        needs_clarification = bool(result.get("needs_clarification", False))
        clarification_question = str(result.get("clarification_question", "")).strip()
        interpreted_query = str(result.get("normalized_query", "")).strip()
        if needs_clarification and not clarification_question:
            clarification_question = (
                "¿Podrías precisarme un poco más tu consulta para orientarte mejor?"
            )

        interpreted = RoutedQuery(
            original_query=routed_query.original_query,
            normalized_query=normalize_for_search(interpreted_query or routed_query.original_query),
            intent=intent,
            in_domain=intent != QueryIntent.OUT_OF_SCOPE,
            matched_keywords=routed_query.matched_keywords,
            confidence=confidence,
            needs_clarification=needs_clarification,
            clarification_question=clarification_question,
            interpreted_query=interpreted_query,
        )
        return IntentInterpretation(interpreted, used_llm=True)


def _with_clarification(routed_query: RoutedQuery, question: str) -> RoutedQuery:
    return RoutedQuery(
        original_query=routed_query.original_query,
        normalized_query=routed_query.normalized_query,
        intent=routed_query.intent,
        in_domain=routed_query.in_domain,
        matched_keywords=routed_query.matched_keywords,
        confidence=min(routed_query.confidence, 0.55),
        needs_clarification=True,
        clarification_question=question,
        interpreted_query=routed_query.interpreted_query,
    )


def _looks_like_ambiguous_money_question(normalized_question: str) -> bool:
    return normalized_question in {
        "cuanto es",
        "y cuanto es",
        "cuanto pago",
        "y cuanto pago",
        "cuanto seria",
        "y cuanto seria",
        "lo de la plata",
        "y la plata",
        "y el pago",
    }


def _looks_like_vague_follow_up(normalized_question: str) -> bool:
    return normalized_question in {
        "y eso",
        "eso",
        "y ahora",
        "como seria",
        "y como es",
        "que hago",
        "y que hago",
        "explicame",
    }


def _coerce_confidence(value, default: float) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        confidence = default
    return max(0.0, min(1.0, confidence))
