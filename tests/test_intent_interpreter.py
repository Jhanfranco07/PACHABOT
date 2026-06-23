from app.core.logger import setup_logging
from app.models.domain import QueryIntent
from app.models.schemas import ConversationTurn, RoutedQuery
from app.services.intent_interpreter import IntentInterpreterService
from app.services.query_router import QueryRouter


class DummyInterpreterLLM:
    def __init__(self, result=None) -> None:
        self.result = result or {}
        self.received_question = None
        self.received_history = None
        self.received_hint = None

    def interpret_intent(self, question, *, history=None, router_hint=""):
        self.received_question = question
        self.received_history = history
        self.received_hint = router_hint
        return self.result


def test_intent_interpreter_trusts_confident_article_query() -> None:
    router = QueryRouter()
    llm = DummyInterpreterLLM()
    service = IntentInterpreterService(router, llm, setup_logging("INFO"))
    routed = router.route("y que dice el art 36?")

    interpretation = service.interpret("y que dice el art 36?", routed, [])

    assert interpretation.routed_query.intent == QueryIntent.NORMATIVA
    assert interpretation.routed_query.needs_clarification is False
    assert interpretation.used_llm is False
    assert llm.received_question is None


def test_intent_interpreter_asks_clarification_for_ambiguous_money_follow_up() -> None:
    router = QueryRouter()
    service = IntentInterpreterService(router, DummyInterpreterLLM(), setup_logging("INFO"))
    routed = router.route("y cuanto es")
    history = [
        ConversationTurn(role="user", text="Como renuevo mi permiso?"),
        ConversationTurn(role="assistant", text="Para renovar debes presentar voucher de pago SISA."),
    ]

    interpretation = service.interpret("y cuanto es", routed, history)

    assert interpretation.routed_query.needs_clarification is True
    assert "TUPA" in interpretation.routed_query.clarification_question
    assert "SISA" in interpretation.routed_query.clarification_question


def test_intent_interpreter_uses_llm_when_router_is_uncertain() -> None:
    llm = DummyInterpreterLLM(
        {
            "intent": "pagos_sisa",
            "confidence": 0.62,
            "normalized_query": "Consulta sobre pago de SISA",
            "needs_clarification": True,
            "clarification_question": "¿Te refieres al pago SISA o al costo del trámite?",
        }
    )
    service = IntentInterpreterService(QueryRouter(), llm, setup_logging("INFO"))
    routed = RoutedQuery(
        original_query="la plata de eso",
        normalized_query="la plata de eso",
        intent=QueryIntent.GENERAL,
        in_domain=True,
        confidence=0.35,
    )

    interpretation = service.interpret("la plata de eso", routed, [])

    assert interpretation.used_llm is True
    assert interpretation.routed_query.intent == QueryIntent.PAGOS_SISA
    assert interpretation.routed_query.needs_clarification is True
    assert interpretation.routed_query.clarification_question.startswith("¿Te refieres")
