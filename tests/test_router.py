from app.models.domain import QueryIntent
from app.services.query_router import QueryRouter


def test_router_detects_payments() -> None:
    router = QueryRouter()
    routed = router.route("Cuanto se paga de SISA para comercio ambulatorio")
    assert routed.in_domain is True
    assert routed.intent == QueryIntent.PAGOS_SISA


def test_router_rejects_out_of_scope_question() -> None:
    router = QueryRouter()
    routed = router.route("Cual es el horario de atencion del impuesto predial")
    assert routed.in_domain is False
    assert routed.intent == QueryIntent.OUT_OF_SCOPE


def test_router_accepts_generic_article_question() -> None:
    router = QueryRouter()
    routed = router.route("Dime que dice el articulo 7")
    assert routed.in_domain is True
    assert routed.intent == QueryIntent.NORMATIVA


def test_router_detects_short_art_alias_as_normative_query() -> None:
    routed = QueryRouter().route("y que dice el art 36?")

    assert routed.in_domain is True
    assert routed.intent == QueryIntent.NORMATIVA


def test_router_accepts_natural_municipal_case_without_exact_keywords() -> None:
    router = QueryRouter()
    routed = router.route("Puedo poner una carretilla para vender en la calle")
    assert routed.in_domain is True


def test_router_identifies_where_selling_is_forbidden_as_zone_query() -> None:
    router = QueryRouter()
    routed = router.route("Donde no puedo vender")

    assert routed.in_domain is True
    assert routed.intent == QueryIntent.ZONAS_RIGIDAS


def test_router_treats_exact_cost_question_as_payment_evidence_query() -> None:
    router = QueryRouter()
    routed = router.route("Cuanto cuesta exactamente este tramite actualmente?")

    assert routed.in_domain is True
    assert routed.intent == QueryIntent.PAGOS_SISA


def test_router_identifies_question_about_governing_ordinance() -> None:
    routed = QueryRouter().route("Cual es la ordenanza de comercio ambulatorio")

    assert routed.in_domain is True
    assert routed.intent == QueryIntent.NORMATIVA


def test_router_treats_getting_a_permit_as_requirements_not_validity() -> None:
    routed = QueryRouter().route("Como puedo sacar un permiso para comercio ambulatorio")

    assert routed.in_domain is True
    assert routed.intent == QueryIntent.REQUISITOS_NUEVO


def test_router_detects_definition_without_jumping_to_process_intents() -> None:
    router = QueryRouter()

    comercio = router.route("Que es comercio ambulatorio")
    tupa = router.route("Que es TUPA")
    giro = router.route("Que significa giro")

    assert comercio.in_domain is True
    assert comercio.intent == QueryIntent.DEFINICION
    assert tupa.intent == QueryIntent.DEFINICION
    assert giro.intent == QueryIntent.DEFINICION


def test_router_distinguishes_new_and_renewal_requirements() -> None:
    router = QueryRouter()

    nuevo = router.route("Soy nuevo, quiero vender en la calle")
    renovacion = router.route("Ya tengo autorizacion y quiero renovar")
    ambiguo = router.route("Requisitos de comercio ambulatorio")

    assert nuevo.intent == QueryIntent.REQUISITOS_NUEVO
    assert renovacion.intent == QueryIntent.REQUISITOS_RENOVACION
    assert ambiguo.intent == QueryIntent.REQUISITOS_AMBIGUO


def test_router_detects_sanctions_for_non_compliance() -> None:
    routed = QueryRouter().route("Que pasa si no pago la SISA")

    assert routed.in_domain is True
    assert routed.intent == QueryIntent.SANCIONES


def test_router_detects_obligations() -> None:
    routed = QueryRouter().route("Que obligaciones debo cumplir como vendedor ambulante")

    assert routed.in_domain is True
    assert routed.intent == QueryIntent.OBLIGACIONES


def test_router_detects_commerce_schedule_without_accepting_predial_schedule() -> None:
    commerce = QueryRouter().route("Cual es el horario autorizado para vender")
    predial = QueryRouter().route("Cual es el horario de atencion del impuesto predial")

    assert commerce.in_domain is True
    assert commerce.intent == QueryIntent.HORARIO
    assert predial.in_domain is False


def test_router_detects_rubros_and_giros() -> None:
    routed = QueryRouter().route("Cuales son los giros permitidos")

    assert routed.in_domain is True
    assert routed.intent == QueryIntent.RUBROS


def test_router_detects_plain_language_non_compliance() -> None:
    routed = QueryRouter().route("Que pasa si no cumplo con mi puesto")

    assert routed.in_domain is True
    assert routed.intent == QueryIntent.SANCIONES
