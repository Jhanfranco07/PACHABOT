from __future__ import annotations

import re

from app.models.domain import QueryIntent
from app.models.schemas import RoutedQuery
from app.utils.query_expansion import expansion_groups_for_query
from app.utils.text_cleaner import normalize_for_search


DOMAIN_HINTS = (
    "comercio ambulatorio",
    "ambulatorio",
    "ambulante",
    "vender",
    "venta",
    "via publica",
    "espacio publico",
    "calle",
    "carretilla",
    "modulo",
    "padron",
    "padrón",
    "tupa",
    "sisa",
    "autorizacion",
    "permiso",
    "giro",
    "giros",
    "rubro",
    "rubros",
    "actividad permitida",
    "actividades permitidas",
    "feria",
    "zona rigida",
    "vendedor",
    "puesto",
    "stand",
    "ordenanza",
    "articulo",
    "municipalidad",
    "obligacion",
    "obligaciones",
    "incumplimiento",
    "no cumple",
    "no cumplo",
    "sancion",
    "revocacion",
    "fiscalizacion",
    "retiro",
    "quitar",
    "banos",
    "baños",
    "servicios higienicos",
    "horario autorizado",
    "zona restringida",
    "zona prohibida",
)

INTENT_HINTS: dict[QueryIntent, tuple[str, ...]] = {
    QueryIntent.DEFINICION: (
        "que es",
        "que significa",
        "definicion",
        "explicame que es",
        "a que se refiere",
        "se entiende por",
    ),
    QueryIntent.REQUISITOS: (
        "requisito",
        "requisitos",
        "solicitar",
        "sacar",
        "sacar permiso",
        "obtener permiso",
        "obtener autorizacion",
        "tramite",
        "tramitar",
        "documentos",
        "que necesito",
        "que debo hacer",
        "como empiezo",
        "como puedo sacar",
        "como saco",
        "como tramito",
        "como puedo vender",
    ),
    QueryIntent.REQUISITOS_NUEVO: (
        "sacar permiso",
        "sacar mi permiso",
        "como saco permiso",
        "como saco mi permiso",
        "como puedo sacar",
        "obtener permiso",
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
        "q necesito pa vender",
        "que necesito y cuanto cuesta",
    ),
    QueryIntent.REQUISITOS_RENOVACION: (
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
    ),
    QueryIntent.REQUISITOS_AMBIGUO: (
        "requisitos de comercio ambulatorio",
        "que requisitos son",
        "que documentos necesito",
        "documentos para comercio ambulatorio",
        "requisitos para comercio ambulatorio",
    ),
    QueryIntent.MODULOS: ("modulo", "medida", "dimensiones", "tamano", "puesto", "stand"),
    QueryIntent.PAGOS_SISA: ("sisa", "pago", "cuota", "monto", "tasa", "cuanto cuesta"),
    QueryIntent.ZONAS_RIGIDAS: (
        "zona rigida",
        "zonas rigidas",
        "zona prohibida",
        "restriccion de zona",
        "donde no puedo vender",
        "donde puedo vender",
        "ubicacion",
        "miguel grau",
        "manchay",
    ),
    QueryIntent.AUTORIZACIONES: ("autorizacion", "permiso", "vigencia", "renovacion", "puedo vender"),
    QueryIntent.RUBROS: (
        "giro",
        "giros",
        "rubro",
        "rubros",
        "actividad permitida",
        "actividades permitidas",
        "que puedo vender",
        "que productos puedo vender",
        "que se puede vender",
        "venta permitida",
        "rubros permitidos",
        "giros permitidos",
    ),
    QueryIntent.FERIAS: ("feria", "ferias", "eventual", "temporal"),
    QueryIntent.OBLIGACIONES: ("obligacion", "obligaciones", "debo cumplir", "que debo cumplir", "cumplir"),
    QueryIntent.PROHIBICIONES: ("prohibido", "prohibiciones", "impedido", "restriccion"),
    QueryIntent.SANCIONES: (
        "sancion",
        "sanciones",
        "multa",
        "decomiso",
        "incumplo",
        "incumple",
        "no cumplo",
        "no cumple",
        "no respetar",
        "no respeto",
        "fiscalizacion",
        "quitar",
        "retirar",
        "retiro",
    ),
    QueryIntent.REVOCACION: ("revocacion", "revocatoria", "retiro definitivo", "quitar autorizacion"),
    QueryIntent.HORARIO: ("horario autorizado", "horario de venta", "a que hora puedo vender"),
    QueryIntent.UBICACION: ("ubicacion exacta", "direccion", "donde queda", "oficina", "sede"),
    QueryIntent.NORMATIVA: ("ordenanza", "norma aplicable", "reglamento", "que norma"),
}

LEGAL_PATTERNS = (
    "que dice",
    "que indica",
    "que senala",
    "articulo",
    "ordenanza",
    "norma",
)


class QueryRouter:
    """Soft router used as a hint provider, not as a hard gatekeeper."""

    def route(self, question: str) -> RoutedQuery:
        normalized = normalize_for_search(question)
        matched_keywords: list[str] = []
        intent_scores = {intent: 0.0 for intent in INTENT_HINTS}
        domain_score = 0.0
        unrelated_municipal_topic = any(
            marker in normalized
            for marker in ("predial", "arbitrios", "licencia de funcionamiento", "catastro")
        ) and not any(marker in normalized for marker in ("ambulator", "vender", "via publica", "sisa"))

        for hint in DOMAIN_HINTS:
            if _contains_keyword(normalized, hint):
                matched_keywords.append(hint)
                domain_score += 1.0

        semantic_groups = expansion_groups_for_query(question)
        domain_semantic_groups = [
            group for group in semantic_groups if group not in {"horario"}
        ]
        if domain_semantic_groups:
            domain_score += 0.6
            matched_keywords.extend(domain_semantic_groups)

        if any(_contains_keyword(normalized, pattern) for pattern in LEGAL_PATTERNS):
            domain_score += 1.0

        definition_query = any(
            marker in normalized
            for marker in (
                "que es",
                "que significa",
                "definicion",
                "explicame que es",
                "a que se refiere",
                "se entiende por",
            )
        )

        if any(
            pattern in normalized
            for pattern in (
                "que necesito para vender",
                "puedo vender",
                "vender en la calle",
                "salir a vender",
                "comerciante informal",
            )
        ):
            domain_score += 1.5

        for intent, hints in INTENT_HINTS.items():
            for hint in hints:
                if _contains_keyword(normalized, hint):
                    intent_scores[intent] += 1.0
                    matched_keywords.append(hint)

        best_intent = QueryIntent.GENERAL
        best_score = 0.0
        for intent, score in intent_scores.items():
            if score > best_score:
                best_intent = intent
                best_score = score

        # CAMBIO FASE OLLAMA 4 — Separar preguntas de costo de requisitos generales.
        # Motivo: un costo exacto requiere evidencia monetaria, no cualquier tramite relacionado.
        # Riesgo mitigado: las consultas SISA existentes conservan el mismo intent.
        asks_cost = any(
            _contains_keyword(normalized, marker)
            for marker in ("cuanto cuesta", "costo", "tasa", "monto", "sisa")
        )
        asks_requirements = any(
            marker in normalized
            for marker in ("requisit", "documento", "que necesito", "que llevo", "papeles")
        )

        if asks_cost and not asks_requirements:
            if intent_scores[QueryIntent.PAGOS_SISA] > 0:
                best_intent = QueryIntent.PAGOS_SISA
                best_score = intent_scores[QueryIntent.PAGOS_SISA]

        new_score = _requirement_case_score(
            normalized,
            (
                "sacar permiso",
                "sacar mi permiso",
                "como saco permiso",
                "como saco mi permiso",
                "como puedo sacar",
                "obtener permiso",
                "primera vez",
                "soy nuevo",
                "quiero vender",
                "que necesito para vender",
                "vender en la calle",
                "vender en la via publica",
                "vender en via publica",
                "permiso para vender",
                "necesito permiso para vender",
                "ingresar al padron",
                "ingreso al padron",
                "inscribirme",
                "q necesito pa vender",
                "que necesito y cuanto cuesta",
                "pa vender",
            ),
        )
        renewal_score = _requirement_case_score(
            normalized,
            (
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
            ),
        )
        generic_requirements = normalized in {"requisitos", "documentos", "papeles", "que requisitos"} or any(
            marker in normalized
            for marker in (
                "requisitos de comercio ambulatorio",
                "requisitos para comercio ambulatorio",
                "que requisitos son",
                "que documentos necesito",
                "documentos para comercio ambulatorio",
                "papeles para comercio ambulatorio",
            )
        )
        if renewal_score > 0:
            best_intent = QueryIntent.REQUISITOS_RENOVACION
            best_score = max(best_score, 3.0 + renewal_score)
        elif new_score > 0:
            best_intent = QueryIntent.REQUISITOS_NUEVO
            best_score = max(best_score, 3.0 + new_score)
        elif generic_requirements:
            best_intent = QueryIntent.REQUISITOS_AMBIGUO
            best_score = max(best_score, 2.5)

        if definition_query and _definition_topic_in_domain(normalized):
            best_intent = QueryIntent.DEFINICION
            best_score = max(best_score, 3.5)

        if not definition_query and any(
            _contains_keyword(normalized, marker)
            for marker in ("cuanto cuesta", "costo", "tasa", "monto", "sisa")
        ) and not asks_requirements:
            if intent_scores[QueryIntent.PAGOS_SISA] > 0:
                best_intent = QueryIntent.PAGOS_SISA
                best_score = intent_scores[QueryIntent.PAGOS_SISA]
        if not definition_query and "sisa" in normalized and any(
            marker in normalized for marker in ("no pago", "no pagar", "incumpl", "deuda", "no cumplo")
        ):
            best_intent = QueryIntent.SANCIONES
            best_score = max(best_score, 2.0)
        if not definition_query and any(
            marker in normalized
            for marker in (
                "multa",
                "decomiso",
                "que pasa si incumplo",
                "que pasa si no cumplo",
                "que pasa si no cumple",
                "no cumple",
                "no cumplo",
                "no respet",
                "me pueden quitar",
                "quitar mi puesto",
                "retirar mi puesto",
            )
        ):
            best_intent = QueryIntent.SANCIONES
            best_score = max(best_score, 2.0)
        if not definition_query and any(marker in normalized for marker in ("revocacion", "revocatoria", "retiro definitivo")):
            best_intent = QueryIntent.REVOCACION
            best_score = max(best_score, 2.0)
        if not definition_query and any(marker in normalized for marker in ("obligacion", "obligaciones", "que debo cumplir")):
            best_intent = QueryIntent.OBLIGACIONES
            best_score = max(best_score, 2.0)
        if not definition_query and any(marker in normalized for marker in ("horario autorizado", "horario de venta", "a que hora puedo vender")):
            best_intent = QueryIntent.HORARIO
            best_score = max(best_score, 2.0)
        if (
            not definition_query
            and any(marker in normalized for marker in ("ubicacion exacta", "direccion", "oficina", "sede"))
            and not any(marker in normalized for marker in ("zona", "miguel grau", "manchay"))
        ):
            best_intent = QueryIntent.UBICACION
            best_score = max(best_score, 2.0)
        if not definition_query and any(
            marker in normalized
            for marker in (
                "sacar permiso",
                "sacar mi permiso",
                "obtener permiso",
                "obtener autorizacion",
                "como puedo sacar",
                "como saco",
                "como tramito",
            )
        ) and not any(marker in normalized for marker in ("cuanto dura", "vigencia", "renovacion")):
            best_intent = QueryIntent.REQUISITOS_NUEVO
            best_score = max(best_score, 2.0)
        if not definition_query and any(
            marker in normalized
            for marker in (
                "giro",
                "giros",
                "rubro",
                "rubros",
                "actividad permitida",
                "actividades permitidas",
                "que puedo vender",
                "que productos puedo vender",
                "que se puede vender",
            )
        ):
            best_intent = QueryIntent.RUBROS
            best_score = max(best_score, 2.0)

        in_domain = domain_score > 0 or best_score > 0
        if unrelated_municipal_topic:
            in_domain = False
        if not in_domain:
            best_intent = QueryIntent.OUT_OF_SCOPE

        return RoutedQuery(
            original_query=question,
            normalized_query=normalized,
            intent=best_intent,
            in_domain=in_domain,
            matched_keywords=list(dict.fromkeys(matched_keywords)),
        )


def _contains_keyword(text: str, keyword: str) -> bool:
    """Match full words or phrases to avoid accidental substring hits."""

    pattern = r"(?<!\w)" + re.escape(keyword) + r"(?!\w)"
    return re.search(pattern, text) is not None


def _requirement_case_score(text: str, markers: tuple[str, ...]) -> float:
    """Score broad requirement-case hints without deciding the final answer text."""

    return float(sum(1 for marker in markers if marker in text))


def _definition_topic_in_domain(text: str) -> bool:
    """Allow conceptual municipal questions without turning them into procedures."""

    return any(
        marker in text
        for marker in (
            "comercio ambulatorio",
            "ambulatorio",
            "comerciante",
            "padron",
            "padrón",
            "tupa",
            "sisa",
            "modulo",
            "giro",
            "zona rigida",
            "autorizacion municipal",
            "permiso municipal",
            "via publica",
        )
    )
