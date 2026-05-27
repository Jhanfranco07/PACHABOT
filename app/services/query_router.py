from __future__ import annotations

import re

from app.models.domain import QueryIntent
from app.models.schemas import RoutedQuery
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
    "sisa",
    "autorizacion",
    "permiso",
    "feria",
    "zona rigida",
    "vendedor",
    "puesto",
    "ordenanza",
    "articulo",
    "municipalidad",
)

INTENT_HINTS: dict[QueryIntent, tuple[str, ...]] = {
    QueryIntent.REQUISITOS: (
        "requisito",
        "requisitos",
        "solicitar",
        "tramite",
        "documentos",
        "que necesito",
        "que debo hacer",
        "como empiezo",
        "como puedo vender",
    ),
    QueryIntent.MODULOS: ("modulo", "medida", "dimensiones", "tamano", "puesto"),
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
    QueryIntent.FERIAS: ("feria", "ferias", "eventual", "temporal"),
    QueryIntent.PROHIBICIONES: ("prohibido", "prohibiciones", "impedido", "restriccion", "sancion"),
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

        for hint in DOMAIN_HINTS:
            if _contains_keyword(normalized, hint):
                matched_keywords.append(hint)
                domain_score += 1.0

        if any(_contains_keyword(normalized, pattern) for pattern in LEGAL_PATTERNS):
            domain_score += 1.0

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
        if any(
            _contains_keyword(normalized, marker)
            for marker in ("cuanto cuesta", "costo", "tasa", "monto", "sisa")
        ):
            if intent_scores[QueryIntent.PAGOS_SISA] > 0:
                best_intent = QueryIntent.PAGOS_SISA
                best_score = intent_scores[QueryIntent.PAGOS_SISA]
        if "sisa" in normalized and any(
            marker in normalized for marker in ("no pago", "no pagar", "incumpl", "deuda")
        ):
            best_intent = QueryIntent.PROHIBICIONES
            best_score = max(best_score, 2.0)

        in_domain = domain_score > 0 or best_score > 0
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
