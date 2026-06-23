from __future__ import annotations

from app.models.domain import QueryIntent
from app.utils.text_cleaner import normalize_for_search


QUERY_EXPANSION_GROUPS: dict[str, tuple[str, ...]] = {
    "comercio_ambulatorio": (
        "comercio ambulatorio",
        "comercio en via publica",
        "venta ambulante",
        "vender en la calle",
        "ambulante",
        "comerciante ambulante",
        "comerciante autorizado",
    ),
    "autorizacion": (
        "autorizacion municipal",
        "permiso",
        "resolucion de autorizacion",
        "documento de autorizacion",
        "autorizado",
        "evaluacion municipal",
    ),
    "modulo": (
        "modulo",
        "puesto",
        "stand",
        "mobiliario",
        "espacio autorizado",
        "ubicacion autorizada",
    ),
    "incumplimiento": (
        "incumplimiento",
        "no cumple",
        "no cumplir",
        "no respetar",
        "infraccion",
        "sancion",
        "revocacion",
        "retiro",
        "desmontaje",
        "fiscalizacion",
    ),
    "obligaciones": (
        "obligaciones",
        "deberes",
        "condiciones",
        "cumplir",
        "comerciante autorizado",
        "feriante autorizado",
        "articulo 57",
    ),
    "zonas": (
        "zona rigida",
        "zona prohibida",
        "zona restringida",
        "ubicacion",
        "calle",
        "avenida",
        "tramo",
        "jr miguel grau",
        "av manchay",
        "av victor malasquez",
        "av 1 de mayo",
        "calle 57",
    ),
    "ferias": (
        "feria",
        "feriante",
        "recinto ferial",
        "stand",
        "promocion comercial",
        "autorizacion de feria",
    ),
    "servicios_feria": (
        "servicios higienicos",
        "banos",
        "baños",
        "agua",
        "limpieza",
        "salubridad",
        "articulo 62",
    ),
    "horario": (
        "horario",
        "hora",
        "jornada",
        "atiende",
        "funcionamiento",
        "articulo 61",
    ),
    "costos": (
        "costo",
        "cuanto cuesta",
        "monto",
        "tasa",
        "pago",
        "tupa",
        "derecho de tramite",
    ),
    "requisitos": (
        "requisitos",
        "documentos",
        "que necesito",
        "solicitud",
        "declaracion jurada",
        "fotografias",
        "padron municipal",
        "articulo 30",
    ),
    "requisitos_nuevo": (
        "tramite nuevo",
        "ingreso al padron municipal",
        "primera vez",
        "soy nuevo",
        "quiero vender",
        "vender en la calle",
        "vender en la via publica",
        "vender en via publica",
        "permiso para vender",
        "que necesito y cuanto cuesta",
        "solicitud de ingreso al padron",
        "foto panoramica",
    ),
    "requisitos_renovacion": (
        "renovacion",
        "renovar",
        "ya tengo permiso",
        "ya tengo autorizacion",
        "permiso por vencer",
        "autorizacion vence",
        "seguir vendiendo",
        "voucher",
        "ultimo comprobante de pago",
        "dos fotos tamano carne",
    ),
    "definiciones": (
        "definicion",
        "concepto",
        "que es",
        "que significa",
        "comercio ambulatorio",
        "padron municipal",
        "tupa",
        "sisa",
        "modulo",
        "giro",
    ),
    "productos_industriales": (
        "productos de origen industrial",
        "producto industrial",
        "productos envasados",
        "golosinas",
        "galletas",
        "caramelos",
        "chocolates",
        "confites",
        "snacks envasados",
        "registro sanitario",
        "fecha de vencimiento",
        "g001",
    ),
    "productos_perecibles": (
        "productos perecibles",
        "frutas",
        "verduras",
        "productos naturales",
        "g002",
        "g003",
    ),
    "productos_preparados": (
        "productos preparados",
        "preparados al dia",
        "emoliente",
        "quinua",
        "maca",
        "soya",
        "potajes",
        "dulces tradicionales",
        "sandwiches",
        "sanguche",
        "jugo de naranja",
        "canchita",
        "confiteria",
        "g004",
        "g005",
        "g006",
        "g007",
        "g008",
        "g009",
    ),
    "objetos_duraderos": (
        "objetos de uso duradero",
        "merceria",
        "bazar",
        "utiles de escritorio",
        "diarios",
        "revistas",
        "libros",
        "loterias",
        "monedas",
        "estampillas",
        "artesanias",
        "articulos religiosos",
        "articulos de limpieza",
        "pilas",
        "relojes",
    ),
    "servicios_comercio_ambulatorio": (
        "duplicado de llaves",
        "cerrajeria",
        "lustrado de calzado",
        "lustradores",
        "artistas plasticos",
        "retratistas",
        "fotografias",
    ),
}


INTENT_EXPANSION_GROUPS: dict[QueryIntent, tuple[str, ...]] = {
    QueryIntent.REQUISITOS: ("comercio_ambulatorio", "autorizacion", "requisitos"),
    QueryIntent.REQUISITOS_NUEVO: (
        "comercio_ambulatorio",
        "autorizacion",
        "requisitos",
        "requisitos_nuevo",
    ),
    QueryIntent.REQUISITOS_RENOVACION: (
        "comercio_ambulatorio",
        "autorizacion",
        "requisitos",
        "requisitos_renovacion",
    ),
    QueryIntent.REQUISITOS_AMBIGUO: (
        "comercio_ambulatorio",
        "autorizacion",
        "requisitos",
    ),
    QueryIntent.PAGOS_SISA: ("costos", "autorizacion"),
    QueryIntent.ZONAS_RIGIDAS: ("zonas", "comercio_ambulatorio"),
    QueryIntent.AUTORIZACIONES: ("autorizacion", "comercio_ambulatorio"),
    QueryIntent.RUBROS: (
        "comercio_ambulatorio",
        "autorizacion",
        "productos_industriales",
        "productos_perecibles",
        "productos_preparados",
        "objetos_duraderos",
        "servicios_comercio_ambulatorio",
    ),
    QueryIntent.FERIAS: ("ferias", "autorizacion"),
    QueryIntent.OBLIGACIONES: ("obligaciones", "comercio_ambulatorio"),
    QueryIntent.PROHIBICIONES: ("incumplimiento", "obligaciones"),
    QueryIntent.SANCIONES: ("incumplimiento", "obligaciones"),
    QueryIntent.REVOCACION: ("incumplimiento", "autorizacion", "modulo"),
    QueryIntent.HORARIO: ("horario", "ferias", "comercio_ambulatorio"),
    QueryIntent.UBICACION: ("zonas", "autorizacion"),
    QueryIntent.NORMATIVA: ("comercio_ambulatorio", "autorizacion"),
}


def expansion_groups_for_query(query: str) -> list[str]:
    """Return semantic groups that match the citizen's wording."""

    normalized_query = normalize_for_search(query)
    matched: list[str] = []
    for group_name, terms in QUERY_EXPANSION_GROUPS.items():
        if any(normalize_for_search(term) in normalized_query for term in terms):
            matched.append(group_name)
    return matched


def expansion_terms(
    query: str,
    intent: QueryIntent | None = None,
    *,
    max_terms_per_group: int = 8,
) -> list[str]:
    """Return related terms useful for retrieval, deduplicated and ordered."""

    group_names = [
        *expansion_groups_for_query(query),
        *INTENT_EXPANSION_GROUPS.get(intent, ()),
    ]
    terms: list[str] = []
    seen: set[str] = set()
    for group_name in group_names:
        for term in QUERY_EXPANSION_GROUPS.get(group_name, ())[:max_terms_per_group]:
            normalized = normalize_for_search(term)
            if normalized and normalized not in seen:
                seen.add(normalized)
                terms.append(term)
    return terms


def expand_query(query: str, intent: QueryIntent | None = None) -> str:
    """Append related municipal vocabulary to improve lexical retrieval."""

    terms = expansion_terms(query, intent)
    if not terms:
        return query
    return " ".join([query, *terms])


def related_search_queries(query: str, intent: QueryIntent | None = None) -> list[str]:
    """Build staged search queries before the system gives up and falls back."""

    queries = [query]
    terms = expansion_terms(query, intent)
    if terms:
        queries.append(" ".join([query, *terms[:10]]))

    for group_name in INTENT_EXPANSION_GROUPS.get(intent, ()):
        group_terms = QUERY_EXPANSION_GROUPS.get(group_name, ())
        if group_terms:
            queries.append(" ".join([query, *group_terms[:8]]))

    return _dedupe_preserving_order(queries)


def _dedupe_preserving_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        normalized = normalize_for_search(value)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(value)
    return ordered
