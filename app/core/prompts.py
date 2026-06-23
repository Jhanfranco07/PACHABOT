from __future__ import annotations

from typing import Iterable

from app.models.schemas import ConversationTurn, RetrievedChunk
from app.prompts.templates import (
    ANTIHALLUCINATION_INSTRUCTION,
    CITATION_FORMAT,
    EVIDENCE_CHECK_PROMPT,
    GENERAL_CHAT_SYSTEM_PROMPT,
    INTENT_INTERPRETATION_SYSTEM_PROMPT,
    LEGACY_SYSTEM_PROMPT,
    NO_INFO_PROMPT,
    QUERY_REWRITE_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
)


def build_context_block(chunks: Iterable[RetrievedChunk]) -> str:
    """Render a compact reference block for the answer model."""

    parts: list[str] = []
    for chunk in chunks:
        header = chunk.source_title
        if chunk.article_label:
            header += f" - Articulo {chunk.article_label}"
        elif chunk.section_title:
            header += f" - {chunk.section_title}"
        header += f" - Estado: {chunk.vigencia.upper()}"
        header += f" - Capa: {chunk.knowledge_layer.upper()}"
        if chunk.fuente:
            header += f" - Sustento: {chunk.fuente}"
        if chunk.requires_review:
            header += " - ADVERTENCIA: PENDIENTE DE VALIDACION HUMANA"
        parts.append(f"[FUENTE: {header}]\n{chunk.text}")
    return "\n\n---\n\n".join(parts)


def _history_messages(history: list[ConversationTurn], *, limit: int = 8) -> list[dict[str, str]]:
    """Convert recent history into provider-compatible chat messages."""

    messages: list[dict[str, str]] = []
    for turn in history[-limit:]:
        role = "assistant" if turn.role == "assistant" else "user"
        messages.append({"role": role, "content": turn.text})
    return messages


def build_answer_messages(
    question: str,
    chunks: list[RetrievedChunk],
    history: list[ConversationTurn],
) -> list[dict[str, str]]:
    """Build messages for municipal answer generation."""

    if not chunks:
        # CAMBIO CONVERSACION UNICA 2 - No contaminar una respuesta sin evidencia con turnos previos.
        # Motivo: el LLM debe explicar la ausencia de respaldo para la pregunta actual.
        # Riesgo mitigado: la conversacion general conserva historial en su flujo propio.
        return [
            {
                "role": "user",
                "content": (
                    f"Pregunta actual: {question}\n\n"
                    f"{NO_INFO_PROMPT.format(tema=question)}\n"
                    "Responde solamente sobre esta pregunta actual en maximo 60 palabras."
                ),
            }
        ]

    messages = _history_messages(history)
    context_block = build_context_block(chunks[:4])
    conversational_guidance = build_conversational_guidance(question, chunks)
    opening_guidance = (
        "ESTILO DE APERTURA: Esta es la primera respuesta documental de la conversacion. "
        "Inicia con un saludo breve y cercano, por ejemplo \"Hola, claro 😊\", y luego orienta. "
        "Usa como maximo 1 emoji.\n\n"
        if not history
        else (
            "ESTILO DE APERTURA: Ya existe historial en la conversacion. No saludes de nuevo, "
            "pero manten un tono cercano: puedes iniciar con \"Claro 😊\", \"Te explico\" o "
            "\"Vamos por partes\" si ayuda. Usa como maximo 1 emoji y evita sonar seco.\n\n"
        )
    )
    guidance_block = (
        f"\n\nGUIA DE CONTINUIDAD CONVERSACIONAL:\n{conversational_guidance}\n"
        if conversational_guidance
        else ""
    )
    messages.append(
        {
            "role": "user",
            "content": (
                "CONTEXTO RECUPERADO (fichas, FAQ, chunks o norma consolidada):\n\n"
                f"{context_block}\n\n"
                "---\n\n"
                f"{EVIDENCE_CHECK_PROMPT}\n\n"
                f"{opening_guidance}"
                f"{guidance_block}"
                f"PREGUNTA DEL CIUDADANO: {question}\n\nRESPUESTA:"
            ),
        }
    )
    return messages


def build_conversational_guidance(question: str, chunks: list[RetrievedChunk]) -> str:
    """Suggest a useful next question without replacing RAG answer generation."""

    normalized = _plain_text(
        " ".join(
            [
                question,
                *[
                    " ".join(
                        [
                            chunk.tipo_contenido,
                            chunk.section_title,
                            chunk.article_label,
                            chunk.text[:500],
                            " ".join(chunk.user_intents),
                            " ".join(
                                f"{key}:{value}"
                                for key, value in chunk.metadata.items()
                                if isinstance(value, (str, int, float, bool))
                            ),
                        ]
                    )
                    for chunk in chunks[:4]
                ],
            ]
        )
    )

    guidance = [
        "Responde primero la pregunta con la evidencia recuperada. Si falta contexto, "
        "cierra con una sola pregunta util para orientar el siguiente paso.",
    ]

    if any(term in normalized for term in ("diferencia", "comparar", "comparacion", " vs ", "versus")):
        guidance.append(
            "Si la pregunta compara dos temas y el contexto solo sustenta uno, no inventes "
            "el segundo. Explica primero lo que si esta documentado, luego aclara de forma "
            "amable que para comparar mejor falta cargar documentos del otro tema. Cierra "
            "ofreciendo revisar el tema disponible o agregar la nueva materia documental."
        )
    elif any(
        term in normalized
        for term in (
            "2do",
            "2 do",
            "segundo",
            "3er",
            "3 er",
            "tercer",
            "tercero",
            "cuarto",
            "quinto",
            "esa parte",
            "ese punto",
            "eso de",
            "lo de",
            "seguimiento enfocado",
        )
    ):
        guidance.append(
            "Si el ciudadano pregunta por una parte especifica de la respuesta anterior "
            "(por ejemplo segundo punto, tercer requisito, esa parte, lo de un pago, una "
            "zona, un giro o una condicion), explica solo esa parte en palabras sencillas. "
            "No repitas toda la lista ni reinicies el tramite."
        )
    elif any(term in normalized for term in ("que es", "que significa", "definicion", "a que se refiere")):
        guidance.append(
            "Si el ciudadano pidio una definicion, da solo el concepto en lenguaje sencillo "
            "y ofrece opciones breves para continuar, sin pedir datos personales ni explicar "
            "todo el tramite todavia."
        )
    elif any(term in normalized for term in ("giro", "giros", "rubro", "rubros", "actividad permitida")):
        guidance.append(
            "Si el caso trata de giro o rubro, pregunta que producto o servicio vende "
            "exactamente, o si su autorizacion indica algun giro especifico."
        )
        guidance.append(
            "Si la evidencia recuperada incluye rubros, codigos de giro o ejemplos "
            "orientativos, usalos para explicar con casos concretos. No digas que no "
            "hay lista exacta si el contexto ya muestra el articulo 21 o el listado de "
            "rubros/giros. Si el usuario menciona un producto, indica a que giro podria "
            "asemejarse y aclara que la municipalidad confirma el giro autorizado."
        )
    elif any(
        term in normalized
        for term in (
            "ubicacion",
            "zona",
            "zonas",
            "calle",
            "avenida",
            "jr ",
            "jiron",
            "manchay",
            "miguel grau",
            "zona rigida",
            "zona prohibida",
        )
    ):
        guidance.append(
            "Si el caso depende de ubicacion, pregunta la avenida, calle o referencia "
            "exacta y si el punto esta en vereda, esquina, cruce peatonal, parque, "
            "berma, paradero o cerca de una zona rigida."
        )
    elif any(term in normalized for term in ("renovar", "renovacion", "venc", "voucher")):
        guidance.append(
            "Si habla de renovacion o autorizacion vencida, pregunta si la autorizacion "
            "ya vencio o sigue vigente, y hasta que fecha fue emitida si lo recuerda."
        )
    elif any(
        term in normalized
        for term in (
            "permiso",
            "autorizacion",
            "requisito",
            "documento",
            "padron",
            "vender en la via publica",
            "vender en la calle",
            "comercio ambulatorio",
        )
    ):
        if "tipo_tramite:nuevo_ingreso_padron" in normalized or "tipo_tramite:renovacion" in normalized:
            guidance.append(
                "El contexto ya identifica si es tramite nuevo o renovacion. No preguntes "
                "otra vez si es primera vez o renovacion; responde el caso recuperado y, si "
                "hace falta continuar, pregunta solo por producto o ubicacion."
            )
        else:
            guidance.append(
                "Si trata de venta en via publica o tramite, pregunta solo si ayuda: si es "
                "solicitud nueva o renovacion, que producto desea vender y en que punto exacto."
            )

    return "\n".join(f"- {item}" for item in guidance)


def _plain_text(text: str) -> str:
    replacements = str.maketrans(
        {
            "á": "a",
            "é": "e",
            "í": "i",
            "ó": "o",
            "ú": "u",
            "ü": "u",
            "ñ": "n",
            "Á": "a",
            "É": "e",
            "Í": "i",
            "Ó": "o",
            "Ú": "u",
            "Ü": "u",
            "Ñ": "n",
        }
    )
    return " ".join(text.translate(replacements).lower().split())


def build_general_chat_messages(
    question: str,
    history: list[ConversationTurn],
) -> list[dict[str, str]]:
    """Build messages for unrestricted general chat."""

    messages = _history_messages(history)
    messages.append({"role": "user", "content": question})
    return messages


def build_query_rewrite_messages(
    question: str,
    history: list[ConversationTurn],
) -> list[dict[str, str]]:
    """Build messages for query rewriting."""

    messages = _history_messages(history, limit=6)
    messages.append(
        {
            "role": "user",
            "content": (
                "Reformula esta pregunta para buscar mejor en las ordenanzas, "
                "quitando ambiguedades y usando el contexto si hace falta:\n"
                f"{question}"
            ),
        }
    )
    return messages
