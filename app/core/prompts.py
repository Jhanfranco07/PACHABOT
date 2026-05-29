from __future__ import annotations

from typing import Iterable

from app.models.schemas import ConversationTurn, RetrievedChunk
from app.prompts.templates import (
    ANTIHALLUCINATION_INSTRUCTION,
    CITATION_FORMAT,
    EVIDENCE_CHECK_PROMPT,
    GENERAL_CHAT_SYSTEM_PROMPT,
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

    if any(term in normalized for term in ("que es", "que significa", "definicion", "a que se refiere")):
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
