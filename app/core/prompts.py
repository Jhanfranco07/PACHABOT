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

    messages = _history_messages(history)

    if not chunks:
        messages.append(
            {
                "role": "user",
                "content": f"Pregunta: {question}\n\n{NO_INFO_PROMPT.format(tema=question)}",
            }
        )
        return messages

    context_block = build_context_block(chunks[:4])
    messages.append(
        {
            "role": "user",
            "content": (
                "Informacion de las ordenanzas:\n\n"
                f"{context_block}\n\n"
                "---\n\n"
                f"{EVIDENCE_CHECK_PROMPT}\n\n"
                f"Pregunta: {question}"
            ),
        }
    )
    return messages


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
