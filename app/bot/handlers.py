from __future__ import annotations

from telegram import Update
from telegram.error import TimedOut
from telegram.ext import ContextTypes

from app.bot.keyboards import (
    build_mode_keyboard,
    build_mode_picker_message,
    build_mode_selected_message,
    is_mode_selection_message,
    resolve_mode_from_label,
)
from app.channels.telegram import build_incoming_message
from app.models.domain import AssistantMode


MAX_TELEGRAM_MESSAGE_LEN = 3200

async def start_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /start."""

    if update.message is None or update.effective_chat is None:
        return
    assistant = context.application.bot_data["assistant_service"]
    session_id = str(update.effective_chat.id)
    active_mode = assistant.get_chat_mode("telegram", session_id)
    await update.message.reply_text(
        build_mode_picker_message() + f"\n\nModo actual: {assistant.describe_mode(active_mode)}.",
        reply_markup=build_mode_keyboard(active_mode),
    )


async def help_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /help."""

    if update.message is None:
        return
    await update.message.reply_text(
        "Ahora este bot tiene dos modos:\n"
        "- Modo General: preguntas libres\n"
        "- Modo Comercio: consultas sobre ordenanzas de comercio ambulatorio\n\n"
        "Comandos utiles:\n"
        "/modo para elegir o ver el modo actual\n"
        "/modo_general para usar el modo General\n"
        "/modo_comercio para usar el modo Comercio\n"
        "/chat para cambiar directo al modo General\n"
        "/pachabot para cambiar directo al modo Comercio\n"
        "/reset para borrar el contexto de este chat\n"
        "/estado para ver el modo actual del asistente"
    )


async def reset_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /reset and clear the current chat memory."""

    if update.effective_chat is None or update.message is None:
        return

    assistant = context.application.bot_data["assistant_service"]
    assistant.reset_conversation("telegram", str(update.effective_chat.id))
    await update.message.reply_text(
        "Listo. Borre el contexto conversacional de este chat. "
        "La siguiente consulta empezara como una conversacion nueva."
    )


async def status_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /estado and show a short runtime status."""

    if update.message is None:
        return

    assistant = context.application.bot_data["assistant_service"]
    await _send_text_safely(
        update,
        assistant.get_runtime_status(
            channel="telegram" if update.effective_chat else None,
            session_id=str(update.effective_chat.id) if update.effective_chat else None,
        ),
    )


async def mode_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show the current mode and how to switch it."""

    if update.message is None or update.effective_chat is None:
        return

    assistant = context.application.bot_data["assistant_service"]
    active_mode = assistant.get_chat_mode("telegram", str(update.effective_chat.id))
    await update.message.reply_text(
        build_mode_picker_message()
        + f"\n\nModo actual: {assistant.describe_mode(active_mode)}.",
        reply_markup=build_mode_keyboard(active_mode),
    )


async def general_mode_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Switch the current chat to general mode."""

    await _switch_mode(update, context, AssistantMode.GENERAL)


async def commerce_mode_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Switch the current chat to commerce mode."""

    await _switch_mode(update, context, AssistantMode.COMMERCE)


async def chat_mode_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Shortcut command for general chat mode."""

    await _switch_mode(update, context, AssistantMode.GENERAL)


async def pachabot_mode_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Shortcut command for municipal commerce mode."""

    await _switch_mode(update, context, AssistantMode.COMMERCE)


async def message_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Resolve user free-text messages."""

    if update.message is None or update.effective_chat is None:
        return

    if is_mode_selection_message(update.message.text or ""):
        selected_mode = resolve_mode_from_label(update.message.text or "")
        await _switch_mode(update, context, selected_mode)
        return

    incoming = build_incoming_message(update)
    if incoming is None:
        return

    assistant = context.application.bot_data["assistant_service"]
    payload = assistant.answer_chat_message(incoming)

    response = payload.answer.strip()
    if payload.sources:
        response += "\n\nFuente(s): " + "; ".join(payload.sources[:3])

    await _send_text_safely(update, response)


async def _send_text_safely(update: Update, text: str) -> None:
    """Send long responses in smaller chunks and retry once on timeout."""

    if update.message is None:
        return

    for part in _split_message(text):
        try:
            await update.message.reply_text(part)
        except TimedOut:
            await update.message.reply_text(part)


async def _switch_mode(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    mode: AssistantMode,
) -> None:
    """Persist a new chat mode and confirm it to the user."""

    if update.message is None or update.effective_chat is None:
        return

    assistant = context.application.bot_data["assistant_service"]
    assistant.set_chat_mode("telegram", str(update.effective_chat.id), mode)
    await update.message.reply_text(
        build_mode_selected_message(mode),
        reply_markup=build_mode_keyboard(mode),
    )


def _split_message(text: str, limit: int = MAX_TELEGRAM_MESSAGE_LEN) -> list[str]:
    """Split a long Telegram response by paragraph boundaries."""

    compact = text.strip()
    if len(compact) <= limit:
        return [compact]

    parts: list[str] = []
    current = ""
    for paragraph in compact.split("\n\n"):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        candidate = f"{current}\n\n{paragraph}".strip() if current else paragraph
        if len(candidate) <= limit:
            current = candidate
            continue

        if current:
            parts.append(current)

        while len(paragraph) > limit:
            parts.append(paragraph[:limit].rstrip())
            paragraph = paragraph[limit:].lstrip()
        current = paragraph

    if current:
        parts.append(current)
    return parts
