from __future__ import annotations

from telegram import Update
from telegram.constants import ChatAction
from telegram.error import TimedOut
from telegram.ext import ContextTypes

from app.channels.telegram import build_incoming_message


MAX_TELEGRAM_MESSAGE_LEN = 3200

async def start_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /start."""

    if update.message is None or update.effective_chat is None:
        return
    await update.message.reply_text(
        "Hola, soy PachaBot. Puedes escribirme de forma natural. "
        "Si preguntas por comercio ambulatorio, usare la base documental municipal "
        "para orientarte con palabras sencillas."
    )


async def help_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /help."""

    if update.message is None:
        return
    await update.message.reply_text(
        "Hola. Conversa conmigo normalmente: puedo saludar y orientar consultas "
        "sobre comercio ambulatorio usando los documentos municipales cargados.\n\n"
        "Comandos utiles:\n"
        " /reset - Borrar el contexto de este chat\n"
        " /estado - Ver el estado actual del asistente"
    )


async def reset_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /reset and clear the current chat memory."""

    if update.effective_chat is None or update.message is None:
        return

    assistant = context.application.bot_data["assistant_service"]
    assistant.reset_conversation("telegram", str(update.effective_chat.id))
    await update.message.reply_text(
        "✅ Listo. He borrado el contexto de este chat. "
        "La siguiente consulta empezará como una conversación nueva."
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
    """Explain the retired mode selector for users of an older prototype."""

    if update.message is None:
        return

    await update.message.reply_text(
        "Ya no necesitas elegir un modo. Escribe cualquier mensaje con naturalidad; "
        "cuando consultes un tema municipal usare los documentos disponibles."
    )


async def general_mode_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Retained command alias after removing chat modes."""

    await mode_handler(update, context)


async def commerce_mode_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Retained command alias after removing chat modes."""

    await mode_handler(update, context)


async def chat_mode_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Shortcut command for general chat mode."""

    await mode_handler(update, context)


async def pachabot_mode_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Shortcut command for municipal commerce mode."""

    await mode_handler(update, context)


async def message_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Resolve user free-text messages."""

    if update.message is None or update.effective_chat is None:
        return

    incoming = build_incoming_message(update)
    if incoming is None:
        return
    # Evitar procesar nuevas consultas si ya hay una respuesta en curso
    chat_data = context.chat_data
    if chat_data.get("busy"):
        await update.message.reply_text(
            "⏳ Estoy respondiendo tu pregunta anterior. Por favor espera un momento antes de enviar otra consulta."
        )
        return

    chat_data["busy"] = True
    try:
        # Indicador visual para el usuario
        await context.bot.send_chat_action(
            chat_id=update.effective_chat.id,
            action=ChatAction.TYPING,
        )

        assistant = context.application.bot_data["assistant_service"]
        payload = assistant.answer_chat_message(incoming)

        # Hacer la respuesta más amigable añadiendo un emoji inicial
        response = "🙂 " + payload.answer.strip()
        if payload.sources:
            response += "\n\nFuente(s): " + "; ".join(payload.sources[:3])

        try:
            await _send_text_safely(update, response)
        except Exception as exc:  # pragma: no cover - runtime send error
            # Log and notify user so the bot does not silently drop the reply
            try:
                context.application.logger.warning("Error sending message: %s", exc)
            except Exception:
                pass
            try:
                await update.message.reply_text(
                    "Lo siento, tuve un problema enviando la respuesta. Intenta escribir de nuevo."
                )
            except Exception:
                # último recurso: nada podemos hacer si incluso esto falla
                pass
    finally:
        # Asegurar que siempre se limpie la bandera aunque falle la generación
        chat_data["busy"] = False


async def _send_text_safely(update: Update, text: str) -> None:
    """Send long responses in smaller chunks and retry once on timeout."""

    if update.message is None:
        return

    for part in _split_message(text):
        try:
            await update.message.reply_text(part)
        except TimedOut:
            await update.message.reply_text(part)


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
