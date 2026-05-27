from __future__ import annotations

from telegram import ReplyKeyboardMarkup

from app.models.domain import AssistantMode


MODE_GENERAL_LABEL = "Usar modo General"
MODE_COMMERCE_LABEL = "Usar modo Comercio"


def build_mode_keyboard(active_mode: AssistantMode) -> ReplyKeyboardMarkup:
    """Build a minimal keyboard that only exposes mode switching."""

    _ = active_mode
    return ReplyKeyboardMarkup(
        [[MODE_GENERAL_LABEL, MODE_COMMERCE_LABEL]],
        resize_keyboard=True,
        one_time_keyboard=False,
    )


def is_mode_selection_message(text: str) -> bool:
    """Return whether the received text matches one of the mode buttons."""

    compact = text.strip()
    return compact in {MODE_GENERAL_LABEL, MODE_COMMERCE_LABEL}


def resolve_mode_from_label(text: str) -> AssistantMode:
    """Map a Telegram button label to an assistant mode."""

    if text.strip() == MODE_COMMERCE_LABEL:
        return AssistantMode.COMMERCE
    return AssistantMode.GENERAL


def build_mode_selected_message(mode: AssistantMode) -> str:
    """Return a short user-facing confirmation after changing mode."""

    if mode == AssistantMode.COMMERCE:
        return (
            "🎉 PachaBot activado.\n\n"
            "Ahora estás en modo Comercio ambulatorio. En este modo revisaré las ordenanzas, "
            "la memoria del chat y el contexto municipal para responderte.\n\n"
            "Escribe tu consulta con tus propias palabras (por ejemplo: 'Qué necesito para vender?')."
        )

    return (
        "💬 Chat General activado.\n\n"
        "Ahora estás en modo General.\n\n"
        "Pregunta lo que quieras de forma natural. Si luego quieres revisar ordenanzas municipales, cambia al modo Comercio."
    )


def build_mode_picker_message() -> str:
    """Return the Telegram intro message for choosing a mode."""

    return (
        "👋 Bienvenido! Elige el tipo de asistente que quieres usar en este chat:\n\n"
        "- 🗨️ Modo General: preguntas libres y conversación general.\n"
        "- 🧾 Modo Comercio: consultas sobre comercio ambulatorio y ordenanzas municipales.\n\n"
        "No necesitas usar preguntas predefinidas: escribe tu consulta con tus propias palabras.\n\n"
        "Atajos rápidos:\n"
        "/chat - modo General\n"
        "/pachabot - modo Comercio"
    )
