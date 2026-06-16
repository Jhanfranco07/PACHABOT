from __future__ import annotations

import sys

from app.channels.schemas import IncomingChatMessage
from app.main import container


CHANNEL = "console"
DEFAULT_SESSION_ID = "console-local"


def _configure_console_encoding() -> None:
    """Allow Windows terminals to display natural-language and symbol output safely."""

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def _parse_args():
    """Read lightweight options for local interactive testing."""

    import argparse

    parser = argparse.ArgumentParser(
        description="Prueba PACHABOT en consola usando el mismo RAG y LLM que la API web."
    )
    parser.add_argument(
        "--session-id",
        default=DEFAULT_SESSION_ID,
        help="Identificador de sesion para conservar memoria entre preguntas.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Muestra intent, confianza, uso de LLM y fuentes recuperadas.",
    )
    return parser.parse_args()


def _print_help() -> None:
    """Show available interactive commands."""

    print(
        "\nComandos:\n"
        "  /reset     Limpiar la memoria de esta sesion\n"
        "  /estado    Mostrar proveedor, modelo e indice\n"
        "  /ayuda     Mostrar estos comandos\n"
        "  /salir     Cerrar la consola\n"
    )


def main() -> None:
    """Run an interactive terminal channel backed by the application services."""

    _configure_console_encoding()
    args = _parse_args()
    assistant = container.assistant_service
    session_id = args.session_id

    print("PACHABOT - consola local")
    provider = container.settings.llm_provider.lower().strip()
    if provider == "ollama":
        print(f"Modelo Ollama: {container.settings.ollama_model}")
        if not container.settings.ollama_enabled:
            print("Ollama desactivado: configura OLLAMA_ENABLED=true para usar IA local.")
    elif provider == "openai":
        print(f"Modelo OpenAI: {container.settings.openai_model}")
    print("Escribe /ayuda para ver comandos o /salir para terminar.")

    while True:
        try:
            question = input("\nTu: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nSesion finalizada.")
            return

        if not question:
            continue

        command = question.lower()
        if command in {"/salir", "/exit", "/quit"}:
            print("Sesion finalizada.")
            return
        if command in {"/ayuda", "/help"}:
            _print_help()
            continue
        if command == "/reset":
            assistant.reset_conversation(CHANNEL, session_id)
            print("PachaBot: Memoria de la conversacion eliminada.")
            continue
        if command == "/estado":
            status = assistant.get_runtime_status(channel=CHANNEL, session_id=session_id)
            print("\n" + status)
            continue
        response = assistant.answer_chat_message(
            IncomingChatMessage(
                channel=CHANNEL,
                session_id=session_id,
                user_id="console-user",
                text=question,
            )
        )
        print("\nPachaBot: " + response.answer)
        if args.debug:
            print(
                "\n[debug] "
                f"intent={response.intent.value} "
                f"confidence={response.confidence:.3f} "
                f"used_llm={response.used_llm} "
                f"origin={response.response_origin} "
                f"sources={response.sources}"
            )


if __name__ == "__main__":
    main()
