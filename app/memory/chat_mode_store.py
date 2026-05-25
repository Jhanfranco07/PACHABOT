from __future__ import annotations

import re
from pathlib import Path

from app.config import Settings
from app.models.domain import AssistantMode
from app.utils.helpers import ensure_directory, read_json, write_json


class ChatModeStore:
    """Persist the selected assistant mode per chat session."""

    def __init__(self, settings: Settings, logger) -> None:
        self.settings = settings
        self.logger = logger.getChild("chat_mode_store")
        ensure_directory(self.settings.chat_modes_dir)

    def get_mode(self, channel: str, session_id: str) -> AssistantMode:
        """Return the stored mode for a chat session, defaulting to general mode."""

        path = self._build_session_path(channel, session_id)
        if not path.exists():
            return self._default_mode()

        try:
            payload = read_json(path)
            raw_mode = str(payload.get("mode", AssistantMode.GENERAL.value)).strip().lower()
            return AssistantMode(raw_mode)
        except Exception as exc:
            self.logger.warning(
                "No se pudo leer el modo guardado de %s/%s: %s",
                channel,
                session_id,
                exc,
            )
            return self._default_mode()

    def set_mode(self, channel: str, session_id: str, mode: AssistantMode) -> None:
        """Persist the selected mode for a chat session."""

        write_json(
            self._build_session_path(channel, session_id),
            {
                "mode": mode.value,
                "channel": channel,
                "session_id": session_id,
            },
        )

    def reset_mode(self, channel: str, session_id: str) -> None:
        """Delete the stored mode for a session if it exists."""

        path = self._build_session_path(channel, session_id)
        if path.exists():
            path.unlink()

    def _build_session_path(self, channel: str, session_id: str) -> Path:
        """Build the mode file path for the given session."""

        safe_channel = _slugify(channel)
        safe_session = _slugify(session_id)
        return self.settings.chat_modes_dir / f"{safe_channel}_{safe_session}.json"

    def _default_mode(self) -> AssistantMode:
        """Resolve the configured default mode safely."""

        raw_mode = str(self.settings.default_assistant_mode).strip().lower()
        try:
            return AssistantMode(raw_mode)
        except ValueError:
            return AssistantMode.GENERAL


def _slugify(value: str) -> str:
    """Make a filesystem-safe slug from any identifier."""

    return re.sub(r"[^a-zA-Z0-9_-]+", "_", value).strip("_") or "session"
