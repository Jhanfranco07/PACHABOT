from __future__ import annotations

import re
from pathlib import Path

from app.config import Settings
from app.models.schemas import ConversationTurn
from app.utils.helpers import ensure_directory, read_json, write_json


class ConversationMemoryStore:
    """Simple file-based memory store for per-session conversation history."""

    def __init__(self, settings: Settings, logger) -> None:
        self.settings = settings
        self.logger = logger.getChild("conversation_memory")
        ensure_directory(self.settings.conversations_dir)

    def load_history(self, channel: str, session_id: str) -> list[ConversationTurn]:
        """Load the full conversation history for a channel session."""

        path = self._build_session_path(channel, session_id)
        if not path.exists():
            path = self._build_legacy_session_path(channel, session_id)
        if not path.exists():
            return []

        try:
            payload = read_json(path)
            return [ConversationTurn(**item) for item in payload]
        except Exception as exc:
            self.logger.warning(
                "No se pudo leer la memoria de %s/%s: %s",
                channel,
                session_id,
                exc,
            )
            return []

    def get_recent_history(
        self,
        channel: str,
        session_id: str,
        *,
        limit: int | None = None,
    ) -> list[ConversationTurn]:
        """Return the last N turns for the session."""

        history = self.load_history(channel, session_id)
        if limit is None:
            limit = self.settings.memory_history_limit
        return history[-limit:]

    def append_turn(self, channel: str, session_id: str, turn: ConversationTurn) -> None:
        """Append a single turn and trim old memory."""

        history = self.load_history(channel, session_id)
        history.append(turn)
        max_turns = self.settings.memory_max_turns
        trimmed_history = history[-max_turns:]
        write_json(self._build_session_path(channel, session_id), trimmed_history)

    def reset_session(self, channel: str, session_id: str) -> None:
        """Delete a session memory file if it exists."""

        for path in (
            self._build_session_path(channel, session_id),
            self._build_legacy_session_path(channel, session_id),
        ):
            if path.exists():
                path.unlink()

    def _build_session_path(self, channel: str, session_id: str) -> Path:
        """Build the conversation file path for the given session."""

        safe_channel = _slugify(channel)
        safe_session = _slugify(session_id)
        return self.settings.conversations_dir / f"{safe_channel}_{safe_session}.json"

    def _build_legacy_session_path(self, channel: str, session_id: str) -> Path:
        """Read legacy runtime data left under processed/ during migration."""

        safe_channel = _slugify(channel)
        safe_session = _slugify(session_id)
        return self.settings.legacy_conversations_dir / f"{safe_channel}_{safe_session}.json"


def _slugify(value: str) -> str:
    """Make a filesystem-safe slug from any identifier."""

    return re.sub(r"[^a-zA-Z0-9_-]+", "_", value).strip("_") or "session"
