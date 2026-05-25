from __future__ import annotations

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """HTTP payload for channel-agnostic chat requests."""

    channel: str = "api"
    session_id: str = "api-session"
    user_id: str = "api-user"
    text: str = Field(min_length=1)
    mode: str | None = None
    user_display_name: str = ""
    metadata: dict[str, str] = Field(default_factory=dict)


class ChatResponse(BaseModel):
    """HTTP response returned by the assistant API."""

    answer: str
    sources: list[str]
    intent: str
    in_domain: bool
    confidence: float
    used_llm: bool
    mode: str


class ConversationResetRequest(BaseModel):
    """Payload used to clear a stored conversation session."""

    channel: str
    session_id: str


class ConversationResetResponse(BaseModel):
    """Response returned after a conversation reset."""

    status: str
    channel: str
    session_id: str
