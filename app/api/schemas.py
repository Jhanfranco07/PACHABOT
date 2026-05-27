from __future__ import annotations

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """HTTP payload for channel-agnostic chat requests."""

    channel: str = "api"
    session_id: str = "api-session"
    user_id: str = "api-user"
    text: str = Field(min_length=1)
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
    response_origin: str
    model_used: str = ""
    evidence: list[dict[str, object]] = Field(default_factory=list)
    evidence_warning: str = ""
    confidence_level: str = "none"


class ConversationResetRequest(BaseModel):
    """Payload used to clear a stored conversation session."""

    channel: str
    session_id: str


class ConversationResetResponse(BaseModel):
    """Response returned after a conversation reset."""

    status: str
    channel: str
    session_id: str


class OllamaRuntimeConfigRequest(BaseModel):
    """Temporary local-generation options used by the web simulator."""

    model: str = Field(min_length=1, max_length=120)
    think: bool = False
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    max_tokens: int = Field(default=400, ge=32, le=2048)


class OllamaRuntimeConfigResponse(BaseModel):
    """Available Ollama models and active simulator configuration."""

    provider: str
    available: bool
    models: list[str]
    active_model: str
    think: bool
    temperature: float
    max_tokens: int
    message: str = ""
