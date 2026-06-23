from __future__ import annotations

from dataclasses import dataclass
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, RedirectResponse

from app.api.schemas import (
    ChatRequest,
    ChatResponse,
    ConversationResetRequest,
    ConversationResetResponse,
    OllamaRuntimeConfigRequest,
    OllamaRuntimeConfigResponse,
)
from app.channels.schemas import IncomingChatMessage
from app.config import Settings, get_settings
from app.core.logger import setup_logging
from app.memory.chat_mode_store import ChatModeStore
from app.memory.conversation_store import ConversationMemoryStore
from app.services.assistant_service import AssistantService
from app.services.document_service import DocumentService
from app.services.evidence_service import EvidenceService
from app.services.intent_interpreter import IntentInterpreterService
from app.services.llm_service import LLMService, OLLAMA_DISABLED_MESSAGE
from app.services.query_router import QueryRouter
from app.services.query_rewriter import QueryRewriter
from app.services.retrieval_service import RetrievalService
from app.tools.document_toolkit import DocumentToolkit


@dataclass(slots=True)
class AppContainer:
    settings: Settings
    assistant_service: AssistantService
    llm_service: LLMService
    document_service: DocumentService
    retrieval_service: RetrievalService
    memory_store: ConversationMemoryStore
    mode_store: ChatModeStore
    document_toolkit: DocumentToolkit
    query_rewriter: QueryRewriter
    evidence_service: EvidenceService
    intent_interpreter: IntentInterpreterService


def build_container() -> AppContainer:
    """Compose the application services."""

    settings = get_settings()
    logger = setup_logging(settings.log_level)

    router = QueryRouter()
    llm_service = LLMService(settings, logger)
    retrieval_service = RetrievalService(settings, logger)
    retrieval_service.load_index()
    memory_store = ConversationMemoryStore(settings, logger)
    mode_store = ChatModeStore(settings, logger)
    query_rewriter = QueryRewriter(settings, llm_service, logger)
    document_toolkit = DocumentToolkit(settings, retrieval_service, query_rewriter, logger)
    document_service = DocumentService(settings, logger)
    evidence_service = EvidenceService(settings, logger)
    intent_interpreter = IntentInterpreterService(router, llm_service, logger)
    assistant_service = AssistantService(
        settings=settings,
        router=router,
        document_toolkit=document_toolkit,
        llm_service=llm_service,
        memory_store=memory_store,
        mode_store=mode_store,
        logger=logger,
        evidence_service=evidence_service,
        intent_interpreter=intent_interpreter,
    )
    return AppContainer(
        settings=settings,
        assistant_service=assistant_service,
        llm_service=llm_service,
        document_service=document_service,
        retrieval_service=retrieval_service,
        memory_store=memory_store,
        mode_store=mode_store,
        document_toolkit=document_toolkit,
        query_rewriter=query_rewriter,
        evidence_service=evidence_service,
        intent_interpreter=intent_interpreter,
    )


container = build_container()
app = FastAPI(title=container.settings.app_name, version="0.2.0")
WEB_INDEX = Path(__file__).resolve().parent / "web" / "index.html"


@app.get("/", include_in_schema=False)
def home() -> RedirectResponse:
    """Open the local chat simulator by default."""

    return RedirectResponse(url="/simulator")


@app.get("/simulator", include_in_schema=False)
def simulator() -> FileResponse:
    """Serve the local WhatsApp-style development simulator."""

    return FileResponse(WEB_INDEX)


@app.get("/health")
def healthcheck() -> dict[str, str]:
    """Basic health endpoint."""

    return {"status": "ok", "environment": container.settings.app_env}


@app.get("/info")
def info() -> dict[str, object]:
    """Describe the current prototype scope."""

    return {
        "name": container.settings.app_name,
        "scope": "Consultas ciudadanas sobre comercio ambulatorio",
        "documents": [
            "Ordenanza 108-2012-MDP/C",
            "Ordenanza 227-2019-MDP/C",
        ],
        "channels": ["api", "web-simulator"],
        "memory_enabled": True,
        "tools": ["query_rewrite", "document_search"],
    }


@app.get("/ollama/config", response_model=OllamaRuntimeConfigResponse)
def ollama_config() -> OllamaRuntimeConfigResponse:
    """Return active LLM runtime options for the web simulator.

    The route name is kept for backwards compatibility with the existing
    simulator, but it must not touch Ollama unless Ollama is the active provider.
    """

    settings = container.settings
    provider = settings.llm_provider.lower().strip() or "openai"
    if provider == "openai":
        return OllamaRuntimeConfigResponse(
            provider="openai",
            available=bool(settings.openai_api_key) and container.llm_service.client is not None,
            models=[settings.openai_model],
            active_model=settings.openai_model,
            think=False,
            temperature=settings.openai_temperature,
            max_tokens=settings.openai_max_output_tokens,
            message="Proveedor activo: OpenAI. Ollama no se consulta ni se inicia.",
        )
    if provider != "ollama":
        return OllamaRuntimeConfigResponse(
            provider=provider,
            available=container.llm_service.client is not None,
            models=[settings.chat_model],
            active_model=settings.chat_model,
            think=False,
            temperature=0.2,
            max_tokens=settings.openai_max_output_tokens,
            message="Ollama no se consulta porque el proveedor activo no es ollama.",
        )
    if not settings.ollama_enabled:
        return OllamaRuntimeConfigResponse(
            provider="ollama",
            available=False,
            models=[],
            active_model=settings.ollama_model,
            think=settings.ollama_think,
            temperature=settings.ollama_temperature,
            max_tokens=settings.ollama_max_tokens,
            message=OLLAMA_DISABLED_MESSAGE,
        )
    try:
        models = container.llm_service.list_ollama_models()
    except RuntimeError as exc:
        return OllamaRuntimeConfigResponse(
            provider="ollama",
            available=False,
            models=[],
            active_model=settings.ollama_model,
            think=settings.ollama_think,
            temperature=settings.ollama_temperature,
            max_tokens=settings.ollama_max_tokens,
            message=str(exc),
        )
    return OllamaRuntimeConfigResponse(
        provider="ollama",
        available=settings.ollama_model in models,
        models=models,
        active_model=settings.ollama_model,
        think=settings.ollama_think,
        temperature=settings.ollama_temperature,
        max_tokens=settings.ollama_max_tokens,
    )


@app.post("/ollama/config", response_model=OllamaRuntimeConfigResponse)
def update_ollama_config(payload: OllamaRuntimeConfigRequest) -> OllamaRuntimeConfigResponse:
    """Apply local simulator generation settings until the API process stops."""

    provider = container.settings.llm_provider.lower().strip() or "openai"
    if provider == "openai":
        container.settings.openai_model = payload.model
        container.settings.openai_temperature = payload.temperature
        container.settings.openai_max_output_tokens = payload.max_tokens
        return ollama_config()
    if provider != "ollama":
        return ollama_config()

    try:
        container.llm_service.configure_ollama_runtime(
            model=payload.model,
            think=payload.think,
            temperature=payload.temperature,
            max_tokens=payload.max_tokens,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return ollama_config()


@app.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest) -> ChatResponse:
    """Channel-agnostic endpoint to test the assistant outside Telegram."""

    result = container.assistant_service.answer_chat_message(
        IncomingChatMessage(
            channel=payload.channel,
            session_id=payload.session_id,
            user_id=payload.user_id,
            text=payload.text,
            user_display_name=payload.user_display_name,
            metadata=payload.metadata,
        )
    )
    provider = container.settings.llm_provider.lower().strip()
    model_used = ""
    if result.used_llm and provider == "ollama":
        model_used = container.settings.ollama_model
    elif result.used_llm and provider == "openai":
        model_used = container.settings.openai_model
    elif result.used_llm:
        model_used = container.settings.chat_model
    return ChatResponse(
        answer=result.answer,
        sources=result.sources,
        intent=result.intent.value,
        in_domain=result.in_domain,
        confidence=result.confidence,
        used_llm=result.used_llm,
        response_origin=result.response_origin,
        model_used=model_used,
        evidence=[asdict(item) for item in result.evidence],
        evidence_warning=result.evidence_warning,
        confidence_level=result.confidence_level,
    )


@app.post("/chat/reset", response_model=ConversationResetResponse)
def reset_chat(payload: ConversationResetRequest) -> ConversationResetResponse:
    """Reset a conversation memory session."""

    container.assistant_service.reset_conversation(payload.channel, payload.session_id)
    return ConversationResetResponse(
        status="ok",
        channel=payload.channel,
        session_id=payload.session_id,
    )
