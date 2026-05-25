from __future__ import annotations

from dataclasses import dataclass

from fastapi import FastAPI

from app.api.schemas import (
    ChatRequest,
    ChatResponse,
    ConversationResetRequest,
    ConversationResetResponse,
)
from app.channels.schemas import IncomingChatMessage
from app.config import Settings, get_settings
from app.core.logger import setup_logging
from app.memory.chat_mode_store import ChatModeStore
from app.memory.conversation_store import ConversationMemoryStore
from app.models.domain import AssistantMode
from app.services.assistant_service import AssistantService
from app.services.document_service import DocumentService
from app.services.llm_service import LLMService
from app.services.query_router import QueryRouter
from app.services.query_rewriter import QueryRewriter
from app.services.retrieval_service import RetrievalService
from app.tools.document_toolkit import DocumentToolkit


@dataclass(slots=True)
class AppContainer:
    settings: Settings
    assistant_service: AssistantService
    document_service: DocumentService
    retrieval_service: RetrievalService
    memory_store: ConversationMemoryStore
    mode_store: ChatModeStore
    document_toolkit: DocumentToolkit
    query_rewriter: QueryRewriter


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
    assistant_service = AssistantService(
        settings=settings,
        router=router,
        document_toolkit=document_toolkit,
        llm_service=llm_service,
        memory_store=memory_store,
        mode_store=mode_store,
        logger=logger,
    )
    return AppContainer(
        settings=settings,
        assistant_service=assistant_service,
        document_service=document_service,
        retrieval_service=retrieval_service,
        memory_store=memory_store,
        mode_store=mode_store,
        document_toolkit=document_toolkit,
        query_rewriter=query_rewriter,
    )


container = build_container()
app = FastAPI(title=container.settings.app_name, version="0.2.0")


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
        "channels": ["telegram", "api"],
        "memory_enabled": True,
        "tools": ["query_rewrite", "document_search"],
    }


@app.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest) -> ChatResponse:
    """Channel-agnostic endpoint to test the assistant outside Telegram."""

    if payload.mode:
        container.assistant_service.set_chat_mode(
            payload.channel,
            payload.session_id,
            AssistantMode(payload.mode.lower().strip()),
        )
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
    return ChatResponse(
        answer=result.answer,
        sources=result.sources,
        intent=result.intent.value,
        in_domain=result.in_domain,
        confidence=result.confidence,
        used_llm=result.used_llm,
        mode=container.assistant_service.get_chat_mode(payload.channel, payload.session_id).value,
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
