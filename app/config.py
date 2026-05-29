from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


def _env_as_bool(name: str, default: bool = False) -> bool:
    """Parse a boolean environment variable in a forgiving way."""

    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "on", "si"}


def _env_as_csv(name: str) -> tuple[str, ...]:
    """Parse a comma-separated environment variable into a tuple of values."""

    raw_value = os.getenv(name, "")
    if not raw_value.strip():
        return ()
    return tuple(
        item.strip()
        for item in raw_value.split(",")
        if item.strip()
    )


@dataclass(slots=True)
class Settings:
    """Centralized application settings."""

    app_name: str = "Asistente de Comercio Ambulatorio"
    app_env: str = os.getenv("APP_ENV", "development")
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    telegram_bot_token: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
    llm_provider: str = os.getenv("LLM_PROVIDER", "openai")
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-5.4-mini")
    openai_max_output_tokens: int = int(os.getenv("OPENAI_MAX_OUTPUT_TOKENS", "500"))
    openai_temperature: float = float(os.getenv("OPENAI_TEMPERATURE", "0.2"))
    openrouter_api_key: str = os.getenv("OPENROUTER_API_KEY", "")
    grok_api_key: str = os.getenv("GROK_API_KEY", "")
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    openai_base_url: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    openrouter_base_url: str = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    openrouter_http_referer: str = os.getenv("OPENROUTER_HTTP_REFERER", "http://localhost")
    openrouter_app_name: str = os.getenv("OPENROUTER_APP_NAME", "PachaBot")
    grok_base_url: str = os.getenv("GROK_BASE_URL", "https://api.x.ai/v1")
    groq_base_url: str = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
    # CAMBIO FASE OLLAMA 1 — Configurar el proveedor local sin credenciales.
    # Motivo: permitir ejecutar el asistente conversacional con el modelo instalado en Ollama.
    # Riesgo mitigado: se agregan opciones independientes sin cambiar proveedores remotos existentes.
    ollama_enabled: bool = _env_as_bool("OLLAMA_ENABLED", False)
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "llama3.1")
    ollama_timeout: float = float(os.getenv("OLLAMA_TIMEOUT", "120"))
    ollama_think: bool = _env_as_bool("OLLAMA_THINK", False)
    ollama_temperature: float = float(os.getenv("OLLAMA_TEMPERATURE", "0.2"))
    ollama_max_tokens: int = int(os.getenv("OLLAMA_MAX_TOKENS", "400"))
    ollama_keep_alive: str = os.getenv("OLLAMA_KEEP_ALIVE", "5m")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "local-tfidf")
    chat_model: str = os.getenv("CHAT_MODEL", os.getenv("OPENAI_MODEL", "gpt-5.4-mini"))
    chat_model_fallbacks: tuple[str, ...] = _env_as_csv("CHAT_MODEL_FALLBACKS")
    model_retry_cooldown_seconds: int = int(os.getenv("MODEL_RETRY_COOLDOWN_SECONDS", "180"))
    llm_mode: str = os.getenv("LLM_MODE", "auto")
    default_assistant_mode: str = os.getenv("DEFAULT_ASSISTANT_MODE", "general")
    allow_general_chat: bool = _env_as_bool("ALLOW_GENERAL_CHAT", False)
    retrieval_top_k: int = int(os.getenv("RETRIEVAL_TOP_K", "4"))
    retrieval_min_score: float = float(os.getenv("RETRIEVAL_MIN_SCORE", "0.35"))
    retrieval_max_results: int = int(os.getenv("RETRIEVAL_MAX_RESULTS", "5"))
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "700"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "120"))
    confidence_threshold: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.05"))
    memory_history_limit: int = int(os.getenv("MEMORY_HISTORY_LIMIT", "12"))
    memory_max_turns: int = int(os.getenv("MEMORY_MAX_TURNS", "40"))
    assistant_max_sources: int = int(os.getenv("ASSISTANT_MAX_SOURCES", "3"))
    rag_debug_trace: bool = _env_as_bool("RAG_DEBUG_TRACE", False)
    raw_data_dir: Path = BASE_DIR / "data" / "raw"
    # CAMBIO FASE 7.1 — Incorporar salidas intermedias del pipeline documental.
    # Motivo: separar texto limpio y norma consolidada de las fuentes originales.
    # Riesgo mitigado: se agregan rutas nuevas sin alterar data/raw ni los puntos de entrada.
    cleaned_data_dir: Path = BASE_DIR / "data" / "cleaned"
    consolidated_data_dir: Path = BASE_DIR / "data" / "consolidated"
    processed_data_dir: Path = BASE_DIR / "data" / "processed"
    tramites_data_dir: Path = BASE_DIR / "data" / "tramites"
    faq_data_dir: Path = BASE_DIR / "data" / "faq"
    vectorstore_dir: Path = BASE_DIR / "data" / "vectorstore"
    runtime_data_dir: Path = BASE_DIR / "data" / "runtime"
    runtime_debug_dir: Path = BASE_DIR / "data" / "runtime" / "debug"
    conversations_dir: Path = BASE_DIR / "data" / "runtime" / "conversations"
    chat_modes_dir: Path = BASE_DIR / "data" / "runtime" / "chat_modes"
    legacy_conversations_dir: Path = BASE_DIR / "data" / "processed" / "conversations"
    legacy_chat_modes_dir: Path = BASE_DIR / "data" / "processed" / "chat_modes"
    processed_chunks_file: Path = BASE_DIR / "data" / "processed" / "chunks.json"
    consolidated_norm_file: Path = BASE_DIR / "data" / "consolidated" / "norma_consolidada.json"
    modification_map_file: Path = BASE_DIR / "data" / "consolidated" / "modification_map.json"
    corpus_validation_report_file: Path = BASE_DIR / "data" / "consolidated" / "corpus_validation_report.json"
    vectorizer_file: Path = BASE_DIR / "data" / "vectorstore" / "tfidf_vectorizer.joblib"
    matrix_file: Path = BASE_DIR / "data" / "vectorstore" / "tfidf_matrix.joblib"


def get_settings() -> Settings:
    """Build and return settings."""

    return Settings()
