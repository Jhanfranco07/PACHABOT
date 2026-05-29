from app.config import Settings
from app.core.logger import setup_logging
from app.models.schemas import ConversationTurn
from app.services.query_rewriter import QueryRewriter


class DummyLLMService:
    def __init__(self) -> None:
        self.client = object()
        self.received_history = None

    def rewrite_query(self, question: str, *, history=None) -> str:
        self.received_history = history
        return "consulta reformulada"


class FailingDummyLLMService:
    def __init__(self) -> None:
        self.client = object()

    def rewrite_query(self, question: str, *, history=None) -> str:
        raise AssertionError("No deberia invocarse el rewrite externo para OpenRouter free")


def test_query_rewriter_passes_history_as_keyword_argument(tmp_path) -> None:
    settings = Settings(
        llm_provider="openai",
        llm_mode="auto",
        chat_model="gpt-test",
        raw_data_dir=tmp_path / "raw",
        processed_data_dir=tmp_path / "processed",
        vectorstore_dir=tmp_path / "vectorstore",
        conversations_dir=tmp_path / "processed" / "conversations",
        chat_modes_dir=tmp_path / "processed" / "chat_modes",
        processed_chunks_file=tmp_path / "processed" / "chunks.json",
        vectorizer_file=tmp_path / "vectorstore" / "tfidf_vectorizer.joblib",
        matrix_file=tmp_path / "vectorstore" / "tfidf_matrix.joblib",
    )
    llm_service = DummyLLMService()
    rewriter = QueryRewriter(settings, llm_service, setup_logging("INFO"))
    history = [ConversationTurn(role="user", text="Que es el SISA")]

    rewritten = rewriter.rewrite("y cuanto cuesta", history)

    assert rewritten == "consulta reformulada"
    assert llm_service.received_history == history


def test_query_rewriter_skips_external_rewrite_for_openrouter_free(tmp_path) -> None:
    settings = Settings(
        llm_provider="openrouter",
        llm_mode="auto",
        chat_model="mistralai/mistral-small-3.1-24b-instruct:free",
        chat_model_fallbacks=("google/gemma-3-12b-it:free",),
        raw_data_dir=tmp_path / "raw",
        processed_data_dir=tmp_path / "processed",
        vectorstore_dir=tmp_path / "vectorstore",
        conversations_dir=tmp_path / "processed" / "conversations",
        chat_modes_dir=tmp_path / "processed" / "chat_modes",
        processed_chunks_file=tmp_path / "processed" / "chunks.json",
        vectorizer_file=tmp_path / "vectorstore" / "tfidf_vectorizer.joblib",
        matrix_file=tmp_path / "vectorstore" / "tfidf_matrix.joblib",
    )
    llm_service = FailingDummyLLMService()
    rewriter = QueryRewriter(settings, llm_service, setup_logging("INFO"))
    history = [ConversationTurn(role="user", text="Que es el SISA")]

    rewritten = rewriter.rewrite("y cuanto cuesta", history)

    assert rewritten == "Que es el SISA. Seguimiento: y cuanto cuesta"


def test_query_rewriter_skips_ollama_for_standalone_question(tmp_path) -> None:
    settings = Settings(
        llm_provider="ollama",
        llm_mode="auto",
        ollama_model="qwen3.5:4b",
        raw_data_dir=tmp_path / "raw",
        processed_data_dir=tmp_path / "processed",
        vectorstore_dir=tmp_path / "vectorstore",
        conversations_dir=tmp_path / "processed" / "conversations",
        chat_modes_dir=tmp_path / "processed" / "chat_modes",
        processed_chunks_file=tmp_path / "processed" / "chunks.json",
        vectorizer_file=tmp_path / "vectorstore" / "tfidf_vectorizer.joblib",
        matrix_file=tmp_path / "vectorstore" / "tfidf_matrix.joblib",
    )
    llm_service = DummyLLMService()
    rewriter = QueryRewriter(settings, llm_service, setup_logging("INFO"))

    rewritten = rewriter.rewrite("Que requisitos necesito para vender", [])

    assert rewritten == "Que requisitos necesito para vender"
    assert llm_service.received_history is None


def test_query_rewriter_expands_location_follow_up_without_external_llm(tmp_path) -> None:
    settings = Settings(
        llm_provider="mock",
        llm_mode="mock",
        raw_data_dir=tmp_path / "raw",
        processed_data_dir=tmp_path / "processed",
        vectorstore_dir=tmp_path / "vectorstore",
        conversations_dir=tmp_path / "processed" / "conversations",
        chat_modes_dir=tmp_path / "processed" / "chat_modes",
        processed_chunks_file=tmp_path / "processed" / "chunks.json",
        vectorizer_file=tmp_path / "vectorstore" / "tfidf_vectorizer.joblib",
        matrix_file=tmp_path / "vectorstore" / "tfidf_matrix.joblib",
    )
    llm_service = DummyLLMService()
    llm_service.client = None
    rewriter = QueryRewriter(settings, llm_service, setup_logging("INFO"))
    history = [ConversationTurn(role="user", text="Puedo vender en zona rigida")]

    rewritten = rewriter.rewrite("y si es en Manchay", history)

    assert "Puedo vender en zona rigida" in rewritten
    assert "Manchay" in rewritten


def test_query_rewriter_reconstructs_non_compliance_follow_up(tmp_path) -> None:
    settings = Settings(
        llm_provider="mock",
        llm_mode="mock",
        raw_data_dir=tmp_path / "raw",
        processed_data_dir=tmp_path / "processed",
        vectorstore_dir=tmp_path / "vectorstore",
        conversations_dir=tmp_path / "processed" / "conversations",
        chat_modes_dir=tmp_path / "processed" / "chat_modes",
        processed_chunks_file=tmp_path / "processed" / "chunks.json",
        vectorizer_file=tmp_path / "vectorstore" / "tfidf_vectorizer.joblib",
        matrix_file=tmp_path / "vectorstore" / "tfidf_matrix.joblib",
    )
    llm_service = DummyLLMService()
    llm_service.client = None
    rewriter = QueryRewriter(settings, llm_service, setup_logging("INFO"))
    history = [ConversationTurn(role="user", text="Que obligaciones tiene un comerciante ambulante")]

    rewritten = rewriter.rewrite("que pasa si no cumple", history)

    assert "obligaciones" in rewritten.lower()
    assert "no cumple" in rewritten.lower()
