import json
import time
from urllib import error as urllib_error

from app.config import Settings
from app.core.logger import setup_logging
from app.models.schemas import ConversationTurn, RetrievedChunk
from app.services import llm_service as llm_module
from app.services.llm_service import LLMService


class DummyOpenAI:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs


def test_llm_service_configures_openrouter_client(monkeypatch) -> None:
    monkeypatch.setattr(llm_module, "OpenAI", DummyOpenAI)
    settings = Settings(
        llm_provider="openrouter",
        llm_mode="auto",
        openrouter_api_key="openrouter-test-key",
        openrouter_base_url="https://openrouter.ai/api/v1",
        openrouter_http_referer="http://localhost:8000",
        openrouter_app_name="PachaBot",
    )

    service = LLMService(settings, setup_logging("INFO"))

    assert service.client is not None
    assert service.client.kwargs["api_key"] == "openrouter-test-key"
    assert service.client.kwargs["base_url"] == "https://openrouter.ai/api/v1"
    assert service.client.kwargs["default_headers"]["HTTP-Referer"] == "http://localhost:8000"
    assert service.client.kwargs["default_headers"]["X-Title"] == "PachaBot"


def test_llm_service_configures_groq_client(monkeypatch) -> None:
    monkeypatch.setattr(llm_module, "OpenAI", DummyOpenAI)
    settings = Settings(
        llm_provider="groq",
        llm_mode="auto",
        groq_api_key="groq-test-key",
        groq_base_url="https://api.groq.com/openai/v1",
    )

    service = LLMService(settings, setup_logging("INFO"))

    assert service.client is not None
    assert service.client.kwargs["api_key"] == "groq-test-key"
    assert service.client.kwargs["base_url"] == "https://api.groq.com/openai/v1"


class DummyResponseResult:
    output_text = ""


class DummyMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class DummyChoice:
    def __init__(self, content: str) -> None:
        self.message = DummyMessage(content)


class DummyCompletion:
    def __init__(self, content: str) -> None:
        self.choices = [DummyChoice(content)]


class FailingResponsesAPI:
    def create(self, **kwargs):
        raise Exception("Developer instruction is not enabled for models/gemma-3-12b-it")


class ChatCompletionsRetryAPI:
    def __init__(self) -> None:
        self.calls: list[list[dict[str, str]]] = []

    def create(self, **kwargs):
        messages = kwargs["messages"]
        self.calls.append(messages)
        if messages and messages[0]["role"] == "system":
            raise Exception("Developer instruction is not enabled for models/gemma-3-12b-it")
        return DummyCompletion("respuesta desde retry compatible")


class DummyClientWithRetry:
    def __init__(self) -> None:
        self.responses = FailingResponsesAPI()
        self.chat = type("ChatAPI", (), {"completions": ChatCompletionsRetryAPI()})()


def test_llm_service_retries_without_system_role_when_provider_rejects_it(monkeypatch) -> None:
    monkeypatch.setattr(llm_module, "OpenAI", DummyOpenAI)
    settings = Settings(
        llm_provider="openrouter",
        llm_mode="auto",
        openrouter_api_key="openrouter-test-key",
        chat_model="google/gemma-3-12b-it:free",
    )
    service = LLMService(settings, setup_logging("INFO"))
    service.client = DummyClientWithRetry()

    answer, used_llm = service.generate_general_answer(
        "Que hora es",
        history=[ConversationTurn(role="user", text="Hola")],
    )

    assert used_llm is True
    assert answer == "respuesta desde retry compatible"
    calls = service.client.chat.completions.calls
    assert len(calls) == 2
    assert calls[0][0]["role"] == "system"
    assert calls[1][0]["role"] == "user"
    assert "instrucciones del asistente" in calls[1][0]["content"].lower()


class AlwaysFailingResponsesAPI:
    def create(self, **kwargs):
        raise Exception("429 temporarily rate-limited upstream")


class ChatCompletionsFallbackAPI:
    def __init__(self) -> None:
        self.models: list[str] = []

    def create(self, **kwargs):
        model_name = kwargs["model"]
        self.models.append(model_name)
        if model_name == "google/gemma-3-12b-it:free":
            raise Exception("429 temporarily rate-limited upstream")
        return DummyCompletion(f"respuesta desde {model_name}")


class DummyClientWithModelFallback:
    def __init__(self) -> None:
        self.responses = AlwaysFailingResponsesAPI()
        self.chat = type("ChatAPI", (), {"completions": ChatCompletionsFallbackAPI()})()


def test_llm_service_tries_fallback_models_on_rate_limit(monkeypatch) -> None:
    monkeypatch.setattr(llm_module, "OpenAI", DummyOpenAI)
    settings = Settings(
        llm_provider="openrouter",
        llm_mode="auto",
        openrouter_api_key="openrouter-test-key",
        chat_model="google/gemma-3-12b-it:free",
        chat_model_fallbacks=(
            "meta-llama/llama-3.1-8b-instruct:free",
            "mistralai/mistral-small-3.1-24b-instruct:free",
        ),
    )
    service = LLMService(settings, setup_logging("INFO"))
    service.client = DummyClientWithModelFallback()

    answer, used_llm = service.generate_general_answer("Hola")

    assert used_llm is True
    assert answer == "respuesta desde meta-llama/llama-3.1-8b-instruct:free"
    assert service.client.chat.completions.models == [
        "google/gemma-3-12b-it:free",
        "meta-llama/llama-3.1-8b-instruct:free",
    ]


class ChatCompletionsNotFoundFallbackAPI:
    def __init__(self) -> None:
        self.models: list[str] = []

    def create(self, **kwargs):
        model_name = kwargs["model"]
        self.models.append(model_name)
        if model_name == "meta-llama/llama-3.1-8b-instruct:free":
            raise Exception("404 No endpoints found for meta-llama/llama-3.1-8b-instruct:free.")
        return DummyCompletion(f"respuesta desde {model_name}")


class DummyClientWithNotFoundFallback:
    def __init__(self) -> None:
        self.responses = AlwaysFailingResponsesAPI()
        self.chat = type("ChatAPI", (), {"completions": ChatCompletionsNotFoundFallbackAPI()})()


def test_llm_service_tries_fallback_models_on_missing_endpoint(monkeypatch) -> None:
    monkeypatch.setattr(llm_module, "OpenAI", DummyOpenAI)
    settings = Settings(
        llm_provider="openrouter",
        llm_mode="auto",
        openrouter_api_key="openrouter-test-key",
        chat_model="meta-llama/llama-3.1-8b-instruct:free",
        chat_model_fallbacks=(
            "mistralai/mistral-small-3.1-24b-instruct:free",
            "qwen/qwen3-4b:free",
        ),
    )
    service = LLMService(settings, setup_logging("INFO"))
    service.client = DummyClientWithNotFoundFallback()

    answer, used_llm = service.generate_general_answer("Dime un chiste")

    assert used_llm is True
    assert answer == "respuesta desde mistralai/mistral-small-3.1-24b-instruct:free"
    assert service.client.chat.completions.models == [
        "meta-llama/llama-3.1-8b-instruct:free",
        "mistralai/mistral-small-3.1-24b-instruct:free",
    ]


def test_llm_service_reports_free_models_saturated_when_all_are_on_cooldown(monkeypatch) -> None:
    monkeypatch.setattr(llm_module, "OpenAI", DummyOpenAI)
    settings = Settings(
        llm_provider="openrouter",
        llm_mode="auto",
        openrouter_api_key="openrouter-test-key",
        chat_model="mistralai/mistral-small-3.1-24b-instruct:free",
        chat_model_fallbacks=(
            "google/gemma-3-12b-it:free",
            "qwen/qwen3-4b:free",
        ),
    )
    service = LLMService(settings, setup_logging("INFO"))
    service.client = DummyClientWithRetry()
    cooldown_until = time.time() + 120
    service._model_cooldowns = {
        "mistralai/mistral-small-3.1-24b-instruct:free": cooldown_until,
        "google/gemma-3-12b-it:free": cooldown_until,
        "qwen/qwen3-4b:free": cooldown_until,
    }

    answer, used_llm = service.generate_general_answer("Dime un chiste")

    assert used_llm is False
    assert "modelos gratuitos estan temporalmente saturados" in answer.lower()


def test_llm_service_reports_free_models_saturated_for_municipal_mode(monkeypatch) -> None:
    monkeypatch.setattr(llm_module, "OpenAI", DummyOpenAI)
    settings = Settings(
        llm_provider="openrouter",
        llm_mode="auto",
        openrouter_api_key="openrouter-test-key",
        chat_model="mistralai/mistral-small-3.1-24b-instruct:free",
        chat_model_fallbacks=(),
    )
    service = LLMService(settings, setup_logging("INFO"))
    service.client = DummyClientWithRetry()
    cooldown_until = time.time() + 120
    service._model_cooldowns = {
        "mistralai/mistral-small-3.1-24b-instruct:free": cooldown_until,
    }

    answer, used_llm = service.generate_answer(
        "Que dice el articulo 7",
        [
            RetrievedChunk(
                chunk_id="vigente-7",
                document_id="ordenanza_227_2019",
                source_title="Ordenanza 227-2019-MDP/C",
                text="Articulo 7. La autorizacion es personal.",
                score=1.0,
                article_label="7",
                vigencia="vigente",
            )
        ],
        history=[],
    )

    assert used_llm is False
    assert "modelos gratuitos estan temporalmente saturados" in answer.lower()


class DummyOllamaResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *args) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


def _ollama_settings() -> Settings:
    return Settings(
        llm_provider="ollama",
        llm_mode="auto",
        ollama_base_url="http://localhost:11434",
        ollama_model="qwen3.5:4b",
        ollama_timeout=120,
        ollama_think=False,
        ollama_temperature=0.2,
        ollama_max_tokens=400,
        ollama_keep_alive="10m",
    )


def test_llm_service_calls_native_ollama_generate_endpoint(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        return DummyOllamaResponse({"response": "Respuesta basada en evidencia."})

    monkeypatch.setattr(llm_module.urllib_request, "urlopen", fake_urlopen)
    service = LLMService(_ollama_settings(), setup_logging("INFO"))

    answer, used_llm = service.generate_answer(
        "Puedo vender en una zona rigida?",
        [
            RetrievedChunk(
                chunk_id="current-zone",
                document_id="ordenanza_227_2019",
                source_title="Ordenanza 227-2019-MDP/C",
                text="No se autoriza comercio ambulatorio en zonas rigidas.",
                score=1.0,
                article_label="16",
                tipo_contenido="zona",
                vigencia="vigente",
            )
        ],
    )

    assert used_llm is True
    assert "Respuesta basada en evidencia." in answer
    assert "[Fuente: Ordenanza 227-2019-MDP/C - Articulo 16 - Estado: VIGENTE]" in answer
    assert captured["url"] == "http://localhost:11434/api/generate"
    assert captured["timeout"] == 120
    assert captured["payload"]["model"] == "qwen3.5:4b"
    assert captured["payload"]["stream"] is False
    assert captured["payload"]["think"] is False
    assert captured["payload"]["keep_alive"] == "10m"
    assert captured["payload"]["options"]["temperature"] == 0.2
    assert captured["payload"]["options"]["num_predict"] == 400
    assert "No se autoriza comercio ambulatorio" in captured["payload"]["prompt"]


def test_llm_service_checks_configured_ollama_model(monkeypatch) -> None:
    def fake_urlopen(request, timeout):
        assert request.full_url == "http://localhost:11434/api/tags"
        assert timeout == 120
        return DummyOllamaResponse(
            {"models": [{"name": "qwen3.5:4b", "model": "qwen3.5:4b"}]}
        )

    monkeypatch.setattr(llm_module.urllib_request, "urlopen", fake_urlopen)
    service = LLMService(_ollama_settings(), setup_logging("INFO"))

    assert service.check_ollama_available() is True


def test_llm_service_ollama_failure_falls_back_without_crashing(monkeypatch) -> None:
    def fake_urlopen(request, timeout):
        raise urllib_error.URLError("connection refused")

    monkeypatch.setattr(llm_module.urllib_request, "urlopen", fake_urlopen)
    service = LLMService(_ollama_settings(), setup_logging("INFO"))

    answer, used_llm = service.generate_answer(
        "Que requisitos necesito?",
        [
            RetrievedChunk(
                chunk_id="current-requirements",
                document_id="ordenanza_227_2019",
                source_title="Ordenanza 227-2019-MDP/C",
                text="Presentar solicitud para autorizacion.",
                score=1.0,
                article_label="30",
                tipo_contenido="requisito",
                vigencia="vigente",
            )
        ],
    )

    assert used_llm is False
    assert "esto es lo que encontre" in answer.lower()
    assert "estado: vigente" in answer.lower()


def test_llm_service_does_not_call_ollama_without_rag_evidence(monkeypatch) -> None:
    def fail_if_called(request, timeout):
        raise AssertionError("No debe invocarse Ollama sin chunks recuperados.")

    monkeypatch.setattr(llm_module.urllib_request, "urlopen", fail_if_called)
    service = LLMService(_ollama_settings(), setup_logging("INFO"))

    answer, used_llm = service.generate_answer("Cuanto cuesta actualmente?", [])

    assert used_llm is False
    assert "no encontre informacion suficiente" in answer.lower()
