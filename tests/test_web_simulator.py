from fastapi.testclient import TestClient

from app.main import app, container


def test_simulator_serves_local_chat_interface() -> None:
    client = TestClient(app)

    response = client.get("/simulator")

    assert response.status_code == 200
    assert "PachaBot" in response.text
    assert "Modelo LLM" in response.text
    assert "Pensamiento" in response.text


def test_ollama_config_lists_installed_models(monkeypatch) -> None:
    client = TestClient(app)
    monkeypatch.setattr(
        container.llm_service,
        "list_ollama_models",
        lambda: ["qwen3.5:0.8b", "qwen3.5:4b"],
    )
    monkeypatch.setattr(container.settings, "llm_provider", "ollama")
    monkeypatch.setattr(container.settings, "ollama_enabled", True)
    monkeypatch.setattr(container.settings, "ollama_model", "qwen3.5:0.8b")

    response = client.get("/ollama/config")

    assert response.status_code == 200
    payload = response.json()
    assert payload["available"] is True
    assert payload["active_model"] == "qwen3.5:0.8b"
    assert "qwen3.5:4b" in payload["models"]


def test_ollama_config_applies_fast_or_thinking_mode(monkeypatch) -> None:
    client = TestClient(app)
    monkeypatch.setattr(container.settings, "llm_provider", "ollama")
    monkeypatch.setattr(container.settings, "ollama_enabled", True)
    monkeypatch.setattr(container.llm_service, "provider", "ollama")
    monkeypatch.setattr(container.settings, "ollama_model", "qwen3.5:0.8b")
    monkeypatch.setattr(container.settings, "ollama_think", False)
    monkeypatch.setattr(container.settings, "ollama_temperature", 0.2)
    monkeypatch.setattr(container.settings, "ollama_max_tokens", 400)
    monkeypatch.setattr(
        container.llm_service,
        "list_ollama_models",
        lambda: ["qwen3.5:0.8b"],
    )

    response = client.post(
        "/ollama/config",
        json={
            "model": "qwen3.5:0.8b",
            "think": True,
            "temperature": 0.2,
            "max_tokens": 700,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["active_model"] == "qwen3.5:0.8b"
    assert payload["think"] is True
    assert payload["max_tokens"] == 700


def test_ollama_config_does_not_list_models_when_openai_is_active(monkeypatch) -> None:
    client = TestClient(app)

    def fail_if_ollama_is_called():
        raise AssertionError("Ollama should not be queried for OpenAI provider")

    monkeypatch.setattr(container.settings, "llm_provider", "openai")
    monkeypatch.setattr(container.settings, "openai_model", "gpt-5.4-mini")
    monkeypatch.setattr(container.settings, "ollama_enabled", False)
    monkeypatch.setattr(container.llm_service, "client", None)
    monkeypatch.setattr(container.llm_service, "list_ollama_models", fail_if_ollama_is_called)

    response = client.get("/ollama/config")

    assert response.status_code == 200
    payload = response.json()
    assert payload["provider"] == "openai"
    assert payload["models"] == ["gpt-5.4-mini"]
    assert payload["active_model"] == "gpt-5.4-mini"
    assert payload["available"] is False


def test_runtime_config_post_uses_openai_without_touching_ollama(monkeypatch) -> None:
    client = TestClient(app)

    def fail_if_ollama_is_called(*args, **kwargs):
        raise AssertionError("Ollama should not be configured for OpenAI provider")

    monkeypatch.setattr(container.settings, "llm_provider", "openai")
    monkeypatch.setattr(container.settings, "openai_api_key", "test-key")
    monkeypatch.setattr(container.settings, "openai_model", "gpt-5.4-mini")
    monkeypatch.setattr(container.llm_service, "client", object())
    monkeypatch.setattr(container.llm_service, "configure_ollama_runtime", fail_if_ollama_is_called)

    response = client.post(
        "/ollama/config",
        json={
            "model": "gpt-5.4-mini",
            "think": False,
            "temperature": 0.2,
            "max_tokens": 500,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["provider"] == "openai"
    assert payload["available"] is True
    assert payload["active_model"] == "gpt-5.4-mini"


def test_ollama_config_returns_disabled_message_without_querying_ollama(monkeypatch) -> None:
    client = TestClient(app)

    def fail_if_ollama_is_called():
        raise AssertionError("Disabled Ollama should not be queried")

    monkeypatch.setattr(container.settings, "llm_provider", "ollama")
    monkeypatch.setattr(container.settings, "ollama_enabled", False)
    monkeypatch.setattr(container.settings, "ollama_model", "llama3.1")
    monkeypatch.setattr(container.llm_service, "list_ollama_models", fail_if_ollama_is_called)

    response = client.get("/ollama/config")

    assert response.status_code == 200
    payload = response.json()
    assert payload["provider"] == "ollama"
    assert payload["available"] is False
    assert "Ollama está desactivado" in payload["message"]
