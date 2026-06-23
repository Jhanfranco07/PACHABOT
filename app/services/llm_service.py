from __future__ import annotations

import json
import logging
import re
import socket
import time
from urllib import error as urllib_error
from urllib import request as urllib_request

from app.config import Settings
from app.core.prompts import (
    GENERAL_CHAT_SYSTEM_PROMPT,
    INTENT_INTERPRETATION_SYSTEM_PROMPT,
    NO_INFO_PROMPT,
    QUERY_REWRITE_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    build_answer_messages,
    build_general_chat_messages,
    build_query_rewrite_messages,
)
from app.models.schemas import ConversationTurn, RetrievedChunk

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None


OLLAMA_DISABLED_MESSAGE = (
    "Ollama está desactivado. Para usar IA local, configura "
    "OLLAMA_ENABLED=true e inicia Ollama manualmente."
)
OPENAI_API_KEY_MISSING_MESSAGE = "Falta configurar OPENAI_API_KEY en el archivo .env."
OPENAI_MODEL_UNAVAILABLE_MESSAGE = (
    "El modelo configurado en OPENAI_MODEL no está disponible. Revisa el nombre del modelo."
)
OPENAI_QUOTA_MESSAGE = "No se pudo completar la consulta por límite de cuota o saldo de OpenAI."
OPENAI_CONNECTION_MESSAGE = (
    "No se pudo conectar con OpenAI. Revisa tu conexion a internet, OPENAI_BASE_URL "
    "y que la API key este vigente."
)
SUPPORTED_PROVIDERS = {"openai", "ollama", "openrouter", "grok", "groq", "mock"}


class LLMService:
    """Generate answers using an external LLM when available, with honest local fallbacks."""

    def __init__(self, settings: Settings, logger: logging.Logger) -> None:
        self.settings = settings
        self.logger = logger.getChild("llm_service")
        self.client = None
        self.provider = (settings.llm_provider or "openai").lower().strip() or "openai"
        if self.provider not in SUPPORTED_PROVIDERS:
            self.logger.warning(
                "Proveedor LLM no reconocido (%s). Se usara OpenAI por defecto.",
                settings.llm_provider,
            )
            self.provider = "openai"
        self._model_cooldowns: dict[str, float] = {}
        self._inline_system_models: set[str] = set()
        self._inactive_reason: str | None = None

        if settings.llm_mode == "mock" and self.provider != "ollama":
            return

        # CAMBIO FASE OLLAMA 2 — Activar Ollama dentro de la capa LLM existente.
        # Motivo: Telegram y RAG consumen una sola interfaz de generacion.
        # Riesgo mitigado: la rama Ollama no modifica los clientes remotos existentes.
        if self.provider == "ollama":
            if not settings.ollama_enabled:
                self._inactive_reason = OLLAMA_DISABLED_MESSAGE
                self.logger.warning(OLLAMA_DISABLED_MESSAGE)
                return
            self.client = object()
            return

        if self.provider == "openai" and not settings.openai_api_key:
            self._inactive_reason = OPENAI_API_KEY_MISSING_MESSAGE
            self.logger.warning(OPENAI_API_KEY_MISSING_MESSAGE)
            return

        if OpenAI is None:
            self._inactive_reason = "Falta instalar la dependencia openai para usar este proveedor."
            return

        client_kwargs = self._build_client_kwargs()
        if client_kwargs is not None:
            self.client = OpenAI(**client_kwargs)

    def generate_answer(
        self,
        question: str,
        chunks: list[RetrievedChunk],
        *,
        history: list[ConversationTurn] | None = None,
        orchestration_notes: list[str] | None = None,
    ) -> tuple[str, bool]:
        """Return a municipal answer and whether an external LLM was used."""

        history = history or []
        _ = orchestration_notes or []

        if not chunks:
            return self._fallback_answer(question, chunks), False

        # CAMBIO FASE OLLAMA 3 — No generar respuestas normativas sin evidencia RAG.
        # Motivo: impedir que cualquier proveedor invente requisitos, costos o normas.
        # Riesgo mitigado: el fallback ya contiene un mensaje ciudadano controlado.
        if self.client is None:
            return self._fallback_answer(question, chunks), False

        try:
            messages = build_answer_messages(question, chunks, history)
            answer = self._call_provider(
                system_prompt=SYSTEM_PROMPT,
                messages=messages,
                temperature=0.15,
                warning_label="respuesta municipal",
            )
            answer = self._apply_municipal_safety_guards(question, answer, chunks)
            return self._ensure_normative_citation(answer, chunks), True
        except Exception as exc:  # pragma: no cover
            self._log_provider_failure("Fallo el uso del LLM externo", exc)
            if self._should_disable_external_client(exc):
                self.logger.warning(
                    "Se desactivara temporalmente el proveedor externo y se continuara en fallback local."
                )
                self.client = None
            return self._fallback_answer(question, chunks, reason=exc), False

    def generate_general_answer(
        self,
        question: str,
        *,
        history: list[ConversationTurn] | None = None,
    ) -> tuple[str, bool]:
        """Answer a free-form question outside the municipal domain."""

        history = history or []

        if self.client is None:
            return self._fallback_general_answer(question), False

        try:
            messages = build_general_chat_messages(question, history)
            answer = self._call_provider(
                system_prompt=GENERAL_CHAT_SYSTEM_PROMPT,
                messages=messages,
                temperature=0.35,
                warning_label="chat general",
            )
            return answer, True
        except Exception as exc:  # pragma: no cover
            self._log_provider_failure("Fallo el chat general con el proveedor externo", exc)
            if self._should_disable_external_client(exc):
                self.logger.warning(
                    "Se desactivara temporalmente el proveedor externo y el chat general seguira en fallback local."
                )
                self.client = None
            return self._fallback_general_answer(question, reason=exc), False

    def rewrite_query(
        self,
        question: str,
        *,
        history: list[ConversationTurn] | None = None,
    ) -> str:
        """Rewrite a follow-up question into a clearer retrieval query."""

        history = history or []

        if self.client is None:
            return question

        try:
            messages = build_query_rewrite_messages(question, history)
            rewritten = self._call_provider(
                system_prompt=QUERY_REWRITE_SYSTEM_PROMPT,
                messages=messages,
                temperature=0.0,
                warning_label="query rewriting",
            )
            cleaned = rewritten.strip().strip("\"'")
            first_line = cleaned.splitlines()[0].strip() if cleaned else question
            return first_line or question
        except Exception as exc:  # pragma: no cover
            self._log_provider_failure("Fallo la reescritura de consulta con el proveedor externo", exc)
            if self._should_disable_external_client(exc):
                self.logger.warning(
                    "Se desactivara temporalmente el proveedor externo y la reescritura seguira en modo heuristico."
                )
                self.client = None
            return question

    def interpret_intent(
        self,
        question: str,
        *,
        history: list[ConversationTurn] | None = None,
        router_hint: str = "",
    ) -> dict:
        """Ask the LLM to interpret intent when the deterministic router is uncertain."""

        history = history or []
        if self.client is None:
            return {}

        recent_history = "\n".join(
            f"{turn.role}: {turn.text}"
            for turn in history[-6:]
        )
        messages = [
            {
                "role": "user",
                "content": (
                    f"HISTORIAL RECIENTE:\n{recent_history or 'Sin historial'}\n\n"
                    f"PISTA DEL ROUTER:\n{router_hint or 'Sin pista'}\n\n"
                    f"PREGUNTA ACTUAL:\n{question}\n\n"
                    "Interpreta la intencion y devuelve solo JSON."
                ),
            }
        ]
        try:
            raw = self._call_provider(
                system_prompt=INTENT_INTERPRETATION_SYSTEM_PROMPT,
                messages=messages,
                temperature=0.0,
                warning_label="interpretacion de intencion",
            )
        except Exception as exc:  # pragma: no cover
            self._log_provider_failure("Fallo la interpretacion de intencion con el proveedor externo", exc)
            return {}

        try:
            return _parse_json_object(raw)
        except ValueError:
            self.logger.warning(
                "La interpretacion de intencion no devolvio JSON valido; se usara el router local. Respuesta: %r",
                raw[:240],
            )
            return {}

    def _call_provider(
        self,
        *,
        system_prompt: str,
        messages: list[dict[str, str]],
        temperature: float,
        warning_label: str,
    ) -> str:
        """Call the external provider using the most compatible API available."""

        if self.provider == "ollama":
            return self._call_ollama(
                system_prompt=system_prompt,
                messages=messages,
                temperature=temperature,
            )

        candidate_models = self._candidate_models()
        if not candidate_models:
            raise RuntimeError("Todos los modelos configurados estan temporalmente en cooldown.")

        last_exc: Exception | None = None
        for model_name in candidate_models:
            try:
                return self._call_provider_for_model(
                    model_name=model_name,
                    system_prompt=system_prompt,
                    messages=messages,
                    temperature=temperature,
                    warning_label=warning_label,
                )
            except Exception as exc:
                last_exc = exc
                if self._should_try_next_model(exc):
                    self._mark_model_on_cooldown(model_name)
                    self.logger.warning(
                        "El modelo %s fallo para %s y se probara un fallback: %s",
                        model_name,
                        warning_label,
                        exc,
                    )
                    continue
                raise

        if last_exc is not None:
            raise last_exc
        raise RuntimeError("No hay modelos configurados para el proveedor externo.")

    def check_ollama_available(self) -> bool:
        """Return whether the configured Ollama model is installed and reachable."""

        if not self.settings.ollama_enabled:
            self.logger.warning(OLLAMA_DISABLED_MESSAGE)
            return False

        try:
            response = self._request_ollama_json("/api/tags")
        except RuntimeError as exc:
            self.logger.warning("No se pudo verificar Ollama: %s", exc)
            return False

        available_models = {
            model_name
            for model in response.get("models", [])
            if isinstance(model, dict)
            for model_name in (model.get("name"), model.get("model"))
            if isinstance(model_name, str)
        }
        if self.settings.ollama_model not in available_models:
            self.logger.warning(
                "Ollama esta disponible, pero el modelo configurado no fue encontrado: %s",
                self.settings.ollama_model,
            )
            return False

        self.logger.info("Modelo Ollama disponible: %s", self.settings.ollama_model)
        return True

    def list_ollama_models(self) -> list[str]:
        """List locally installed Ollama models for development interfaces."""

        self._ensure_ollama_enabled()
        response = self._request_ollama_json("/api/tags")
        models = {
            model_name
            for model in response.get("models", [])
            if isinstance(model, dict)
            for model_name in (model.get("name"),)
            if isinstance(model_name, str) and model_name.strip()
        }
        return sorted(models)

    def configure_ollama_runtime(
        self,
        *,
        model: str,
        think: bool,
        temperature: float,
        max_tokens: int,
    ) -> None:
        """Apply validated in-memory options for the local web simulator."""

        if self.provider != "ollama":
            raise RuntimeError("El proveedor activo no es Ollama.")
        self._ensure_ollama_enabled()
        installed_models = self.list_ollama_models()
        if model not in installed_models:
            raise ValueError(f"El modelo Ollama no esta instalado: {model}")

        # CAMBIO FASE SIMULADOR 1 - Permitir elegir configuracion Ollama en pruebas web.
        # Motivo: comparar rapidez y calidad sin modificar .env en cada consulta.
        # Riesgo mitigado: solo acepta modelos instalados y limites validados por la API.
        self.settings.ollama_model = model
        self.settings.ollama_think = think
        self.settings.ollama_temperature = temperature
        self.settings.ollama_max_tokens = max_tokens
        self.logger.info(
            "Configuracion Ollama temporal actualizada: model=%s think=%s max_tokens=%s",
            model,
            think,
            max_tokens,
        )

    def _call_ollama(
        self,
        *,
        system_prompt: str,
        messages: list[dict[str, str]],
        temperature: float,
    ) -> str:
        """Generate text through Ollama's local native API."""

        self._ensure_ollama_enabled()
        _ = temperature
        response = self._request_ollama_json(
            "/api/generate",
            payload={
                "model": self.settings.ollama_model,
                "prompt": self._render_ollama_prompt(system_prompt, messages),
                "stream": False,
                "think": self.settings.ollama_think,
                "keep_alive": self.settings.ollama_keep_alive,
                "options": {
                    "temperature": self.settings.ollama_temperature,
                    "num_predict": self.settings.ollama_max_tokens,
                },
            },
        )
        generated_text = response.get("response")
        if not isinstance(generated_text, str) or not generated_text.strip():
            raise RuntimeError("Ollama devolvio una respuesta vacia.")
        self.logger.debug(
            "Respuesta Ollama generada con %s tokens de entrada y %s de salida.",
            response.get("prompt_eval_count", "n/d"),
            response.get("eval_count", "n/d"),
        )
        return generated_text.strip()

    def _request_ollama_json(
        self,
        endpoint: str,
        *,
        payload: dict[str, object] | None = None,
    ) -> dict[str, object]:
        """Execute one Ollama request and validate its JSON response."""

        self._ensure_ollama_enabled()
        url = f"{self.settings.ollama_base_url}{endpoint}"
        body = json.dumps(payload).encode("utf-8") if payload is not None else None
        request = urllib_request.Request(
            url=url,
            data=body,
            headers={"Content-Type": "application/json"} if body is not None else {},
            method="POST" if body is not None else "GET",
        )
        try:
            with urllib_request.urlopen(request, timeout=self.settings.ollama_timeout) as raw_response:
                raw_body = raw_response.read().decode("utf-8")
        except urllib_error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:300]
            raise RuntimeError(f"Ollama devolvio HTTP {exc.code}: {detail}") from exc
        except urllib_error.URLError as exc:
            raise RuntimeError(
                "No se pudo conectar con Ollama local. Verifica que Ollama este ejecutandose."
            ) from exc
        except (TimeoutError, socket.timeout) as exc:
            raise RuntimeError("Ollama excedio el tiempo maximo de respuesta.") from exc

        try:
            decoded = json.loads(raw_body)
        except json.JSONDecodeError as exc:
            raise RuntimeError("Ollama devolvio JSON invalido.") from exc
        if not isinstance(decoded, dict):
            raise RuntimeError("Ollama devolvio un formato de respuesta inesperado.")
        return decoded

    def _ensure_ollama_enabled(self) -> None:
        """Stop local model requests unless Ollama was explicitly enabled."""

        if not self.settings.ollama_enabled:
            raise RuntimeError(OLLAMA_DISABLED_MESSAGE)

    @staticmethod
    def _render_ollama_prompt(
        system_prompt: str,
        messages: list[dict[str, str]],
    ) -> str:
        """Render chat-style messages into the prompt accepted by /api/generate."""

        parts = [f"INSTRUCCIONES DEL SISTEMA:\n{system_prompt}"]
        for message in messages:
            role = "ASISTENTE" if message.get("role") == "assistant" else "USUARIO"
            parts.append(f"{role}:\n{message.get('content', '')}")
        parts.append("ASISTENTE:")
        return "\n\n".join(parts)

    def _call_provider_for_model(
        self,
        *,
        model_name: str,
        system_prompt: str,
        messages: list[dict[str, str]],
        temperature: float,
        warning_label: str,
    ) -> str:
        """Call the external provider for a single model candidate."""

        request_temperature = self.settings.openai_temperature if self.provider == "openai" else temperature
        full_messages = [{"role": "system", "content": system_prompt}, *messages]
        inline_system_messages = self._inline_system_prompt(system_prompt, messages)

        if model_name in self._inline_system_models:
            return self._call_chat_completions(
                model_name=model_name,
                messages=inline_system_messages,
                temperature=request_temperature,
            )

        if not self._should_use_responses_api():
            return self._call_chat_with_compatibility(
                model_name=model_name,
                warning_label=warning_label,
                full_messages=full_messages,
                inline_system_messages=inline_system_messages,
                temperature=request_temperature,
            )

        try:
            response_kwargs: dict[str, object] = {
                "model": model_name,
                "input": full_messages,
                "max_output_tokens": self.settings.openai_max_output_tokens,
                "temperature": request_temperature,
            }
            response = self.client.responses.create(**response_kwargs)
            output_text = getattr(response, "output_text", "").strip()
            if output_text:
                return output_text
        except Exception as exc:
            self.logger.warning(
                "Fallo el endpoint responses para %s con %s; se intentara chat.completions: %s",
                warning_label,
                model_name,
                exc,
            )
            if self._requires_inline_system_prompt(exc):
                self._inline_system_models.add(model_name)
                self.logger.warning(
                    "El proveedor no acepta instrucciones de sistema para %s con %s; se reintentara embebiendolas en el mensaje del usuario.",
                    warning_label,
                    model_name,
                )
                return self._call_chat_completions(
                    model_name=model_name,
                    messages=inline_system_messages,
                    temperature=request_temperature,
                )

        return self._call_chat_with_compatibility(
            model_name=model_name,
            warning_label=warning_label,
            full_messages=full_messages,
            inline_system_messages=inline_system_messages,
            temperature=request_temperature,
        )

    def _fallback_answer(
        self,
        question: str,
        chunks: list[RetrievedChunk],
        *,
        reason: Exception | None = None,
    ) -> str:
        """Return an honest fallback when no municipal LLM is active."""

        provider_message = self._provider_error_message(reason)
        if provider_message and chunks:
            return provider_message

        memory_limited_ollama = (
            reason is not None
            and self.provider == "ollama"
            and "requires more system memory" in str(reason).lower()
        )
        if reason is not None and self._should_try_next_model(reason):
            if chunks:
                best = chunks[0]
                source = best.source_title
                if best.article_label:
                    source += f", Articulo {best.article_label} (Art. {best.article_label})"
                elif best.section_title:
                    source += f", {best.section_title}"
                return (
                    "Los modelos gratuitos estan temporalmente saturados, asi que no pude hacer una "
                    "explicacion con IA en este momento.\n\n"
                    f"Lo mas relevante que encontre en la base documental fue: {source}.\n\n"
                    "Si quieres, puedes reintentar en un minuto para que vuelva a intentar sintetizarlo."
                )
            return (
                "Los modelos gratuitos estan temporalmente saturados y no pude generar una respuesta "
                "con IA en este momento. Intenta de nuevo en un minuto."
            )

        if not chunks:
            normalized_question = question.lower()
            if any(term in normalized_question for term in ("cuanto cuesta", "costo", "monto", "pago")):
                return (
                    "Por ahora no encontré el costo exacto actualizado en los documentos cargados. "
                    "⚠️ Ese monto debe verificarse en el TUPA vigente o con el área municipal responsable "
                    "antes de realizar el pago."
                )
            if any(term in normalized_question for term in ("ubicacion", "zona", "jr.", "jiron", "avenida")):
                return (
                    "Por ahora no tengo información suficiente para confirmar esa ubicación. "
                    "Para orientarte mejor, dime la avenida, calle o referencia exacta; luego conviene "
                    "validarlo con el área municipal competente y el plano vigente."
                )
            return (
                "Por ahora no encontré información suficiente en los documentos cargados para responderte "
                "con seguridad. Te recomiendo validarlo con el área municipal correspondiente."
            )

        # Produce una respuesta clara y paraphraseada del fragmento recuperado
        best = chunks[0]
        if best.tipo_contenido == "definicion":
            return self._fallback_definition_answer(best)

        if best.document_id == "requisitos_comercio_ambulatorio":
            return self._fallback_requirement_answer(question, best, chunks)

        source = best.fuente or best.source_title
        if best.article_label and not best.fuente:
            source += f" - Art. {best.article_label}"
        elif best.section_title and not best.fuente:
            source += f" - {best.section_title}"

        # Simple heuristic para resumir: tomar oraciones principales del fragmento
        raw = " ".join(best.text.split()).strip()
        safe_raw = re.sub(r"(?<=\d)\.(?=\d)", "__DECIMAL__", raw.replace("S/.", "S/"))
        sentences = [
            sentence.strip().replace("__DECIMAL__", ".")
            for sentence in safe_raw.replace('"', '').split(".")
            if sentence.strip()
        ]
        summary_sentences = sentences[:3] if sentences else [raw]
        summary = ". ".join(summary_sentences).rstrip()
        if not summary.endswith('.'):
            summary += '.'
        extra_summaries: list[str] = []
        for extra in chunks[1:3]:
            if extra.chunk_id == best.chunk_id:
                continue
            extra_raw = " ".join(extra.text.split()).strip()
            if not extra_raw:
                continue
            extra_safe = re.sub(r"(?<=\d)\.(?=\d)", "__DECIMAL__", extra_raw.replace("S/.", "S/"))
            extra_sentences = [
                sentence.strip().replace("__DECIMAL__", ".")
                for sentence in extra_safe.replace('"', '').split(".")
                if sentence.strip()
            ]
            if extra_sentences:
                extra_summaries.append(". ".join(extra_sentences[:3]))
        if extra_summaries:
            summary = f"{summary} Tambien encontre: {' '.join(extra_summaries)}."

        # Construir respuesta en lenguaje claro
        detail = (
            "Ollama no pudo cargar el modelo configurado por memoria insuficiente. "
            "Para desarrollo local usa OLLAMA_MODEL=qwen3.5:0.8b o libera memoria y vuelve a intentar."
            if memory_limited_ollama
            else "Te dejo esta orientación con la evidencia disponible."
        )
        if reason is not None and not memory_limited_ollama:
            detail = (
                "No pude usar el proveedor LLM configurado en este momento; "
                "te dejo la evidencia recuperada por el RAG."
            )
        paraphrased = (
            f"Claro, te explico con lo que encontré en la base documental: {summary}\n\n"
            f"📌 Fuente: {source}\n\n"
            f"{detail}"
        )
        if best.requires_review:
            paraphrased += (
                "\n\nEsta orientacion debe confirmarse con el area municipal competente "
                "porque la fuente esta pendiente de validacion."
            )
        return paraphrased

    def _fallback_requirement_answer(
        self,
        question: str,
        best: RetrievedChunk,
        chunks: list[RetrievedChunk],
    ) -> str:
        """Render structured requirement evidence plainly when no LLM is active."""

        tipo_tramite = str(best.metadata.get("tipo_tramite", ""))
        normalized_question = question.lower()
        simple_explanation = _extract_plain_section(best.text, "Version simple:", "No confundir:")
        if simple_explanation and (
            len(normalized_question.split()) <= 5
            or any(marker in normalized_question for marker in ("q ", " pa ", "hijito", "desayuno"))
        ):
            source = best.fuente or best.source_title
            return f"Claro 😊 {simple_explanation}\n\nFuente: {source}"

        requisitos = _extract_bulleted_section(best.text, "Requisitos:", "Explicacion ciudadana:")
        source = best.fuente or best.source_title
        cost_requested = any(chunk.tipo_contenido == "costo" for chunk in chunks)

        if tipo_tramite == "renovacion":
            intro = "Claro 😊 Para renovar tu permiso de comercio ambulatorio, necesitas:"
        else:
            intro = "Claro 😊 Para sacar tu permiso de comercio ambulatorio por primera vez, necesitas:"

        lines = [intro, ""]
        for index, requisito in enumerate(requisitos, start=1):
            lines.append(f"{index}. {requisito}")

        if tipo_tramite != "renovacion":
            lines.extend(
                [
                    "",
                    "Ademas, la municipalidad revisara si el lugar y el giro son adecuados.",
                ]
            )
        if cost_requested:
            lines.extend(
                [
                    "",
                    "Sobre el costo exacto, debe validarse con el TUPA vigente.",
                ]
            )
        lines.extend(["", f"Fuente: {source}"])
        return "\n".join(lines).strip()

    def _fallback_definition_answer(self, best: RetrievedChunk) -> str:
        """Render a concise definition without jumping ahead to procedures."""

        definition = _extract_plain_section(
            best.text,
            "Definicion ciudadana:",
            "Pregunta orientadora:",
        )
        follow_up = _extract_plain_section(best.text, "Pregunta orientadora:", "")
        if not definition:
            raw = " ".join(best.text.split()).strip()
            definition = re.sub(r"^(Articulo\s+\d+\.?\s*)", "", raw, flags=re.IGNORECASE)
        if follow_up:
            follow_up = _naturalize_definition_follow_up(follow_up)
            follow_up = follow_up.strip()
            if "¿" not in follow_up and not follow_up.startswith("¿"):
                follow_up = f"¿{follow_up}"
            if not follow_up.endswith("?"):
                follow_up = f"{follow_up}?"
        else:
            follow_up = _default_definition_follow_up(best)

        source = best.fuente or best.source_title
        parts = [definition]
        if follow_up:
            parts.extend(["", follow_up])
        parts.extend(["", f"Fuente: {source}"])
        return "\n".join(part for part in parts if part is not None).strip()

    def _ensure_normative_citation(self, answer: str, chunks: list[RetrievedChunk]) -> str:
        """Append a retrieved legal source if a generated municipal answer omitted citation."""

        if not chunks:
            return answer

        # CAMBIO FASE OLLAMA 6 — Garantizar trazabilidad aun si el modelo omite la cita.
        # Motivo: una respuesta normativa debe identificar su evidencia recuperada.
        # Riesgo mitigado: solo se adjunta metadata del primer chunk ya seleccionado por RAG.
        primary = chunks[0]
        source = primary.fuente or primary.source_title
        if primary.article_label and not primary.fuente:
            source += f" - Articulo {primary.article_label}"
        elif primary.section_title and not primary.fuente:
            source += f" - {primary.section_title}"
        rendered = answer.rstrip()
        if "fuente:" not in answer.lower():
            rendered += f"\n\nFuente: {source}"
        if primary.requires_review and "pendiente de validacion" not in answer.lower():
            rendered += (
                "\n\nInformacion orientativa pendiente de validacion; "
                "confirma el dato con el area municipal competente."
            )
        return rendered

    def _apply_municipal_safety_guards(
        self,
        question: str,
        answer: str,
        chunks: list[RetrievedChunk],
    ) -> str:
        """Remove two unsafe overstatements that small local models may introduce."""

        guarded = re.sub(r"\bTUPA\s*\([^)]*\)", "TUPA", answer, flags=re.IGNORECASE)
        guarded = re.sub(
            r"(?is)^\s*(?:hola[!.]?\s*)?soy\s+pachabot[,.]?\s*",
            "Hola, ",
            guarded,
        ).lstrip()
        normalized_question = question.lower()
        supports_miguel_grau_segment = any(
            chunk.article_label == "17.4"
            and "miguel grau" in chunk.text.lower()
            for chunk in chunks
        )
        if "miguel grau" in normalized_question and supports_miguel_grau_segment:
            guarded = re.sub(
                r"(?i)no,\s*no se puede ejercer el comercio ambulatorio en (?:el\s+)?"
                r"(?:jiron|jirón|jr\.)\s+miguel grau\.",
                (
                    "No se autoriza el comercio ambulatorio en el tramo de Jr. Miguel Grau "
                    "identificado como zona rigida por la norma."
                ),
                guarded,
                count=1,
            )
        return guarded

    def classify_response_origin(
        self,
        question: str,
        answer: str,
        chunks: list[RetrievedChunk],
        *,
        used_llm: bool,
    ) -> str:
        """Describe whether an LLM response used recovered document evidence."""

        if not used_llm:
            return "fallback"
        _ = (question, answer)
        if not chunks:
            return "llm_no_evidence"
        return "llm"

    def _provider_error_message(self, reason: Exception | None = None) -> str | None:
        """Map provider configuration/runtime failures to citizen-friendly messages."""

        if self._inactive_reason:
            return self._inactive_reason
        if reason is None:
            return None

        message = str(reason).lower()
        if OLLAMA_DISABLED_MESSAGE.lower() in message:
            return OLLAMA_DISABLED_MESSAGE
        if self.provider == "openai":
            if "api key" in message or "authentication" in message or "401" in message:
                return OPENAI_API_KEY_MISSING_MESSAGE
            if any(term in message for term in ("connection error", "unsupportedprotocol", "unsupported protocol")):
                return OPENAI_CONNECTION_MESSAGE
            if any(term in message for term in ("insufficient_quota", "quota", "billing", "429")):
                return OPENAI_QUOTA_MESSAGE
            if any(
                term in message
                for term in ("model_not_found", "model not found", "does not exist", "not available", "404")
            ):
                return OPENAI_MODEL_UNAVAILABLE_MESSAGE
        return None

    def _fallback_general_answer(self, question: str, *, reason: Exception | None = None) -> str:
        """Return a simple and honest message for general chat without an active LLM."""

        _ = question
        provider_message = self._provider_error_message(reason)
        if provider_message:
            return provider_message
        if reason is not None and self._should_try_next_model(reason):
            return (
                "Los modelos gratuitos estan temporalmente saturados en este momento. "
                "Intenta de nuevo en un minuto o cambia al modo Comercio si quieres consultar las ordenanzas locales."
            )
        return (
            "Para responder preguntas generales con naturalidad necesito "
            "un proveedor LLM activo. Puedes configurarlo con Ollama, xAI, "
            "OpenAI, OpenRouter o Groq desde tu archivo .env."
        )

    def _build_client_kwargs(self) -> dict | None:
        """Build provider-specific OpenAI-compatible client settings."""

        if self.provider == "grok" and self.settings.grok_api_key:
            return {
                "api_key": self.settings.grok_api_key,
                "base_url": self.settings.grok_base_url,
                "max_retries": 0,
            }

        if self.provider == "groq" and self.settings.groq_api_key:
            return {
                "api_key": self.settings.groq_api_key,
                "base_url": self.settings.groq_base_url,
                "max_retries": 0,
            }

        if self.provider == "openrouter":
            api_key = self.settings.openrouter_api_key or self.settings.openai_api_key
            if not api_key:
                return None

            headers: dict[str, str] = {}
            if self.settings.openrouter_http_referer:
                headers["HTTP-Referer"] = self.settings.openrouter_http_referer
            if self.settings.openrouter_app_name:
                headers["X-Title"] = self.settings.openrouter_app_name

            client_kwargs: dict[str, object] = {
                "api_key": api_key,
                "base_url": self.settings.openrouter_base_url,
                "max_retries": 0,
            }
            if headers:
                client_kwargs["default_headers"] = headers
            return client_kwargs

        if self.provider == "openai" and self.settings.openai_api_key:
            base_url = (self.settings.openai_base_url or "").strip() or "https://api.openai.com/v1"
            client_kwargs: dict[str, object] = {
                "api_key": self.settings.openai_api_key,
                "base_url": base_url,
                "max_retries": 0,
            }
            return client_kwargs

        return None

    def _should_disable_external_client(self, exc: Exception) -> bool:
        """Disable the provider after persistent permission or model errors."""

        message = str(exc).lower()
        return any(
            marker in message
            for marker in (
                "403",
                "permission",
                "credits or licenses",
                "model not found",
            )
        )

    def _log_provider_failure(self, prefix: str, exc: Exception) -> None:
        """Avoid giant tracebacks for expected free-tier/provider failures."""

        if self._is_expected_provider_failure(exc):
            self.logger.warning("%s: %s", prefix, exc)
            return
        self.logger.exception("%s: %s", prefix, exc)

    def _call_chat_completions(
        self,
        *,
        model_name: str,
        messages: list[dict[str, str]],
        temperature: float,
    ) -> str:
        """Call the chat completions endpoint with a prepared message list."""

        request_kwargs: dict[str, object] = {
            "model": model_name,
            "messages": messages,
            "temperature": temperature,
        }
        if self.provider == "openai":
            request_kwargs["max_tokens"] = self.settings.openai_max_output_tokens

        completion = self.client.chat.completions.create(**request_kwargs)
        return completion.choices[0].message.content.strip()

    def _should_use_responses_api(self) -> bool:
        """Return whether the current provider should hit the responses endpoint first."""

        return self.provider in {"grok", "openai"}

    def _call_chat_with_compatibility(
        self,
        *,
        model_name: str,
        warning_label: str,
        full_messages: list[dict[str, str]],
        inline_system_messages: list[dict[str, str]],
        temperature: float,
    ) -> str:
        """Call chat.completions and retry with inlined instructions when needed."""

        try:
            return self._call_chat_completions(
                model_name=model_name,
                messages=full_messages,
                temperature=temperature,
            )
        except Exception as exc:
            if self._requires_inline_system_prompt(exc):
                self._inline_system_models.add(model_name)
                self.logger.warning(
                    "El proveedor rechazo el role system para %s con %s; se reintentara con instrucciones embebidas.",
                    warning_label,
                    model_name,
                )
                return self._call_chat_completions(
                    model_name=model_name,
                    messages=inline_system_messages,
                    temperature=temperature,
                )
            raise

    def _candidate_models(self) -> list[str]:
        """Return the ordered list of primary and fallback models to try."""

        if self.provider == "openai":
            models = [self.settings.openai_model]
        else:
            models = [self.settings.chat_model, *self.settings.chat_model_fallbacks]
        ordered: list[str] = []
        seen: set[str] = set()
        for model_name in models:
            compact = model_name.strip()
            if not compact or compact in seen:
                continue
            seen.add(compact)
            ordered.append(compact)
        available_now = [model for model in ordered if not self._is_model_on_cooldown(model)]
        return available_now

    def _inline_system_prompt(
        self,
        system_prompt: str,
        messages: list[dict[str, str]],
    ) -> list[dict[str, str]]:
        """Embed system instructions into the first user message for limited providers."""

        if not messages:
            return [{"role": "user", "content": f"Instrucciones:\n{system_prompt}"}]

        inlined_messages: list[dict[str, str]] = []
        inserted = False
        for message in messages:
            if not inserted and message.get("role") == "user":
                inlined_messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Sigue estas instrucciones del asistente y luego responde normalmente.\n\n"
                            f"{system_prompt}\n\n"
                            "---\n\n"
                            f"{message.get('content', '')}"
                        ),
                    }
                )
                inserted = True
                continue
            inlined_messages.append(message)

        if not inserted:
            inlined_messages.insert(
                0,
                {
                    "role": "user",
                    "content": (
                        "Sigue estas instrucciones del asistente y luego responde normalmente.\n\n"
                        f"{system_prompt}"
                    ),
                },
            )
        return inlined_messages

    def _requires_inline_system_prompt(self, exc: Exception) -> bool:
        """Detect providers that reject system or developer instructions."""

        message = str(exc).lower()
        return any(
            marker in message
            for marker in (
                "developer instruction is not enabled",
                "system instruction is not enabled",
                "role system",
            )
        )

    def _should_try_next_model(self, exc: Exception) -> bool:
        """Detect transient provider/model failures where a fallback model is worth trying."""

        message = str(exc).lower()
        return any(
            marker in message
            for marker in (
                "cooldown",
                "404",
                "no endpoints found",
                "model not available",
                "provider returned error",
                "429",
                "rate-limit",
                "rate limit",
                "temporarily rate-limited upstream",
                "temporarily unavailable",
                "overloaded",
                "try again later",
            )
        )

    def _mark_model_on_cooldown(self, model_name: str) -> None:
        """Temporarily avoid a model that just failed for a transient provider reason."""

        self._model_cooldowns[model_name] = time.time() + self._cooldown_seconds_for_exception()

    def _is_model_on_cooldown(self, model_name: str) -> bool:
        """Return whether a model is temporarily skipped after a recent failure."""

        cooldown_until = self._model_cooldowns.get(model_name)
        if cooldown_until is None:
            return False
        if cooldown_until <= time.time():
            self._model_cooldowns.pop(model_name, None)
            return False
        return True

    def _cooldown_seconds_for_exception(self) -> int:
        """Return the default cooldown window for transient model failures."""

        return self.settings.model_retry_cooldown_seconds

    def _is_expected_provider_failure(self, exc: Exception) -> bool:
        """Detect noisy-but-expected failures from unstable free providers."""

        message = str(exc).lower()
        return any(
            marker in message
            for marker in (
                "ollama",
                "429",
                "404",
                "cooldown",
                "temporarily rate-limited upstream",
                "no endpoints found",
                "developer instruction is not enabled",
                "system instruction is not enabled",
                "connection error",
                "unsupportedprotocol",
                "unsupported protocol",
            )
        )


def _extract_bulleted_section(text: str, start_marker: str, end_marker: str) -> list[str]:
    """Extract simple dash-list items from a rendered evidence section."""

    start = text.find(start_marker)
    if start < 0:
        return []
    start += len(start_marker)
    end = text.find(end_marker, start)
    section = text[start:] if end < 0 else text[start:end]
    items: list[str] = []
    for line in section.splitlines():
        cleaned = line.strip()
        if not cleaned.startswith("-"):
            continue
        cleaned = cleaned.lstrip("-").strip()
        if cleaned:
            items.append(cleaned.rstrip("."))
    return items


def _extract_plain_section(text: str, start_marker: str, end_marker: str) -> str:
    start = text.find(start_marker)
    if start < 0:
        return ""
    start += len(start_marker)
    end = text.find(end_marker, start) if end_marker else -1
    section = text[start:] if end < 0 else text[start:end]
    return re.sub(r"\s+", " ", section).strip()


def _default_definition_follow_up(best: RetrievedChunk) -> str:
    normalized = f"{best.section_title} {best.text}".lower()
    if "padron" in normalized or "padrón" in normalized:
        return "¿Quieres que te explique cómo se ingresa al padrón o qué documentos se presentan?"
    if "tupa" in normalized:
        return "¿Quieres que te explique qué parte del trámite depende del TUPA?"
    if "sisa" in normalized:
        return "¿Quieres que te explique cuándo corresponde pagar SISA?"
    if "modulo" in normalized or "módulo" in normalized or "puesto" in normalized:
        return "¿Quieres que te explique qué condiciones debe cumplir el módulo o en qué zonas puede ubicarse?"
    if "giro" in normalized or "rubro" in normalized:
        return "¿Quieres que te explique qué giros o rubros aparecen como permitidos?"
    return "¿Quieres que te explique cómo sacar el permiso, qué requisitos necesitas o en qué zonas se puede vender?"


def _naturalize_definition_follow_up(follow_up: str) -> str:
    normalized = " ".join(follow_up.strip().strip("¿?").split()).lower()
    if (
        "como sacar el permiso" in normalized
        and "requisitos" in normalized
        and "zonas" in normalized
    ):
        return (
            "Podemos seguir por partes: te puedo explicar cómo sacar el permiso, "
            "qué documentos necesitas, cómo renovar si ya tienes autorización o "
            "en qué zonas no se puede vender. ¿Qué te gustaría revisar?"
        )
    return follow_up


def _parse_json_object(raw: str) -> dict:
    """Parse a provider JSON object, tolerating small wrappers around it."""

    text = raw.strip()
    if not text:
        return {}
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return {}
    try:
        parsed = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}
