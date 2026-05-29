from __future__ import annotations

import logging
import re

from app.channels.schemas import IncomingChatMessage
from app.config import Settings
from app.memory.chat_mode_store import ChatModeStore
from app.memory.conversation_store import ConversationMemoryStore
from app.models.domain import AssistantMode, QueryIntent
from app.models.schemas import AnswerPayload, ConversationTurn, EvidenceItem
from app.services.evidence_service import EvidenceService
from app.services.llm_service import LLMService
from app.services.query_router import QueryRouter
from app.tools.document_toolkit import DocumentToolkit
from app.utils.text_cleaner import normalize_for_search


GREETING_MARKERS = (
    "hola",
    "buenas",
    "buenos dias",
    "buenas tardes",
    "buenas noches",
    "que tal",
)

THANKS_MARKERS = (
    "gracias",
    "muchas gracias",
    "ok gracias",
    "vale gracias",
)

ACKNOWLEDGEMENT_MARKERS = (
    "si",
    "sí",
    "si ayudame",
    "sí ayudame",
    "si, ayudame",
    "sí, ayudame",
    "ayudame",
    "ayúdame",
    "dale",
    "ok",
    "okay",
    "claro",
    "continua",
    "continúa",
    "sigue",
)

FOLLOW_UP_MARKERS = (
    "y ",
    "entonces",
    "osea",
    "o sea",
    "eso",
    "esa",
    "ese",
    "si es",
    "y si",
    "que pasa si",
    "explicamelo",
    "explicame mas simple",
    "mas simple",
    "no entiendo",
    "no entendi",
    "otra cosa",
    "otra pregunta",
    "otra consulta",
    "en que articulo",
    "que articulo",
    "donde dice",
)

ORIENTATION_CONTINUATION_TEXT = (
    "Claro 😊 Podemos ver primero cómo sacar el permiso, los requisitos, "
    "cómo renovar si ya tienes autorización o las zonas donde no se puede vender. "
    "Puedes decirme con tus propias palabras qué parte te gustaría revisar."
)

ORIENTATION_PENDING_OPTIONS = {
    "1": {
        "label": "Cómo sacar el permiso por primera vez",
        "intent": QueryIntent.REQUISITOS_NUEVO.value,
        "query": "Cómo sacar el permiso de comercio ambulatorio por primera vez",
        "aliases": ["primera", "la primera", "permiso", "sacar permiso", "quiero sacar mi permiso"],
    },
    "2": {
        "label": "Qué requisitos se presentan",
        "intent": QueryIntent.REQUISITOS_NUEVO.value,
        "query": "Qué requisitos se presentan para sacar permiso nuevo de comercio ambulatorio por primera vez",
        "aliases": ["segunda", "la segunda", "requisitos", "documentos", "papeles", "que documentos llevo"],
    },
    "3": {
        "label": "Renovación",
        "intent": QueryIntent.REQUISITOS_RENOVACION.value,
        "query": "Cómo renovar autorización de comercio ambulatorio",
        "aliases": ["tercera", "la tercera", "renovar", "renovacion", "renovación", "la de renovar"],
    },
    "4": {
        "label": "Zonas no permitidas",
        "intent": QueryIntent.ZONAS_RIGIDAS.value,
        "query": "Zonas rígidas o prohibidas para comercio ambulatorio",
        "aliases": ["cuarta", "la cuarta", "zonas", "zona", "zonas no permitidas", "zona rigida"],
    },
}


class AssistantService:
    """Main conversational orchestrator for all supported chat channels."""

    def __init__(
        self,
        settings: Settings,
        router: QueryRouter,
        document_toolkit: DocumentToolkit,
        llm_service: LLMService,
        memory_store: ConversationMemoryStore,
        mode_store: ChatModeStore,
        logger: logging.Logger,
        evidence_service: EvidenceService | None = None,
    ) -> None:
        self.settings = settings
        self.router = router
        self.document_toolkit = document_toolkit
        self.llm_service = llm_service
        self.memory_store = memory_store
        self.mode_store = mode_store
        self.logger = logger.getChild("assistant_service")
        self.evidence_service = evidence_service or EvidenceService(settings, logger)

    def answer_chat_message(self, message: IncomingChatMessage) -> AnswerPayload:
        """Answer naturally, adding municipal evidence only when the topic needs it."""

        question = message.text.strip()
        normalized_question = normalize_for_search(question)
        history = self.memory_store.get_recent_history(
            message.channel,
            message.session_id,
            limit=self.settings.memory_history_limit,
        )
        active_mode = self.get_chat_mode(message.channel, message.session_id)

        memory_answer = self._answer_from_memory_if_applicable(normalized_question, history)
        if memory_answer is not None:
            self._remember_exchange(message, question, memory_answer)
            return memory_answer

        pending_option_query = self._resolve_pending_orientation_option(normalized_question, history)
        if pending_option_query:
            question = pending_option_query
            normalized_question = normalize_for_search(question)

        orientation_reply = self._answer_orientation_follow_up(normalized_question, history)
        if orientation_reply is not None:
            self._remember_exchange(message, question, orientation_reply)
            return orientation_reply

        if self._is_greeting(normalized_question):
            return self._remember_social_reply(message, question, self._build_mode_welcome(active_mode))
        if self._is_thanks(normalized_question) or self._is_acknowledgement(normalized_question):
            if self._is_thanks(normalized_question):
                return self._remember_social_reply(
                    message,
                    question,
                    self._build_thanks_reply(active_mode),
                )
            return self._remember_social_reply(
                message,
                question,
                self._build_mode_acknowledgement(active_mode),
            )

        prepared_question, preparation_notes = self.document_toolkit.prepare_query(question)
        routed = self.router.route(prepared_question)
        uses_documents = routed.in_domain or self._is_follow_up_to_in_domain_topic(
            normalized_question,
            history,
        )

        if not uses_documents:
            if active_mode == AssistantMode.COMMERCE and not self.settings.allow_general_chat:
                payload = self._build_payload(
                    answer=(
                        "Este chat está en modo Comercio ambulatorio. Disculpa, no logré entender bien tu consulta. "
                        "¿Te refieres a permiso nuevo, renovación, requisitos o zonas donde se puede vender?"
                    ),
                    sources=[],
                    confidence=0.0,
                    used_llm=False,
                    response_origin="system",
                    in_domain=False,
                    intent=QueryIntent.OUT_OF_SCOPE,
                )
                self._remember_exchange(message, question, payload)
                return payload
            # CAMBIO CONVERSACION UNICA 1 - Toda charla no documental pasa al LLM.
            # Motivo: PachaBot debe conversar naturalmente, incluidos saludos.
            # Riesgo mitigado: los hechos municipales solo se contestan con RAG abajo.
            answer, used_llm = self.llm_service.generate_general_answer(question, history=history)
            payload = self._build_payload(
                answer=answer,
                sources=[],
                confidence=0.25 if used_llm else 0.0,
                used_llm=used_llm,
                response_origin="llm_conversation" if used_llm else "fallback",
                in_domain=False,
                intent=QueryIntent.GENERAL,
            )
            self._remember_exchange(message, question, payload)
            return payload

        if routed.intent == QueryIntent.REQUISITOS_AMBIGUO:
            payload = self._build_payload(
                answer=(
                    "¿Es la primera vez que vas a solicitar el permiso o ya tienes "
                    "autorización y quieres renovarla? Los requisitos cambian según el caso."
                ),
                sources=[],
                confidence=1.0,
                used_llm=False,
                response_origin="system",
                in_domain=True,
                intent=QueryIntent.REQUISITOS_AMBIGUO,
                confidence_level="clarification",
            )
            self._remember_exchange(message, question, payload)
            return payload

        knowledge = self.document_toolkit.gather_knowledge(
            prepared_question,
            routed,
            history,
            original_question=question,
            preparation_notes=preparation_notes,
        )
        assessment = self.evidence_service.assess(knowledge)
        answer_chunks = knowledge.chunks if assessment.sufficient else []
        answer, used_llm = self.llm_service.generate_answer(
            knowledge.effective_query,
            answer_chunks,
            history=history,
        )
        response_origin = self.llm_service.classify_response_origin(
            knowledge.effective_query,
            answer,
            answer_chunks,
            used_llm=used_llm,
        )
        payload = self._build_payload(
            answer=answer,
            sources=(
                knowledge.sources[: self.settings.assistant_max_sources]
                if assessment.sufficient
                else []
            ),
            confidence=knowledge.confidence if assessment.sufficient else 0.0,
            used_llm=used_llm,
            response_origin=response_origin,
            in_domain=True,
            intent=routed.intent if routed.intent != QueryIntent.OUT_OF_SCOPE else QueryIntent.GENERAL,
            evidence=assessment.items,
            evidence_warning=assessment.warning,
            confidence_level=assessment.confidence_level,
        )
        self.evidence_service.write_trace(message, routed, knowledge, assessment, payload)
        self._remember_exchange(message, question, payload)
        return payload

    def answer_user_query(self, question: str) -> AnswerPayload:
        """Convenience wrapper for local/API usage without a Telegram update."""

        return self.answer_chat_message(
            IncomingChatMessage(
                channel="local",
                session_id="local-default",
                user_id="local-user",
                text=question,
            )
        )

    def reset_conversation(self, channel: str, session_id: str) -> None:
        """Clear stored memory for a specific conversation session."""

        self.memory_store.reset_session(channel, session_id)

    def get_runtime_status(
        self,
        *,
        channel: str | None = None,
        session_id: str | None = None,
    ) -> str:
        """Expose a short runtime summary for operational commands."""

        llm_mode = "externo" if self.llm_service.client is not None else "fallback local"
        provider = self.settings.llm_provider.lower().strip()
        if provider == "ollama":
            configured_model = self.settings.ollama_model
        elif provider == "openai":
            configured_model = self.settings.openai_model
        else:
            configured_model = self.settings.chat_model
        chunk_count = len(self.document_toolkit.retrieval_service.chunks)
        _ = (channel, session_id)
        return (
            f"Canal activo: Telegram\n"
            f"Modo de respuesta: {llm_mode}\n"
            f"Proveedor configurado: {self.settings.llm_provider}\n"
            f"Modelo configurado: {configured_model}\n"
            f"Conversacion unica con RAG documental: habilitada\n"
            f"Fragmentos indexados: {chunk_count}\n"
            f"Memoria por sesion: hasta {self.settings.memory_max_turns} turnos"
        )

    def get_chat_mode(self, channel: str, session_id: str) -> AssistantMode:
        """Return the current assistant mode for the given session."""

        return self.mode_store.get_mode(channel, session_id)

    def set_chat_mode(self, channel: str, session_id: str, mode: AssistantMode) -> AssistantMode:
        """Persist and return the chosen assistant mode."""

        self.mode_store.set_mode(channel, session_id, mode)
        return mode

    def describe_mode(self, mode: AssistantMode) -> str:
        """Return a user-facing mode label."""

        if mode == AssistantMode.COMMERCE:
            return "Comercio ambulatorio"
        return "General"

    def _build_payload(
        self,
        *,
        answer: str,
        sources: list[str],
        confidence: float,
        used_llm: bool,
        in_domain: bool,
        intent: QueryIntent,
        response_origin: str | None = None,
        evidence: list[EvidenceItem] | None = None,
        evidence_warning: str = "",
        confidence_level: str = "none",
    ) -> AnswerPayload:
        """Create a normalized answer payload."""

        return AnswerPayload(
            answer=answer.strip(),
            sources=sources,
            intent=intent,
            in_domain=in_domain,
            confidence=confidence,
            used_llm=used_llm,
            response_origin=response_origin or ("llm" if used_llm else "fallback"),
            evidence=evidence or [],
            evidence_warning=evidence_warning,
            confidence_level=confidence_level,
        )

    def _remember_social_reply(
        self,
        message: IncomingChatMessage,
        question: str,
        answer: str,
    ) -> AnswerPayload:
        """Store a social or operational reply in memory."""

        payload = self._build_payload(
            answer=answer,
            sources=[],
            confidence=1.0,
            used_llm=False,
            in_domain=True,
            intent=QueryIntent.GENERAL,
            response_origin="system",
        )
        self._remember_exchange(message, question, payload)
        return payload

    def _remember_exchange(
        self,
        message: IncomingChatMessage,
        user_text: str,
        payload: AnswerPayload,
    ) -> None:
        """Persist the user and assistant turns for follow-up questions."""

        session_metadata = {
            "user_id": message.user_id,
            "user_display_name": message.user_display_name,
            "channel_metadata": message.metadata,
            "conversation_mode": self.get_chat_mode(message.channel, message.session_id).value,
        }
        self.memory_store.append_turn(
            message.channel,
            message.session_id,
            ConversationTurn(
                role="user",
                text=user_text,
                metadata=session_metadata,
            ),
        )
        self.memory_store.append_turn(
            message.channel,
            message.session_id,
            ConversationTurn(
                role="assistant",
                text=payload.answer,
                metadata={
                    "sources": payload.sources,
                    "confidence": payload.confidence,
                    "used_llm": payload.used_llm,
                    "intent": payload.intent.value,
                    "response_origin": payload.response_origin,
                    "confidence_level": payload.confidence_level,
                    "evidence_warning": payload.evidence_warning,
                    "evidence_sources": [item.source for item in payload.evidence],
                    "pending_data": [payload.evidence_warning] if payload.evidence_warning else [],
                    "orientation_options": self._orientation_options_for_payload(payload),
                    "pending_options": self._pending_options_for_payload(payload),
                },
            ),
        )

    def _is_greeting(self, normalized_question: str) -> bool:
        """Detect short greeting-only messages."""

        normalized_question = self._normalize_social_message(normalized_question)
        remainder = normalized_question
        recognized_greeting = False
        for marker in sorted(GREETING_MARKERS, key=len, reverse=True):
            if remainder == marker:
                return True
            if remainder.startswith(f"{marker} "):
                recognized_greeting = True
                remainder = remainder[len(marker) :].strip()
        return recognized_greeting and remainder in GREETING_MARKERS

    def _is_thanks(self, normalized_question: str) -> bool:
        """Detect simple acknowledgement messages."""

        normalized_question = self._normalize_social_message(normalized_question)
        return any(
            normalized_question == marker or normalized_question.startswith(f"{marker} ")
            for marker in THANKS_MARKERS
        )

    def _is_acknowledgement(self, normalized_question: str) -> bool:
        """Detect generic confirmations that are not actual document questions."""

        normalized_question = self._normalize_social_message(normalized_question)
        return any(
            normalized_question == marker or normalized_question.startswith(f"{marker} ")
            for marker in ACKNOWLEDGEMENT_MARKERS
        )

    @staticmethod
    def _normalize_social_message(normalized_question: str) -> str:
        """Ignore punctuation when routing short conversational messages."""

        # CAMBIO FASE SIMULADOR 3 - Reconocer saludos naturales escritos con signos.
        # Motivo: "hola, buenas tardes" no debe tratarse como consulta fuera del dominio.
        # Riesgo mitigado: esta normalizacion solo aplica a respuestas sociales breves.
        return re.sub(r"\s+", " ", re.sub(r"[,.;:!?¿¡]+", " ", normalized_question)).strip()

    def _answer_from_memory_if_applicable(
        self,
        normalized_question: str,
        history: list[ConversationTurn],
    ) -> AnswerPayload | None:
        """Answer simple chat-history questions directly from memory."""

        if not history:
            return None

        asks_first_question = any(
            marker in normalized_question
            for marker in (
                "que te pregunte primero",
                "que te dije primero",
                "cual fue mi primera pregunta",
                "que te pregunte al comienzo",
            )
        )
        if asks_first_question:
            first_user_turn = next(
                (
                    turn
                    for turn in history
                    if turn.role == "user"
                    and not self._is_greeting(normalize_for_search(turn.text))
                    and not self._is_acknowledgement(normalize_for_search(turn.text))
                ),
                None,
            )
            answer = (
                "Todavia no tengo una pregunta previa clara en este chat."
                if first_user_turn is None
                else f'La primera consulta clara que me hiciste en este tramo del chat fue: "{first_user_turn.text}".'
            )
            return self._build_payload(
                answer=answer,
                sources=[],
                confidence=1.0,
                used_llm=False,
                in_domain=True,
                intent=QueryIntent.GENERAL,
            )

        asks_last_answer = any(
            marker in normalized_question
            for marker in (
                "que me respondiste",
                "que dijiste",
                "que me acabas de decir",
                "que respondiste",
            )
        )
        if asks_last_answer:
            last_assistant_turn = next((turn for turn in reversed(history) if turn.role == "assistant"), None)
            answer = (
                "Todavia no tengo una respuesta previa registrada en este chat."
                if last_assistant_turn is None
                else f'Lo ultimo que te respondi fue: "{last_assistant_turn.text}".'
            )
            return self._build_payload(
                answer=answer,
                sources=[],
                confidence=1.0,
                used_llm=False,
                in_domain=True,
                intent=QueryIntent.GENERAL,
            )

        return None

    def _answer_orientation_follow_up(
        self,
        normalized_question: str,
        history: list[ConversationTurn],
    ) -> AnswerPayload | None:
        """Handle short replies to a previous definition-oriented offer."""

        if not self._last_assistant_offered_orientation(history):
            return None

        normalized = self._normalize_social_message(normalized_question)
        if normalized in {"si", "sí", "ya", "claro", "ok", "okay", "dale"}:
            return self._build_payload(
                answer=ORIENTATION_CONTINUATION_TEXT,
                sources=[],
                confidence=1.0,
                used_llm=False,
                response_origin="system",
                in_domain=True,
                intent=QueryIntent.GENERAL,
                confidence_level="orientation",
            )

        return None

    def _resolve_pending_orientation_option(
        self,
        normalized_question: str,
        history: list[ConversationTurn],
    ) -> str | None:
        """Convert a short option reply into a natural RAG query."""

        pending_options = self._last_pending_options(history)
        if not pending_options:
            return None

        normalized = self._normalize_social_message(normalized_question)
        for key, option in pending_options.items():
            if normalized == key:
                return str(option.get("query", "")).strip() or None
            aliases = option.get("aliases", [])
            if any(
                normalized == normalize_for_search(str(alias))
                or normalize_for_search(str(alias)) in normalized
                for alias in aliases
            ):
                return str(option.get("query", "")).strip() or None
        return None

    @staticmethod
    def _orientation_options_for_payload(payload: AnswerPayload) -> list[str]:
        if payload.intent == QueryIntent.DEFINICION:
            return ["permiso", "requisitos", "renovacion", "zonas"]
        if payload.confidence_level == "orientation":
            return ["permiso", "requisitos", "renovacion", "zonas"]
        return []

    @staticmethod
    def _pending_options_for_payload(payload: AnswerPayload) -> dict[str, dict[str, object]]:
        if payload.intent == QueryIntent.DEFINICION or payload.confidence_level == "orientation":
            return ORIENTATION_PENDING_OPTIONS
        return {}

    @staticmethod
    def _last_assistant_offered_orientation(history: list[ConversationTurn]) -> bool:
        last_assistant = next((turn for turn in reversed(history) if turn.role == "assistant"), None)
        if last_assistant is None:
            return False
        options = last_assistant.metadata.get("orientation_options", [])
        if options:
            return True
        normalized = normalize_for_search(last_assistant.text)
        return "quieres que te explique" in normalized and any(
            marker in normalized for marker in ("requisitos", "permiso", "zonas", "renov")
        )

    @staticmethod
    def _last_pending_options(history: list[ConversationTurn]) -> dict[str, dict[str, object]]:
        last_assistant = next((turn for turn in reversed(history) if turn.role == "assistant"), None)
        if last_assistant is None:
            return {}
        pending_options = last_assistant.metadata.get("pending_options", {})
        return pending_options if isinstance(pending_options, dict) else {}

    def _should_treat_as_in_domain(
        self,
        routed,
        knowledge,
        *,
        normalized_question: str,
        history: list[ConversationTurn],
    ) -> bool:
        """Prevent weak document matches from hijacking clearly general questions."""

        if routed.in_domain:
            return True

        if not knowledge.chunks:
            return False

        if self._is_follow_up_to_in_domain_topic(normalized_question, history):
            return knowledge.confidence >= self.settings.confidence_threshold

        # CAMBIO FASE 7.2 — Evitar que relevancia documental sustituya al filtro de dominio.
        # Motivo: las bonificaciones de vigencia no prueban que una pregunta sea municipal.
        # Riesgo mitigado: las repreguntas sobre un tema municipal siguen habilitadas arriba.
        return False

    def _is_follow_up_to_in_domain_topic(
        self,
        normalized_question: str,
        history: list[ConversationTurn],
    ) -> bool:
        """Treat short clarifications as in-domain when the previous user turn clearly was."""

        if not any(
            normalized_question == marker or normalized_question.startswith(marker)
            for marker in FOLLOW_UP_MARKERS
        ):
            return False

        previous_user_turn = next(
            (
                turn
                for turn in reversed(history)
                if turn.role == "user"
                and normalize_for_search(turn.text) != normalized_question
            ),
            None,
        )
        if previous_user_turn is None:
            return False

        previous_routed = self.router.route(previous_user_turn.text)
        return previous_routed.in_domain

    def _build_mode_welcome(self, active_mode: AssistantMode) -> str:
        """Build a short greeting that reflects the active chat mode."""

        if active_mode == AssistantMode.COMMERCE:
            return (
                "¡Hola! 😊 Soy PachaBot. Te puedo orientar sobre comercio ambulatorio, "
                "permisos, requisitos, renovación, zonas rígidas y pagos relacionados. "
                "Puedes escribirme con tus propias palabras."
            )

        return (
            "¡Hola! 😊 Este chat está en modo General. Soy PachaBot y puedo orientarte con palabras sencillas. "
            "Si luego quieres revisar comercio ambulatorio, también puedo ayudarte."
        )

    def _build_thanks_reply(self, active_mode: AssistantMode) -> str:
        """Build a warm closing without sounding like a fixed menu."""

        if active_mode == AssistantMode.COMMERCE:
            return (
                "De nada 😊 Si tienes otra duda sobre tu permiso, renovacion, zona "
                "o requisitos, puedes escribirme con tus propias palabras."
            )

        return "De nada 😊 Cuando quieras, puedes escribirme con tus propias palabras."

    def _build_mode_acknowledgement(self, active_mode: AssistantMode) -> str:
        """Build a follow-up hint that reflects the active chat mode."""

        if active_mode == AssistantMode.COMMERCE:
            return (
                "Claro 😊 Puedes contarme qué necesitas: sacar un permiso, renovar, "
                "revisar requisitos o consultar una zona."
            )

        return (
            "Claro 😊 Puedes escribirme tu consulta con tus propias palabras."
        )
