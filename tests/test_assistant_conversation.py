from pathlib import Path

from app.channels.schemas import IncomingChatMessage
from app.config import Settings
from app.core.logger import setup_logging
from app.memory.chat_mode_store import ChatModeStore
from app.memory.conversation_store import ConversationMemoryStore
from app.models.domain import AssistantMode, QueryIntent
from app.models.schemas import ConversationTurn, DocumentChunk
from app.services.assistant_service import AssistantService
from app.services.llm_service import LLMService
from app.services.query_router import QueryRouter
from app.services.query_rewriter import QueryRewriter
from app.services.retrieval_service import RetrievalService
from app.tools.document_toolkit import DocumentToolkit


def test_assistant_handles_greeting_without_rejecting_scope(tmp_path: Path) -> None:
    assistant, _memory = build_assistant(tmp_path)

    payload = assistant.answer_chat_message(
        IncomingChatMessage(
            channel="telegram",
            session_id="chat-1",
            user_id="user-1",
            text="Hola",
        )
    )

    assert payload.in_domain is True
    assert "comercio ambulatorio" in payload.answer.lower()


def test_assistant_handles_greeting_with_punctuation_without_llm(tmp_path: Path) -> None:
    assistant, _memory = build_assistant(tmp_path)

    payload = assistant.answer_chat_message(
        IncomingChatMessage(
            channel="web",
            session_id="chat-greeting-punctuation",
            user_id="user-greeting-punctuation",
            text="Hola, buenas tardes",
        )
    )

    assert payload.in_domain is True
    assert payload.used_llm is False
    assert payload.response_origin == "system"
    assert "comercio ambulatorio" in payload.answer.lower()


def test_greeting_prefix_does_not_hide_municipal_question(tmp_path: Path) -> None:
    assistant, _memory = build_assistant(tmp_path)

    payload = assistant.answer_chat_message(
        IncomingChatMessage(
            channel="web",
            session_id="chat-greeting-plus-question",
            user_id="user-greeting-plus-question",
            text="Hola, quiero saber que es comercio ambulatorio",
        )
    )

    assert payload.response_origin != "system"
    assert payload.sources


def test_que_tal_with_ambulatorio_question_is_not_only_a_greeting(tmp_path: Path) -> None:
    assistant, _memory = build_assistant(tmp_path)

    payload = assistant.answer_chat_message(
        IncomingChatMessage(
            channel="web",
            session_id="chat-que-tal-plus-question",
            user_id="user-que-tal-plus-question",
            text="Que tal, consulta, quiero saber que es ser ambulatorio",
        )
    )

    assert payload.response_origin != "system"
    assert payload.in_domain is True


def test_assistant_greeting_mentions_general_mode_when_enabled(tmp_path: Path) -> None:
    assistant, _memory = build_assistant(tmp_path, allow_general_chat=True)
    assistant.set_chat_mode("telegram", "chat-general-greeting", AssistantMode.GENERAL)

    payload = assistant.answer_chat_message(
        IncomingChatMessage(
            channel="telegram",
            session_id="chat-general-greeting",
            user_id="user-general-greeting",
            text="Hola",
        )
    )

    assert "modo general" in payload.answer.lower()


def test_assistant_handles_acknowledgement_without_running_retrieval(tmp_path: Path) -> None:
    assistant, _memory = build_assistant(tmp_path)

    payload = assistant.answer_chat_message(
        IncomingChatMessage(
            channel="telegram",
            session_id="chat-ack",
            user_id="user-ack",
            text="Si, ayudame",
        )
    )

    assert "puedes contarme" in payload.answer.lower()
    assert payload.sources == []


def test_saludo_amable_con_emoji_moderado(tmp_path: Path) -> None:
    assistant, _memory = build_assistant(tmp_path)

    payload = assistant.answer_chat_message(
        IncomingChatMessage("telegram", "chat-friendly-greeting", "user-friendly-greeting", "hola")
    )

    answer = payload.answer.lower()
    assert "pachabot" in answer
    assert "puedes escribirme con tus propias palabras" in answer
    assert payload.answer.count("😊") <= 1


def test_assistant_can_answer_out_of_domain_when_general_mode_is_enabled(tmp_path: Path) -> None:
    assistant, _memory = build_assistant(tmp_path, allow_general_chat=True)
    assistant.set_chat_mode("telegram", "chat-general-mode", AssistantMode.GENERAL)

    payload = assistant.answer_chat_message(
        IncomingChatMessage(
            channel="telegram",
            session_id="chat-general-mode",
            user_id="user-general-mode",
            text="Cuentame un chiste",
        )
    )

    assert "proveedor llm activo" in payload.answer.lower() or "openai_api_key" in payload.answer.lower()
    assert payload.in_domain is False


def test_assistant_uses_general_mode_when_chat_mode_is_general(tmp_path: Path) -> None:
    assistant, _memory = build_assistant(tmp_path)
    assistant.set_chat_mode("telegram", "chat-general-mode-explicit", AssistantMode.GENERAL)

    payload = assistant.answer_chat_message(
        IncomingChatMessage(
            channel="telegram",
            session_id="chat-general-mode-explicit",
            user_id="user-general-mode-explicit",
            text="Que hora es",
        )
    )

    assert "proveedor llm activo" in payload.answer.lower() or "openai_api_key" in payload.answer.lower()
    assert payload.in_domain is False


def test_assistant_blocks_general_question_when_chat_mode_is_commerce(tmp_path: Path) -> None:
    assistant, _memory = build_assistant(tmp_path)
    assistant.set_chat_mode("telegram", "chat-commerce-only", AssistantMode.COMMERCE)

    payload = assistant.answer_chat_message(
        IncomingChatMessage(
            channel="telegram",
            session_id="chat-commerce-only",
            user_id="user-commerce-only",
            text="Que hora es",
        )
    )

    assert "modo comercio" in payload.answer.lower()
    assert payload.in_domain is False


def test_general_question_is_not_hijacked_by_weak_document_matches(tmp_path: Path) -> None:
    assistant, _memory = build_assistant(tmp_path, allow_general_chat=True)
    assistant.set_chat_mode("telegram", "chat-general-time", AssistantMode.GENERAL)

    payload = assistant.answer_chat_message(
        IncomingChatMessage(
            channel="telegram",
            session_id="chat-general-time",
            user_id="user-general-time",
            text="Que hora es",
        )
    )

    assert "proveedor llm activo" in payload.answer.lower() or "openai_api_key" in payload.answer.lower()
    assert payload.in_domain is False


def test_toolkit_uses_history_for_follow_up_queries(tmp_path: Path) -> None:
    _assistant, _memory, toolkit, router = build_assistant(tmp_path, include_router=True)
    history = [
        ConversationTurn(
            role="user",
            text="Explicame la autorizacion municipal para comercio ambulatorio",
        )
    ]

    bundle = toolkit.gather_knowledge("y cuanto dura?", router.route("y cuanto dura?"), history)

    assert bundle.effective_query != "y cuanto dura?"
    assert "Seguimiento" in bundle.effective_query
    assert bundle.chunks


def test_toolkit_uses_history_for_clarifying_follow_up(tmp_path: Path) -> None:
    _assistant, _memory, toolkit, router = build_assistant(tmp_path, include_router=True)
    history = [
        ConversationTurn(
            role="user",
            text="Que es el pago SISA",
        )
    ]

    bundle = toolkit.gather_knowledge(
        "osea que pago 1.00 diario?",
        router.route("osea que pago 1.00 diario?"),
        history,
    )

    assert bundle.effective_query != "osea que pago 1.00 diario?"
    assert "Seguimiento" in bundle.effective_query


def test_toolkit_does_not_force_history_into_new_topic(tmp_path: Path) -> None:
    _assistant, _memory, toolkit, router = build_assistant(tmp_path, include_router=True)
    history = [
        ConversationTurn(
            role="user",
            text="Que es una zona rigida",
        )
    ]

    bundle = toolkit.gather_knowledge(
        "Cuanto mide un modulo",
        router.route("Cuanto mide un modulo"),
        history,
    )

    assert bundle.effective_query == "Cuanto mide un modulo"


def test_toolkit_prepares_query_with_probable_typo_correction(tmp_path: Path) -> None:
    _assistant, _memory, toolkit, _router = build_assistant(tmp_path, include_router=True)

    corrected, notes = toolkit.prepare_query("ke es una sona rigida")

    assert corrected == "que es una zona rigida"
    assert notes


def test_toolkit_does_not_damage_normal_queries(tmp_path: Path) -> None:
    _assistant, _memory, toolkit, _router = build_assistant(tmp_path, include_router=True)

    corrected, notes = toolkit.prepare_query("puedo salir a vender en la calle sin autorizacion")

    assert corrected == "puedo salir a vender en la calle sin autorizacion"
    assert notes == []


def test_assistant_reset_clears_session_memory(tmp_path: Path) -> None:
    assistant, memory = build_assistant(tmp_path)
    first_message = IncomingChatMessage(
        channel="telegram",
        session_id="chat-2",
        user_id="user-2",
        text="Explicame la autorizacion municipal",
    )
    second_message = IncomingChatMessage(
        channel="telegram",
        session_id="chat-2",
        user_id="user-2",
        text="y cuanto dura?",
    )

    assistant.answer_chat_message(first_message)
    assistant.answer_chat_message(second_message)
    assert len(memory.load_history("telegram", "chat-2")) == 4

    assistant.reset_conversation("telegram", "chat-2")
    assert memory.load_history("telegram", "chat-2") == []


def test_assistant_can_answer_from_memory_about_first_question(tmp_path: Path) -> None:
    assistant, _memory = build_assistant(tmp_path)
    assistant.answer_chat_message(
        IncomingChatMessage(
            channel="telegram",
            session_id="chat-memory",
            user_id="user-memory",
            text="Que es el pago sisa",
        )
    )

    payload = assistant.answer_chat_message(
        IncomingChatMessage(
            channel="telegram",
            session_id="chat-memory",
            user_id="user-memory",
            text="que te pregunte primero",
        )
    )

    assert "la primera consulta clara" in payload.answer.lower()
    assert "que es el pago sisa" in payload.answer.lower()


def test_fallback_explains_sisa_instead_of_echoing_random_text(tmp_path: Path) -> None:
    assistant, _memory = build_assistant(tmp_path)

    payload = assistant.answer_chat_message(
        IncomingChatMessage(
            channel="telegram",
            session_id="chat-3",
            user_id="user-3",
            text="Que es el pago sisa",
        )
    )

    assert "claro, te explico" in payload.answer.lower()
    assert "sisa" in payload.answer.lower()


def test_fallback_sounds_more_conversational_for_sisa_summary(tmp_path: Path) -> None:
    assistant, _memory = build_assistant(tmp_path)

    payload = assistant.answer_chat_message(
        IncomingChatMessage(
            channel="telegram",
            session_id="chat-sisa-style",
            user_id="user-sisa-style",
            text="Hablame del sisa",
        )
    )

    assert "claro, te explico" in payload.answer.lower()
    assert "segun la evidencia recuperada" not in payload.answer.lower()
    assert "s/ 1.00" in payload.answer.lower() or "s/. 1.00" in payload.answer.lower()


def test_follow_up_confirmation_about_sisa_answers_directly(tmp_path: Path) -> None:
    assistant, _memory = build_assistant(tmp_path)
    assistant.answer_chat_message(
        IncomingChatMessage(
            channel="telegram",
            session_id="chat-sisa-follow-up",
            user_id="user-sisa-follow-up",
            text="Hablame del sisa",
        )
    )

    payload = assistant.answer_chat_message(
        IncomingChatMessage(
            channel="telegram",
            session_id="chat-sisa-follow-up",
            user_id="user-sisa-follow-up",
            text="entonces si es un sol",
        )
    )

    assert "claro, te explico" in payload.answer.lower()
    assert "s/ 1.00" in payload.answer.lower() or "s/. 1.00" in payload.answer.lower() or "sisa" in payload.answer.lower()


def test_fallback_answers_module_question_with_honest_limit(tmp_path: Path) -> None:
    assistant, _memory = build_assistant(tmp_path)

    payload = assistant.answer_chat_message(
        IncomingChatMessage(
            channel="telegram",
            session_id="chat-4",
            user_id="user-4",
            text="Cuanto mide un modulo",
        )
    )

    assert "claro, te explico" in payload.answer.lower()
    assert "especificaciones tecnicas" in payload.answer.lower() or "parametros tecnicos" in payload.answer.lower()


def test_fallback_article_answer_strips_title_noise(tmp_path: Path) -> None:
    assistant, _memory = build_assistant(tmp_path)

    payload = assistant.answer_chat_message(
        IncomingChatMessage(
            channel="telegram",
            session_id="chat-5",
            user_id="user-5",
            text="Que dice el articulo 7",
        )
    )

    assert "art. 7" in payload.answer.lower()
    assert "autorizacion municipal es personal" in payload.answer.lower()


def test_fallback_case_answer_orients_the_user_from_a_scenario(tmp_path: Path) -> None:
    assistant, _memory = build_assistant(tmp_path)

    payload = assistant.answer_chat_message(
        IncomingChatMessage(
            channel="telegram",
            session_id="chat-6",
            user_id="user-6",
            text="Que pasa si un comerciante vende en via publica sin autorizacion municipal",
        )
    )

    assert "claro, te explico" in payload.answer.lower()
    assert "autorizacion" in payload.answer.lower()


def test_sacar_permiso_prioritizes_new_requirement_case_over_validity(tmp_path: Path) -> None:
    assistant, _memory = build_assistant(
        tmp_path,
        chunks=[
            DocumentChunk(
                chunk_id="req-new",
                document_id="requisitos_comercio_ambulatorio",
                source_title="Ficha interna de requisitos de comercio ambulatorio",
                text=(
                    "Tramite nuevo / ingreso al padron municipal\n"
                    "Requisitos:\n"
                    "- Solicitud con caracter de declaracion jurada de ingreso al padron municipal.\n"
                    "- Declaraciones juradas correspondientes.\n"
                    "- Copia de DNI.\n"
                    "- Foto panoramica del lugar donde se desea realizar el comercio.\n"
                    "Explicacion ciudadana:\n"
                    "Para sacar tu permiso por primera vez, presenta solicitud, declaraciones, DNI y foto del lugar."
                ),
                tipo_contenido="requisito",
                vigencia="vigente",
                prioridad_retrieval=5,
                fuente="Ficha interna de requisitos de comercio ambulatorio",
                knowledge_layer="tramites",
                metadata={"tipo_tramite": "nuevo_ingreso_padron"},
            ),
            DocumentChunk(
                chunk_id="vig-005",
                document_id="ordenanza_227_2019",
                source_title="Ordenanza 227-2019-MDP/C",
                text="Articulo 5.- La autorizacion municipal tiene vigencia de un año.",
                article_label="5",
                tipo_contenido="procedimiento",
                vigencia="vigente",
                prioridad_retrieval=3,
            ),
        ],
    )

    payload = assistant.answer_chat_message(
        IncomingChatMessage(
            channel="telegram",
            session_id="chat-sacar-permiso",
            user_id="user-sacar-permiso",
            text="Como puedo sacar un permiso para comercio ambulatorio",
        )
    )

    assert payload.intent == QueryIntent.REQUISITOS_NUEVO
    assert "permiso de comercio ambulatorio por primera vez" in payload.answer.lower()
    assert "foto panoramica" in payload.answer.lower()
    assert "dos fotos" not in payload.answer.lower()
    assert "voucher" not in payload.answer.lower()


def test_requisitos_nuevo_no_mezcla_renovacion_en_lenguaje_ciudadano(tmp_path: Path) -> None:
    assistant, _memory = build_assistant(tmp_path, chunks=_differentiated_requirement_chunks())

    for index, question in enumerate(
        [
            "Como saco mi permiso de comercio ambulatorio?",
            "Que necesito para vender?",
            "Quiero vender por primera vez",
            "Soy nuevo, quiero vender",
            "q necesito pa vender",
            "quiero vender desayuno",
        ],
        start=1,
    ):
        payload = assistant.answer_chat_message(
            IncomingChatMessage(
                channel="telegram",
                session_id=f"chat-new-requirements-{index}",
                user_id=f"user-new-requirements-{index}",
                text=question,
            )
        )

        answer = payload.answer.lower()
        assert payload.intent == QueryIntent.REQUISITOS_NUEVO
        assert "claro" in answer
        assert "corrigiendo" not in answer
        assert "primera vez" in answer
        assert "foto panoramica" in answer
        assert "dos fotos" not in answer
        assert "voucher" not in answer


def test_requisitos_renovacion_no_mezcla_ingreso_al_padron(tmp_path: Path) -> None:
    assistant, _memory = build_assistant(tmp_path, chunks=_differentiated_requirement_chunks())

    for index, question in enumerate(
        [
            "Como renuevo mi permiso?",
            "Ya tengo autorizacion y quiero renovar",
            "Mi permiso esta por vencer",
            "tengo mi voucher, que mas llevo",
        ],
        start=1,
    ):
        payload = assistant.answer_chat_message(
            IncomingChatMessage(
                channel="telegram",
                session_id=f"chat-renewal-requirements-{index}",
                user_id=f"user-renewal-requirements-{index}",
                text=question,
            )
        )

        answer = payload.answer.lower()
        assert payload.intent == QueryIntent.REQUISITOS_RENOVACION
        assert "renovar" in answer
        assert "dos fotos" in answer
        assert "voucher" in answer
        assert "foto panoramica" not in answer


def test_requisitos_ambiguos_piden_aclaracion_simple(tmp_path: Path) -> None:
    assistant, _memory = build_assistant(tmp_path, chunks=_differentiated_requirement_chunks())

    payload = assistant.answer_chat_message(
        IncomingChatMessage(
            channel="telegram",
            session_id="chat-ambiguous-requirements",
            user_id="user-ambiguous-requirements",
            text="Requisitos de comercio ambulatorio",
        )
    )

    assert payload.intent == QueryIntent.REQUISITOS_AMBIGUO
    assert "primera vez" in payload.answer.lower()
    assert "renovarla" in payload.answer.lower()
    assert payload.used_llm is False


def test_requisitos_y_costo_responde_parcial_sin_inventar_monto(tmp_path: Path) -> None:
    assistant, _memory = build_assistant(tmp_path, chunks=_differentiated_requirement_chunks(include_cost=True))

    payload = assistant.answer_chat_message(
        IncomingChatMessage(
            channel="telegram",
            session_id="chat-new-cost-and-requirements",
            user_id="user-new-cost-and-requirements",
            text="Que necesito y cuanto cuesta?",
        )
    )

    answer = payload.answer.lower()
    assert payload.intent == QueryIntent.REQUISITOS_NUEVO
    assert "foto panoramica" in answer
    assert "tupa vigente" in answer
    assert "s/" not in answer


def test_definicion_comercio_ambulatorio_orienta_por_etapas(tmp_path: Path) -> None:
    assistant, _memory = build_assistant(
        tmp_path,
        chunks=[*_definition_guidance_chunks(), *_differentiated_requirement_chunks()],
    )

    payload = assistant.answer_chat_message(
        IncomingChatMessage(
            channel="telegram",
            session_id="chat-definition-commerce",
            user_id="user-definition-commerce",
            text="Que es comercio ambulatorio",
        )
    )

    answer = payload.answer.lower()
    assert payload.intent == QueryIntent.DEFINICION
    assert "venta de productos o servicios" in answer
    assert "te puedo explicar" in answer or "podemos seguir" in answer
    assert "sacar el permiso" in answer
    assert "\n1." not in answer
    assert "¿podemos seguir" not in answer
    assert "seleccione" not in answer
    assert "menu" not in answer
    assert "foto panoramica" not in answer
    assert "copia de dni" not in answer
    assert "voucher" not in answer
    assert "tupa vigente" not in answer


def test_definicion_padron_y_tupa_no_adelantan_requisitos_ni_costos(tmp_path: Path) -> None:
    assistant, _memory = build_assistant(
        tmp_path,
        chunks=[*_definition_guidance_chunks(), *_differentiated_requirement_chunks(include_cost=True)],
    )

    padron = assistant.answer_chat_message(
        IncomingChatMessage("telegram", "chat-definition-padron", "user-definition-padron", "Que es padron municipal")
    )
    tupa = assistant.answer_chat_message(
        IncomingChatMessage("telegram", "chat-definition-tupa", "user-definition-tupa", "Que es TUPA")
    )

    assert padron.intent == QueryIntent.DEFINICION
    assert "registro" in padron.answer.lower()
    assert "como se ingresa al padron" in padron.answer.lower()
    assert "foto panoramica" not in padron.answer.lower()
    assert tupa.intent == QueryIntent.DEFINICION
    assert "pagos oficiales" in tupa.answer.lower()
    assert "s/." not in tupa.answer.lower()
    assert "monto exacto" in tupa.answer.lower()


def test_respuesta_si_despues_de_definicion_presenta_opciones(tmp_path: Path) -> None:
    assistant, _memory = build_assistant(
        tmp_path,
        chunks=[*_definition_guidance_chunks(), *_differentiated_requirement_chunks()],
    )
    session_id = "chat-definition-yes"

    assistant.answer_chat_message(
        IncomingChatMessage("telegram", session_id, "user-definition-yes", "Que es comercio ambulatorio")
    )
    payload = assistant.answer_chat_message(
        IncomingChatMessage("telegram", session_id, "user-definition-yes", "si")
    )

    answer = payload.answer.lower()
    assert payload.confidence_level == "orientation"
    assert "puedes decirme con tus propias palabras" in answer
    assert "sacar el permiso" in answer
    assert "renovar" in answer
    assert "foto panoramica" not in answer
    assert "hazme una consulta concreta" not in answer
    assert "\n1." not in answer
    assert "seleccione" not in answer


def test_requisitos_despues_de_definicion_usa_atajo_a_permiso_nuevo(tmp_path: Path) -> None:
    assistant, _memory = build_assistant(
        tmp_path,
        chunks=[*_definition_guidance_chunks(), *_differentiated_requirement_chunks()],
    )
    session_id = "chat-definition-requirements"

    assistant.answer_chat_message(
        IncomingChatMessage("telegram", session_id, "user-definition-requirements", "Que es comercio ambulatorio")
    )
    payload = assistant.answer_chat_message(
        IncomingChatMessage("telegram", session_id, "user-definition-requirements", "requisitos")
    )

    assert payload.intent == QueryIntent.REQUISITOS_NUEVO
    assert "primera vez" in payload.answer.lower()
    assert "foto panoramica" in payload.answer.lower()


def test_numero_despues_de_opciones_funciona_como_atajo_no_menu(tmp_path: Path) -> None:
    assistant, memory = build_assistant(
        tmp_path,
        chunks=[*_definition_guidance_chunks(), *_differentiated_requirement_chunks()],
    )
    session_id = "chat-definition-number-shortcut"

    assistant.answer_chat_message(
        IncomingChatMessage("telegram", session_id, "user-number-shortcut", "Que es comercio ambulatorio")
    )
    payload = assistant.answer_chat_message(
        IncomingChatMessage("telegram", session_id, "user-number-shortcut", "1")
    )

    assert payload.intent == QueryIntent.REQUISITOS_NUEVO
    assert "primera vez" in payload.answer.lower()
    assert "foto panoramica" in payload.answer.lower()
    context = memory.get_conversation_context("telegram", session_id)
    assert context["last_intent"] == QueryIntent.REQUISITOS_NUEVO.value


def test_aliases_despues_de_opciones_resuelven_renovacion_y_zonas(tmp_path: Path) -> None:
    chunks = [
        *_definition_guidance_chunks(),
        *_differentiated_requirement_chunks(),
        DocumentChunk(
            chunk_id="zone",
            document_id="zonas_restringidas_comercio_ambulatorio",
            source_title="Zonas restringidas",
            text="Las zonas rigidas o prohibidas no autorizan comercio ambulatorio.",
            tipo_contenido="zona",
            vigencia="vigente",
            prioridad_retrieval=4,
            knowledge_layer="zonas",
            fuente="Ordenanza 227-2019-MDP/C",
        ),
    ]
    assistant, _memory = build_assistant(tmp_path, chunks=chunks)

    assistant.answer_chat_message(
        IncomingChatMessage("telegram", "chat-renewal-alias", "user-renewal-alias", "Que es comercio ambulatorio")
    )
    renewal = assistant.answer_chat_message(
        IncomingChatMessage("telegram", "chat-renewal-alias", "user-renewal-alias", "la de renovar")
    )
    assistant.answer_chat_message(
        IncomingChatMessage("telegram", "chat-zone-alias", "user-zone-alias", "Que es comercio ambulatorio")
    )
    zones = assistant.answer_chat_message(
        IncomingChatMessage("telegram", "chat-zone-alias", "user-zone-alias", "zonas")
    )

    assert renewal.intent == QueryIntent.REQUISITOS_RENOVACION
    assert "dos fotos" in renewal.answer.lower()
    assert zones.intent == QueryIntent.ZONAS_RIGIDAS
    assert "zona" in zones.answer.lower()


def test_gracias_cierra_amable_sin_pregunta_repetitiva(tmp_path: Path) -> None:
    assistant, _memory = build_assistant(tmp_path)

    payload = assistant.answer_chat_message(
        IncomingChatMessage("telegram", "chat-thanks", "user-thanks", "gracias")
    )

    answer = payload.answer.lower()
    assert "de nada" in answer
    assert "😊" in payload.answer
    assert "propias palabras" in answer
    assert "tienes alguna otra pregunta" not in answer


def test_no_entiendo_simplifica_con_paciencia(tmp_path: Path) -> None:
    assistant, _memory = build_assistant(
        tmp_path,
        chunks=[*_definition_guidance_chunks(), *_differentiated_requirement_chunks()],
    )
    session_id = "chat-not-understood"
    assistant.answer_chat_message(
        IncomingChatMessage("telegram", session_id, "user-not-understood", "Que es padron municipal")
    )

    payload = assistant.answer_chat_message(
        IncomingChatMessage("telegram", session_id, "user-not-understood", "no entiendo")
    )

    answer = payload.answer.lower()
    assert "registro" in answer or "mas sencillo" in answer or "más sencillo" in answer
    assert "corrigiendo" not in answer


def test_si_porfa_con_contexto_pendiente_no_reinicia_conversacion(tmp_path: Path) -> None:
    assistant, _memory = build_assistant(
        tmp_path,
        chunks=[*_definition_guidance_chunks(), *_differentiated_requirement_chunks()],
    )
    session_id = "chat-yes-please"
    assistant.answer_chat_message(
        IncomingChatMessage("telegram", session_id, "user-yes-please", "Que es comercio ambulatorio")
    )

    payload = assistant.answer_chat_message(
        IncomingChatMessage("telegram", session_id, "user-yes-please", "si porfa")
    )

    answer = payload.answer.lower()
    assert "hazme una consulta concreta" not in answer
    assert "permiso" in answer
    assert "renovar" in answer


def test_memory_keeps_conversational_context_metadata(tmp_path: Path) -> None:
    assistant, memory = build_assistant(tmp_path)

    assistant.answer_chat_message(
        IncomingChatMessage(
            channel="telegram",
            session_id="chat-context-metadata",
            user_id="user-context-metadata",
            text="Que dice el articulo 7",
        )
    )
    context = memory.get_conversation_context("telegram", "chat-context-metadata")

    assert context["last_intent"]
    assert context["last_sources"]
    assert context["last_response_origin"] == "fallback"


def test_assistant_keeps_in_domain_queries_even_without_clear_chunks(tmp_path: Path) -> None:
    assistant, _memory = build_assistant(tmp_path, chunks=[])

    payload = assistant.answer_chat_message(
        IncomingChatMessage(
            channel="telegram",
            session_id="chat-low-confidence",
            user_id="user-low-confidence",
            text="Que pasa con las ferias gastronomicas escolares",
        )
    )

    assert "documentos cargados" in payload.answer.lower()
    assert "área municipal" in payload.answer.lower()
    assert payload.in_domain is True


def test_respuesta_cita_ordenanza_y_articulo(tmp_path: Path) -> None:
    assistant, _memory = build_assistant(tmp_path)

    payload = assistant.answer_chat_message(
        IncomingChatMessage(
            channel="telegram",
            session_id="chat-citation",
            user_id="user-citation",
            text="Que dice el articulo 7",
        )
    )

    assert "ordenanza" in payload.answer.lower()
    assert "articulo" in payload.answer.lower()
    assert "estado: vigente" not in payload.answer.lower()


def test_bot_no_responde_con_articulo_historico(tmp_path: Path) -> None:
    chunks = [
        DocumentChunk(
            chunk_id="old-30",
            document_id="ordenanza_108_2012",
            source_title="Ordenanza 108-2012-MDP/C",
            text="Articulo 30. Requisitos antiguos de autorizacion.",
            article_label="30",
            tipo_contenido="requisito",
            vigencia="historico",
            prioridad_retrieval=1,
        ),
        DocumentChunk(
            chunk_id="current-30",
            document_id="ordenanza_227_2019",
            source_title="Ordenanza 227-2019-MDP/C",
            text="Articulo 30. Presentar solicitud y fotografias para autorizacion.",
            article_label="30",
            tipo_contenido="requisito",
            vigencia="vigente",
            prioridad_retrieval=3,
        ),
    ]
    assistant, _memory = build_assistant(tmp_path, chunks=chunks)

    payload = assistant.answer_chat_message(
        IncomingChatMessage(
            channel="telegram",
            session_id="chat-current-law",
            user_id="user-current-law",
            text="Que documentos necesito para vender en la calle",
        )
    )

    assert "Ordenanza 227-2019-MDP/C" in payload.answer
    assert "Ordenanza 108-2012-MDP/C" not in payload.answer


def test_requisitos_actualizados_no_mezclan_chunks_secundarios(tmp_path: Path) -> None:
    chunks = [
        DocumentChunk(
            chunk_id="current-30",
            document_id="ordenanza_227_2019",
            source_title="Ordenanza 227-2019-MDP/C",
            text="Articulo 30. Presentar solicitud para autorizacion.",
            article_label="30",
            tipo_contenido="requisito",
            vigencia="vigente",
            prioridad_retrieval=3,
        ),
        DocumentChunk(
            chunk_id="other-zone",
            document_id="ordenanza_108_2012",
            source_title="Ordenanza 108-2012-MDP/C",
            text="Articulo 10. La ubicacion es autorizada por la Municipalidad.",
            article_label="10",
            tipo_contenido="requisito",
            vigencia="vigente",
            prioridad_retrieval=3,
        ),
    ]
    _assistant, _memory, toolkit, router = build_assistant(
        tmp_path,
        include_router=True,
        chunks=chunks,
    )
    question = "Que requisitos necesito para vender en la via publica?"

    bundle = toolkit.gather_knowledge(question, router.route(question), [])

    assert [chunk.article_label for chunk in bundle.chunks] == ["30"]


def test_consulta_de_ordenanza_usa_norma_base_y_modificatoria(tmp_path: Path) -> None:
    chunks = [
        DocumentChunk(
            chunk_id="base-title",
            document_id="ordenanza_108_2012",
            source_title="Ordenanza 108-2012-MDP/C",
            text="ORDENANZA QUE REGLAMENTA EL COMERCIO AMBULATORIO Y FERIAL EN EL DISTRITO.",
            tipo_contenido="disposicion",
            prioridad_retrieval=2,
        ),
        DocumentChunk(
            chunk_id="mod-title",
            document_id="ordenanza_227_2019",
            source_title="Ordenanza 227-2019-MDP/C",
            text="ORDENANZA MODIFICATORIA DE LA ORDENANZA N° 108-2012-MDP/C.",
            tipo_contenido="considerando",
            prioridad_retrieval=0,
        ),
        DocumentChunk(
            chunk_id="noise",
            document_id="ordenanza_108_2012",
            source_title="Ordenanza 108-2012-MDP/C",
            text="Articulo 12. Se determinan zonas reguladas.",
            article_label="12",
            tipo_contenido="zona",
            prioridad_retrieval=3,
        ),
    ]
    _assistant, _memory, toolkit, router = build_assistant(
        tmp_path,
        include_router=True,
        chunks=chunks,
    )
    question = "Cual es la ordenanza de comercio ambulatorio"

    bundle = toolkit.gather_knowledge(question, router.route(question), [])

    assert {chunk.chunk_id for chunk in bundle.chunks} == {"base-title", "mod-title"}


def test_definicion_prioriza_chunks_definitorios_vigentes(tmp_path: Path) -> None:
    chunks = [
        DocumentChunk(
            chunk_id="definition",
            document_id="ordenanza_227_2019",
            source_title="Ordenanza 227-2019-MDP/C",
            text="Articulo 2. Se entiende por comercio ambulatorio la actividad autorizada.",
            article_label="2",
            tipo_contenido="definicion",
            vigencia="vigente",
            prioridad_retrieval=2,
        ),
        DocumentChunk(
            chunk_id="procedure",
            document_id="ordenanza_227_2019",
            source_title="Ordenanza 227-2019-MDP/C",
            text="Articulo 30. Presentar solicitud para autorizacion municipal.",
            article_label="30",
            tipo_contenido="requisito",
            vigencia="vigente",
            prioridad_retrieval=3,
        ),
    ]
    _assistant, _memory, toolkit, router = build_assistant(
        tmp_path,
        include_router=True,
        chunks=chunks,
    )
    question = "Que es una autorizacion municipal para comercio ambulatorio?"

    bundle = toolkit.gather_knowledge(question, router.route(question), [])

    assert bundle.chunks
    assert all(chunk.tipo_contenido == "definicion" for chunk in bundle.chunks)


def test_bot_activa_no_info_cuando_no_hay_evidencia(tmp_path: Path) -> None:
    assistant, _memory = build_assistant(tmp_path, chunks=[])

    payload = assistant.answer_chat_message(
        IncomingChatMessage(
            channel="telegram",
            session_id="chat-no-info",
            user_id="user-no-info",
            text="Que requisitos tienen las ferias escolares",
        )
    )

    assert "documentos cargados" in payload.answer.lower()
    assert "área municipal" in payload.answer.lower()


def test_bot_no_asigna_sisa_a_costo_de_tramite_no_identificado(tmp_path: Path) -> None:
    assistant, _memory = build_assistant(tmp_path)

    payload = assistant.answer_chat_message(
        IncomingChatMessage(
            channel="telegram",
            session_id="chat-ambiguous-cost",
            user_id="user-ambiguous-cost",
            text="Cuanto cuesta exactamente este tramite actualmente?",
        )
    )

    assert "costo exacto" in payload.answer.lower()
    assert "tupa vigente" in payload.answer.lower()
    assert payload.answer.count("⚠️") <= 1
    assert "s/ 1.00" not in payload.answer.lower()


def test_pregunta_seguimiento_se_reformula_correctamente(tmp_path: Path) -> None:
    _assistant, _memory, toolkit, router = build_assistant(tmp_path, include_router=True)
    history = [
        ConversationTurn(role="user", text="Que requisitos necesito para comercio ambulatorio")
    ]

    bundle = toolkit.gather_knowledge(
        "Y si vendo alimentos?",
        router.route("Y si vendo alimentos?"),
        history,
    )

    assert "Seguimiento" in bundle.effective_query
    assert "alimentos" in bundle.effective_query.lower()


def test_pregunta_de_giros_recupera_rubros_permitidos(tmp_path: Path) -> None:
    chunks = [
        DocumentChunk(
            chunk_id="rubros-21",
            document_id="tramite_comercio_ambulatorio",
            source_title="Ficha de tramite",
            text=(
                "Rubro 3 - Venta de productos preparados al dia\n"
                "- G004: Bebidas saludables: emoliente, quinua, maca, soya.\n"
                "- G007: Sandwiches.\n"
                "Fuente: Ordenanza 227-2019-MDP/C, Articulo 21"
            ),
            section_title="RUBROS Y GIROS PERMITIDOS",
            article_label="21",
            tipo_contenido="rubro",
            vigencia="vigente_con_observacion",
            prioridad_retrieval=3,
            knowledge_layer="tramites",
            fuente="Ordenanza 227-2019-MDP/C, Articulo 21",
            requires_review=True,
        )
    ]
    assistant, _memory = build_assistant(tmp_path, chunks=chunks)

    payload = assistant.answer_chat_message(
        IncomingChatMessage(
            channel="telegram",
            session_id="chat-rubros",
            user_id="user-rubros",
            text="Cuales son los giros?",
        )
    )

    assert payload.intent == QueryIntent.RUBROS
    assert "emoliente" in payload.answer.lower() or "sandwich" in payload.answer.lower()
    assert "te lo busco" not in payload.answer.lower()


def test_follow_up_no_cumple_busca_sanciones_sin_fallback_total(tmp_path: Path) -> None:
    chunks = [
        DocumentChunk(
            chunk_id="obligaciones-57",
            document_id="ordenanza_227_2019",
            source_title="Ordenanza 227-2019-MDP/C",
            text=(
                "Articulo 57. El comerciante ambulante autorizado debe desarrollar "
                "solo el giro autorizado, exhibir su autorizacion y respetar el espacio asignado."
            ),
            article_label="57",
            tipo_contenido="obligacion",
            vigencia="vigente",
            prioridad_retrieval=3,
        ),
        DocumentChunk(
            chunk_id="revocatoria-50",
            document_id="ordenanza_108_2012",
            source_title="Ordenanza 108-2012-MDP/C",
            text=(
                "Articulo 50. El incumplimiento de condiciones de la autorizacion "
                "puede generar revocatoria y retiro del modulo."
            ),
            article_label="50",
            tipo_contenido="sancion",
            vigencia="vigente",
            prioridad_retrieval=3,
        ),
    ]
    assistant, _memory = build_assistant(tmp_path, chunks=chunks)
    assistant.answer_chat_message(
        IncomingChatMessage(
            channel="telegram",
            session_id="chat-no-cumple",
            user_id="user-no-cumple",
            text="Que obligaciones tiene un comerciante ambulante?",
        )
    )

    payload = assistant.answer_chat_message(
        IncomingChatMessage(
            channel="telegram",
            session_id="chat-no-cumple",
            user_id="user-no-cumple",
            text="y si no cumple?",
        )
    )

    assert payload.intent in {QueryIntent.SANCIONES, QueryIntent.OBLIGACIONES}
    assert "base documental cargada" not in payload.answer.lower()
    assert "revocatoria" in payload.answer.lower() or "retiro" in payload.answer.lower()
    assert payload.answer.count("😊") == 0


def test_consulta_manchay_no_activa_fallback_si_hay_zona(tmp_path: Path) -> None:
    chunks = [
        DocumentChunk(
            chunk_id="zona-manchay",
            document_id="zonas_restringidas_comercio_ambulatorio",
            source_title="Zonas restringidas",
            text="Av. Manchay, cuadra 7 y 8, es zona rigida para comercio ambulatorio.",
            article_label="17.4",
            tipo_contenido="zona",
            vigencia="vigente",
            prioridad_retrieval=3,
            knowledge_layer="zonas",
        )
    ]
    assistant, _memory = build_assistant(tmp_path, chunks=chunks)

    payload = assistant.answer_chat_message(
        IncomingChatMessage(
            channel="telegram",
            session_id="chat-manchay",
            user_id="user-manchay",
            text="Puedo vender en Av. Manchay?",
        )
    )

    assert "base documental cargada" not in payload.answer.lower()
    assert "manchay" in payload.answer.lower()


def test_consulta_costo_y_requisitos_responde_evidencia_parcial(tmp_path: Path) -> None:
    assistant, _memory = build_assistant(
        tmp_path,
        chunks=_differentiated_requirement_chunks(include_cost=True),
    )

    payload = assistant.answer_chat_message(
        IncomingChatMessage(
            channel="telegram",
            session_id="chat-cost-req",
            user_id="user-cost-req",
            text="Que necesito y cuanto cuesta?",
        )
    )

    assert "solicitud" in payload.answer.lower()
    assert "tupa" in payload.answer.lower()
    assert "foto panoramica" in payload.answer.lower()
    assert "dos fotos" not in payload.answer.lower()
    assert "base documental cargada" not in payload.answer.lower()


def test_consulta_feria_horario_y_banos_recupera_articulos(tmp_path: Path) -> None:
    chunks = [
        DocumentChunk(
            chunk_id="feria-61",
            document_id="ordenanza_227_2019",
            source_title="Ordenanza 227-2019-MDP/C",
            text="Articulo 61. El horario de la feria debe estar autorizado por la Municipalidad.",
            article_label="61",
            tipo_contenido="horario",
            vigencia="vigente",
            prioridad_retrieval=3,
        ),
        DocumentChunk(
            chunk_id="feria-62",
            document_id="ordenanza_227_2019",
            source_title="Ordenanza 227-2019-MDP/C",
            text="Articulo 62. La feria debe contar con servicios higienicos.",
            article_label="62",
            tipo_contenido="requisito",
            vigencia="vigente",
            prioridad_retrieval=3,
        ),
    ]
    assistant, _memory = build_assistant(tmp_path, chunks=chunks)

    horario = assistant.answer_chat_message(
        IncomingChatMessage("telegram", "chat-feria", "user-feria", "Que horario tiene una feria?")
    )
    banos = assistant.answer_chat_message(
        IncomingChatMessage("telegram", "chat-feria", "user-feria", "Tiene que haber baños en la feria?")
    )

    assert "art. 61" in horario.answer.lower() or "articulo 61" in horario.answer.lower()
    assert "art. 62" in banos.answer.lower() or "articulo 62" in banos.answer.lower()


def _definition_guidance_chunks() -> list[DocumentChunk]:
    return [
        DocumentChunk(
            chunk_id="glosario-comercio-ambulatorio",
            document_id="glosario_comercio_ambulatorio",
            source_title="Glosario ciudadano de comercio ambulatorio",
            text=(
                "Concepto: Comercio ambulatorio\n"
                "Variantes: comercio en via publica | venta ambulante | vender en la calle\n"
                "Definicion ciudadana: El comercio ambulatorio es la venta de productos o servicios "
                "en espacios publicos, como calles o zonas autorizadas, de manera temporal y con "
                "autorizacion de la municipalidad.\n"
                "Pregunta orientadora: Quieres que te explique como sacar el permiso, que requisitos "
                "necesitas o en que zonas se puede vender?"
            ),
            tipo_contenido="definicion",
            vigencia="vigente",
            prioridad_retrieval=5,
            fuente="Ordenanza N. 227-2019-MDP/C",
            knowledge_layer="tramites",
            metadata={
                "concepto": "comercio_ambulatorio",
                "pregunta_orientadora": (
                    "Quieres que te explique como sacar el permiso, que requisitos necesitas "
                    "o en que zonas se puede vender?"
                ),
            },
        ),
        DocumentChunk(
            chunk_id="glosario-padron",
            document_id="glosario_comercio_ambulatorio",
            source_title="Glosario ciudadano de comercio ambulatorio",
            text=(
                "Concepto: Padron municipal\n"
                "Variantes: padron | registro municipal\n"
                "Definicion ciudadana: El padron municipal es un registro donde la municipalidad "
                "identifica a las personas evaluadas o registradas para realizar una actividad, "
                "como el comercio ambulatorio.\n"
                "Pregunta orientadora: Quieres que te explique como se ingresa al padron o que "
                "documentos se presentan?"
            ),
            tipo_contenido="definicion",
            vigencia="vigente",
            prioridad_retrieval=5,
            fuente="Ordenanza N. 227-2019-MDP/C",
            knowledge_layer="tramites",
            metadata={"concepto": "padron_municipal"},
        ),
        DocumentChunk(
            chunk_id="glosario-tupa",
            document_id="glosario_comercio_ambulatorio",
            source_title="Glosario ciudadano de comercio ambulatorio",
            text=(
                "Concepto: TUPA\n"
                "Variantes: pago tupa | derecho tupa\n"
                "Definicion ciudadana: El TUPA es el documento donde la municipalidad publica sus "
                "tramites y pagos oficiales. Para saber un monto exacto, se debe revisar el TUPA vigente.\n"
                "Pregunta orientadora: Quieres que te explique que parte del tramite depende del TUPA?"
            ),
            tipo_contenido="definicion",
            vigencia="vigente",
            prioridad_retrieval=5,
            fuente="TUPA vigente / Ordenanza N. 227-2019-MDP/C",
            knowledge_layer="tramites",
            metadata={"concepto": "tupa"},
        ),
    ]


def _differentiated_requirement_chunks(*, include_cost: bool = False) -> list[DocumentChunk]:
    chunks = [
        DocumentChunk(
            chunk_id="req-new",
            document_id="requisitos_comercio_ambulatorio",
            source_title="Ficha interna de requisitos de comercio ambulatorio",
            text=(
                "Tramite nuevo / ingreso al padron municipal\n"
                "Cuando aplica:\n"
                "- Cuando la persona solicita permiso por primera vez.\n"
                "Requisitos:\n"
                "- Solicitud con caracter de declaracion jurada de ingreso al padron municipal.\n"
                "- Declaraciones juradas correspondientes.\n"
                "- Copia de DNI.\n"
                "- Foto panoramica del lugar donde se desea realizar el comercio.\n"
                "Explicacion ciudadana:\n"
                "Para sacar tu permiso por primera vez, presenta solicitud, declaraciones, DNI y foto del lugar."
            ),
            tipo_contenido="requisito",
            vigencia="vigente",
            prioridad_retrieval=5,
            fuente="Ficha interna de requisitos de comercio ambulatorio",
            knowledge_layer="tramites",
            metadata={"tipo_tramite": "nuevo_ingreso_padron"},
        ),
        DocumentChunk(
            chunk_id="req-renewal",
            document_id="requisitos_comercio_ambulatorio",
            source_title="Ficha interna de requisitos de comercio ambulatorio",
            text=(
                "Renovacion de autorizacion de comercio ambulatorio\n"
                "Cuando aplica:\n"
                "- Cuando la persona ya tiene autorizacion vigente o anterior.\n"
                "Requisitos:\n"
                "- Formato o solicitud de renovacion.\n"
                "- Declaraciones juradas correspondientes.\n"
                "- Copia de DNI.\n"
                "- Dos fotos tamano carne.\n"
                "- Copia del ultimo voucher o comprobante de pago.\n"
                "Explicacion ciudadana:\n"
                "Si ya tienes autorizacion y quieres renovarla, presenta formato, declaraciones, DNI, fotos y voucher."
            ),
            tipo_contenido="requisito",
            vigencia="vigente",
            prioridad_retrieval=5,
            fuente="Ficha interna de requisitos de comercio ambulatorio",
            knowledge_layer="tramites",
            metadata={"tipo_tramite": "renovacion"},
        ),
    ]
    if include_cost:
        chunks.append(
            DocumentChunk(
                chunk_id="cost-tupa",
                document_id="tramite_comercio_ambulatorio",
                source_title="Ficha de tramite: Comercio ambulatorio",
                text="El monto exacto debe verificarse en el TUPA vigente.",
                tipo_contenido="costo",
                vigencia="vigente",
                prioridad_retrieval=3,
                fuente="TUPA vigente",
                knowledge_layer="tramites",
            )
        )
    return chunks


def build_assistant(
    tmp_path: Path,
    include_router: bool = False,
    allow_general_chat: bool = False,
    chunks: list[DocumentChunk] | None = None,
):
    settings = Settings(
        llm_provider="mock",
        llm_mode="mock",
        default_assistant_mode="commerce",
        allow_general_chat=allow_general_chat,
        raw_data_dir=tmp_path / "raw",
        processed_data_dir=tmp_path / "processed",
        vectorstore_dir=tmp_path / "vectorstore",
        conversations_dir=tmp_path / "processed" / "conversations",
        chat_modes_dir=tmp_path / "processed" / "chat_modes",
        processed_chunks_file=tmp_path / "processed" / "chunks.json",
        vectorizer_file=tmp_path / "vectorstore" / "tfidf_vectorizer.joblib",
        matrix_file=tmp_path / "vectorstore" / "tfidf_matrix.joblib",
    )
    logger = setup_logging("INFO")
    router = QueryRouter()
    llm = LLMService(settings, logger)
    retrieval = RetrievalService(settings, logger)
    if chunks is None:
        chunks = [
            DocumentChunk(
                chunk_id="doc-001",
                document_id="ordenanza_108_2012",
                source_title="Ordenanza 108-2012-MDP/C",
                text=(
                    "Articulo 6.- Para ejercer el comercio ambulatorio se requiere "
                    "autorizacion municipal previa."
                ),
                section_title="TITULO III | AUTORIZACION MUNICIPAL",
                article_label="6",
                metadata={},
            ),
            DocumentChunk(
                chunk_id="doc-002",
                document_id="ordenanza_108_2012",
                source_title="Ordenanza 108-2012-MDP/C",
                text=(
                    "TITULO III\nDE LA AUTORIZACION MUNICIPAL\n"
                    "Articulo 7.- La autorizacion municipal es personal e intransferible "
                    "y su otorgamiento sera registrado en el padron correspondiente."
                ),
                section_title="TITULO III | AUTORIZACION MUNICIPAL",
                article_label="7",
                metadata={},
            ),
            DocumentChunk(
                chunk_id="doc-003",
                document_id="ordenanza_108_2012",
                source_title="Ordenanza 108-2012-MDP/C",
                text=(
                    "Articulo 8.- La autorizacion municipal tiene vigencia anual y "
                    "puede renovarse segun evaluacion municipal."
                ),
                section_title="TITULO III | AUTORIZACION MUNICIPAL",
                article_label="8",
                metadata={},
            ),
            DocumentChunk(
                chunk_id="doc-004",
                document_id="ordenanza_108_2012",
                source_title="Ordenanza 108-2012-MDP/C",
                text=(
                    "Articulo 36.- El comerciante informal se encuentra obligado al pago "
                    "por concepto de SISA, cuyo monto es de S/. 1.00 diario."
                ),
                section_title="TITULO VII | PAGOS",
                article_label="36",
                metadata={},
            ),
            DocumentChunk(
                chunk_id="doc-005",
                document_id="ordenanza_227_2019",
                source_title="Ordenanza 227-2019-MDP/C",
                text=(
                    "Zonas rigidas.- Areas de la via publica del distrito en las que, "
                    "por razones de ornato, seguridad o de ordenamiento urbano, no se "
                    "autoriza el ejercicio del comercio en la via publica."
                ),
                section_title="ARTICULO 2 | DEFINICIONES",
                article_label="2",
                metadata={},
            ),
            DocumentChunk(
                chunk_id="doc-006",
                document_id="ordenanza_227_2019",
                source_title="Ordenanza 227-2019-MDP/C",
                text=(
                    "Modulo.- Es el mobiliario desmontable y movible destinado "
                    "exclusivamente para desarrollar la actividad comercial. Debe cumplir "
                    "las especificaciones tecnicas aprobadas por la autoridad municipal."
                ),
                section_title="ARTICULO 2 | DEFINICIONES",
                article_label="2",
                metadata={},
            ),
        ]
    if chunks:
        retrieval.build_index(chunks)
    memory = ConversationMemoryStore(settings, logger)
    mode_store = ChatModeStore(settings, logger)
    query_rewriter = QueryRewriter(settings, llm, logger)
    toolkit = DocumentToolkit(settings, retrieval, query_rewriter, logger)
    assistant = AssistantService(
        settings=settings,
        router=router,
        document_toolkit=toolkit,
        llm_service=llm,
        memory_store=memory,
        mode_store=mode_store,
        logger=logger,
    )

    if include_router:
        return assistant, memory, toolkit, router
    return assistant, memory
