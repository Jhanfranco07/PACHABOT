from pathlib import Path

from app.channels.schemas import IncomingChatMessage
from app.config import Settings
from app.core.logger import setup_logging
from app.memory.chat_mode_store import ChatModeStore
from app.memory.conversation_store import ConversationMemoryStore
from app.models.domain import AssistantMode
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

    assert "hazme una consulta concreta" in payload.answer.lower()
    assert payload.sources == []


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

    assert "esto es lo que encontre" in payload.answer.lower()
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

    assert "esto es lo que encontre" in payload.answer.lower()
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

    assert "esto es lo que encontre" in payload.answer.lower()
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

    assert "esto es lo que encontre" in payload.answer.lower()
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

    assert "esto es lo que encontre" in payload.answer.lower()
    assert "autorizacion" in payload.answer.lower()


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

    assert "no encontre informacion suficiente" in payload.answer.lower()
    assert "respaldo documental verificado" in payload.answer.lower()
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
    assert "estado: vigente" in payload.answer.lower()


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
            text="Que documentos necesito para una autorizacion",
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

    assert "no encontre informacion suficiente" in payload.answer.lower()
    assert "respaldo documental verificado" in payload.answer.lower()


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

    assert "costo exacto actualizado" in payload.answer.lower()
    assert "tupa vigente" in payload.answer.lower()
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
