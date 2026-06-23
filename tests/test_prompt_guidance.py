from app.core.prompts import build_answer_messages, build_conversational_guidance
from app.models.schemas import ConversationTurn, RetrievedChunk


def test_prompt_guidance_for_definition_keeps_orientation_by_steps() -> None:
    guidance = build_conversational_guidance(
        "Que es comercio ambulatorio?",
        [
            RetrievedChunk(
                chunk_id="definition",
                document_id="glosario_comercio_ambulatorio",
                source_title="Glosario ciudadano",
                text="Definicion ciudadana: venta temporal en espacios publicos con autorizacion municipal.",
                score=1.0,
                tipo_contenido="definicion",
            )
        ],
    )

    assert "solo el concepto" in guidance
    assert "opciones breves" in guidance
    assert "producto o servicio vende" not in guidance


def test_prompt_guidance_for_giro_asks_useful_context_not_fixed_answer() -> None:
    guidance = build_conversational_guidance(
        "Y si el giro no pertenece a lo que vendo?",
        [
            RetrievedChunk(
                chunk_id="rubros",
                document_id="tramite_comercio_ambulatorio",
                source_title="Ficha de tramite",
                text="Giro es el tipo de producto o servicio autorizado para vender.",
                score=1.0,
                tipo_contenido="rubro",
            )
        ],
    )

    assert "producto o servicio vende" in guidance
    assert "autorizacion indica algun giro especifico" in guidance


def test_answer_messages_include_context_and_conversational_guidance() -> None:
    messages = build_answer_messages(
        "Puedo vender en Av. Manchay?",
        [
            RetrievedChunk(
                chunk_id="zone",
                document_id="zonas_restringidas_comercio_ambulatorio",
                source_title="Zonas restringidas",
                text="Av. Manchay aparece en tramos sujetos a validacion municipal.",
                score=1.0,
                tipo_contenido="zona",
                knowledge_layer="zonas",
            )
        ],
        [],
    )
    content = messages[-1]["content"]

    assert "CONTEXTO RECUPERADO" in content
    assert "GUIA DE CONTINUIDAD CONVERSACIONAL" in content
    assert "avenida, calle o referencia exacta" in content


def test_first_document_answer_prompt_requests_warm_opening() -> None:
    messages = build_answer_messages(
        "Que requisitos necesito para vender?",
        [
            RetrievedChunk(
                chunk_id="requirements",
                document_id="requisitos_comercio_ambulatorio",
                source_title="Ficha interna",
                text="Requisitos: solicitud, declaraciones juradas, DNI y foto del lugar.",
                score=1.0,
                tipo_contenido="requisito",
            )
        ],
        [],
    )

    content = messages[-1]["content"]

    assert "primera respuesta documental" in content.lower()
    assert "saludo breve" in content.lower()
    assert "maximo 1 emoji" in content.lower()


def test_following_document_answer_prompt_avoids_repeated_greeting() -> None:
    messages = build_answer_messages(
        "Y cuanto cuesta?",
        [
            RetrievedChunk(
                chunk_id="cost",
                document_id="comercio_ambulatorio",
                source_title="Ficha de tramite",
                text="El costo exacto debe validarse con el TUPA vigente.",
                score=1.0,
                tipo_contenido="costo",
            )
        ],
        [ConversationTurn(role="assistant", text="Hola, claro 😊 Ya revisamos los requisitos.")],
    )

    content = messages[-1]["content"]

    assert "no saludes de nuevo" in content.lower()
    assert "manten un tono cercano" in content.lower()
    assert "evita sonar seco" in content.lower()


def test_prompt_guidance_does_not_ask_new_or_renewal_when_case_is_known() -> None:
    guidance = build_conversational_guidance(
        "Que requisitos necesito para vender en la via publica?",
        [
            RetrievedChunk(
                chunk_id="new-requirements",
                document_id="requisitos_comercio_ambulatorio",
                source_title="Ficha interna",
                text="Requisitos del tramite nuevo / ingreso al padron municipal.",
                score=1.0,
                tipo_contenido="requisito",
                metadata={"tipo_tramite": "nuevo_ingreso_padron"},
            )
        ],
    )

    normalized = guidance.lower()
    assert "no preguntes otra vez si es primera vez o renovacion" in normalized
    assert "responde el caso recuperado" in normalized


def test_prompt_guidance_for_numbered_requirements_limits_the_answer() -> None:
    guidance = build_conversational_guidance(
        "y el 2do y 3 er requisitos que son?",
        [
            RetrievedChunk(
                chunk_id="new-requirements",
                document_id="requisitos_comercio_ambulatorio",
                source_title="Ficha interna",
                text=(
                    "Requisitos: solicitud de ingreso al padron; declaracion jurada "
                    "de no tener quejas; declaracion jurada de no superar dos UIT."
                ),
                score=1.0,
                tipo_contenido="requisito",
                metadata={"tipo_tramite": "nuevo_ingreso_padron"},
            )
        ],
    )

    normalized = guidance.lower()
    assert "explica solo esa parte" in normalized
    assert "no repitas toda la lista" in normalized


def test_prompt_guidance_for_focused_follow_up_is_general_not_only_requirements() -> None:
    guidance = build_conversational_guidance(
        "que significa el segundo punto?",
        [
            RetrievedChunk(
                chunk_id="zones",
                document_id="zonas_restringidas_comercio_ambulatorio",
                source_title="Zonas restringidas",
                text="Zonas rigidas; Jr. Miguel Grau; Av. Manchay.",
                score=1.0,
                tipo_contenido="zona",
            )
        ],
    )

    normalized = guidance.lower()
    assert "parte especifica de la respuesta anterior" in normalized
    assert "una zona" in normalized
    assert "no repitas toda la lista" in normalized


def test_prompt_guidance_for_partial_comparison_is_warm_and_limited() -> None:
    messages = build_answer_messages(
        "Que diferencia hay entre comercio ambulatorio y anuncios publicitarios?",
        [
            RetrievedChunk(
                chunk_id="definition-commerce",
                document_id="glosario_comercio_ambulatorio",
                source_title="Glosario ciudadano",
                text="El comercio ambulatorio es venta temporal en espacios publicos con autorizacion municipal.",
                score=1.0,
                tipo_contenido="definicion",
                fuente="Ordenanza N. 227-2019-MDP/C",
            )
        ],
        [],
    )
    content = messages[-1]["content"]

    assert "pregunta compara dos temas" in content.lower()
    assert "no inventes el segundo" in content.lower()
    assert "falta cargar documentos del otro tema" in content.lower()
