from app.core.prompts import build_answer_messages, build_conversational_guidance
from app.models.schemas import RetrievedChunk


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
