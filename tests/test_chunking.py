from app.documents.consolidator import article_status_map
from app.utils.chunking import split_text_into_chunks


def test_split_text_into_multiple_chunks() -> None:
    text = ("Parrafo de prueba. " * 60) + "\n\n" + ("Otro parrafo. " * 60)
    chunks = split_text_into_chunks(
        text,
        document_id="demo",
        source_title="Demo",
        chunk_size=300,
        overlap=50,
    )

    assert len(chunks) >= 2
    assert chunks[0].chunk_id == "demo-001"
    assert chunks[0].source_title == "Demo"


def test_split_legal_text_preserves_article_metadata() -> None:
    text = (
        "TÍTULO III\n"
        "DE LA AUTORIZACION MUNICIPAL\n"
        "Artículo 6°.- Para ejercer el comercio informal se requiere autorización municipal.\n"
        "Artículo 7°.- La autorización municipal es personal e intransferible.\n"
    )
    chunks = split_text_into_chunks(
        text,
        document_id="ordenanza_demo",
        source_title="Ordenanza Demo",
        chunk_size=400,
        overlap=50,
    )

    assert len(chunks) >= 2
    assert chunks[0].section_title.startswith("TÍTULO III")
    assert chunks[0].article_label == "6"
    assert chunks[1].article_label == "7"


def test_split_legal_text_ignores_preamble_article_references() -> None:
    text = (
        "CONSIDERANDO:\n"
        "Que, el artículo 194° de la Constitución Política del Perú establece principios generales.\n"
        "TÍTULO I\n"
        "DEFINICIONES\n"
        "Artículo 2°.- Se entiende por comercio ambulatorio la actividad autorizada.\n"
    )
    chunks = split_text_into_chunks(
        text,
        document_id="ordenanza_demo",
        source_title="Ordenanza Demo",
        chunk_size=400,
        overlap=50,
    )

    assert any(chunk.section_title == "PREAMBULO" and chunk.article_label == "" for chunk in chunks)
    assert any(chunk.article_label == "2" for chunk in chunks)


def test_todos_los_chunks_tienen_metadatos_completos() -> None:
    chunks = split_text_into_chunks(
        'TÍTULO VI\nArtículo 30° "Para obtener autorización se deben presentar documentos."',
        document_id="ordenanza_227_2019",
        source_title="Ordenanza 227-2019-MDP/C",
        article_status=article_status_map("ordenanza_227_2019"),
    )

    assert chunks
    assert all(chunk.normalized_text for chunk in chunks)
    assert all(chunk.tipo_contenido for chunk in chunks)
    assert all(chunk.vigencia for chunk in chunks)
    assert all(chunk.prioridad_retrieval is not None for chunk in chunks)


def test_considerandos_tienen_prioridad_cero() -> None:
    text = (
        "CONSIDERANDO:\nQue, la Ley Orgánica de Municipalidades regula competencias.\n"
        "TÍTULO I\nBASE LEGAL\nArtículo 2°.- Se entiende por comercio la actividad autorizada."
    )
    chunks = split_text_into_chunks(text, document_id="demo", source_title="Demo")

    contextual = [
        chunk for chunk in chunks if chunk.tipo_contenido in {"considerando", "base_legal"}
    ]
    assert contextual
    assert all(chunk.prioridad_retrieval == 0 for chunk in contextual)


def test_articulos_modificados_marcados_correctamente() -> None:
    for label in ("2", "5", "6", "7", "16", "21", "23", "30", "36", "38", "41", "42", "43", "52", "54"):
        status = article_status_map("ordenanza_108_2012")[label]
        assert status["vigencia"] == "historico"
        assert status["modificado_por"]


def test_articulos_ordenanza_227_son_vigentes() -> None:
    statuses = article_status_map("ordenanza_227_2019")
    for label in ("2", "5", "6", "7", "16", "21", "30", "36", "38", "41", "42", "43", "52", "54"):
        assert statuses[label]["vigencia"] == "vigente"
    assert statuses["23"]["vigencia"] == "vigente_con_observacion"
    assert statuses["23"]["requires_review"] is True
    assert statuses["57"]["exclude_from_retrieval"] is True


def test_article_label_no_es_incorrecto() -> None:
    text = (
        "TÍTULO I\nBASE LEGAL, CRITERIOS Y DEFINICIONES\n"
        "artículo 23° de la Ordenanza N° 108 debiendo quedar redactado:\n"
        "Artículo 23°. Para la venta de alimentos se aplican condiciones.\n"
        "ARTICULO DECIMO SEGUNDO. - MODIFICAR el\n"
        "artículo 30° de la Ordenanza N° 108 debiendo quedar redactado:\n"
        'Artículo 30° "Para obtener la autorización se deben presentar documentos."\n'
        "ARTICULO DECIMO TERCERO. - MODIFICAR el\n"
        "artículo 36° de la Ordenanza N° 108 debiendo quedar redactado:\n"
        'Artículo 36° "Se debe pagar SISA por valor diario."\n'
        "ARTICULO DECIMO CUARTO. - MODIFICAR el\n"
        "artículo 38° de la Ordenanza N° 108 debiendo quedar redactado:\n"
        "38°. En caso que no cumpla el pago, procede el retiro y revocación.\n"
    )
    chunks = split_text_into_chunks(
        text,
        document_id="ordenanza_227_2019",
        source_title="Ordenanza 227-2019-MDP/C",
        article_status=article_status_map("ordenanza_227_2019"),
    )

    labels = {
        chunk.article_label
        for chunk in chunks
        if any(term in chunk.text for term in ("documentos", "SISA", "revocación"))
    }
    assert {"30", "36", "38"}.issubset(labels)


def test_referencia_a_titulo_en_considerando_no_inicia_el_cuerpo_normativo() -> None:
    text = (
        "CONSIDERANDO:\n"
        "Que, el Título V de la norma anterior y el artículo 5° de otra ordenanza sirven de antecedente.\n"
        "TÍTULO I\nDEFINICIONES\n"
        "Artículo 2°.- Se entiende por comercio la actividad autorizada.\n"
    )
    chunks = split_text_into_chunks(text, document_id="demo", source_title="Demo")

    assert any(chunk.tipo_contenido == "considerando" for chunk in chunks)
    assert not any(chunk.article_label == "5" for chunk in chunks)
