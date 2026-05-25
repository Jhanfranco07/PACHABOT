from __future__ import annotations

import re

from app.models.schemas import DocumentChunk
from app.utils.text_cleaner import clean_text, normalize_for_search


# CAMBIO FASE 7.1 — Restringir encabezados a titulos internos numerados.
# Motivo: "Titulo Preliminar de la Ley" no es una seccion de la ordenanza consultada.
# Riesgo mitigado: se admiten variantes legales habituales de titulos y articulos.
TITLE_PATTERN = re.compile(
    r"^((?:T[IÍ]TULO)\s+(?:[IVXLC]+|[0-9]+)(?:\b|\s).*)$",
)
ARTICLE_HEADER_PATTERN = re.compile(
    r"^(?:(?:Artículo|ARTÍCULO|ARTICULO)\s+"
    r"([0-9]+(?:\.[0-9]+)?[A-Z]?)\s*[°º]?\s*(?:\.\s*-?|-|:|[\"“])"
    r"|(?:Artículo|artículo|ARTÍCULO|ARTICULO)\s+([0-9]+(?:\.[0-9]+)?[A-Z]?)\s*"
    r"[°º]?\s+de\s+la\s+Ordenanza\b)",
)
AMENDMENT_ACTION_PATTERN = re.compile(
    r"^(?:ARTÍCULO|ARTICULO)\s+(?:PRIMERO|SEGUNDO|TERCERO|CUARTO|QUINTO|"
    r"SEXTO|S[ÉE]PTIMO|OCTAVO|NOVENO|D[ÉE]CIMO|VIG[ÉE]SIMO)\b.*"
    r"(?:MODIFICAR|INCORPORAR)",
    re.IGNORECASE,
)
SUBSECTION_PATTERN = re.compile(r"^[A-ZÁÉÍÓÚÑ0-9 ,;:/().\-]{3,}$")


def split_text_into_chunks(
    text: str,
    *,
    document_id: str,
    source_title: str,
    chunk_size: int = 700,
    overlap: int = 120,
    article_status: dict[str, dict] | None = None,
) -> list[DocumentChunk]:
    """Split legal text into meaningful chunks using titles and article headers."""

    cleaned = clean_text(text)
    structured_chunks = _extract_structured_legal_chunks(
        cleaned,
        document_id=document_id,
        source_title=source_title,
        chunk_size=chunk_size,
        overlap=overlap,
        article_status=article_status or {},
    )
    if structured_chunks:
        return structured_chunks

    return _fallback_paragraph_chunks(
        cleaned,
        document_id=document_id,
        source_title=source_title,
        chunk_size=chunk_size,
        overlap=overlap,
        article_status=article_status or {},
    )


def _extract_structured_legal_chunks(
    text: str,
    *,
    document_id: str,
    source_title: str,
    chunk_size: int,
    overlap: int,
    article_status: dict[str, dict],
) -> list[DocumentChunk]:
    """Chunk legal documents, preserving titles and ignoring preamble article references."""

    lines = [line.strip() for line in text.splitlines()]
    chunks: list[DocumentChunk] = []
    chunk_number = 1
    base_title = ""
    current_title = ""
    title_buffer: list[str] = []
    seen_regulation_body = False
    preamble_buffer: list[str] = []

    index = 0
    while index < len(lines):
        line = lines[index]
        if not line:
            index += 1
            continue

        if AMENDMENT_ACTION_PATTERN.match(line):
            index += 1
            continue

        if TITLE_PATTERN.match(line):
            seen_regulation_body = True
            base_title = line
            current_title = line
            title_buffer = [line]
            if index + 1 < len(lines):
                next_line = lines[index + 1]
                if next_line and not TITLE_PATTERN.match(next_line) and not ARTICLE_HEADER_PATTERN.match(next_line):
                    title_buffer.append(next_line)
                    base_title = f"{line} | {next_line}"
                    current_title = base_title
                    index += 1
            index += 1
            continue

        if not seen_regulation_body:
            preamble_buffer.append(line)
            index += 1
            continue

        article_match = ARTICLE_HEADER_PATTERN.match(line)
        if article_match:
            article_label = (article_match.group(1) or article_match.group(2)).upper()
            article_lines = [] if article_match.group(2) else [line]
            index += 1
            while index < len(lines):
                next_line = lines[index]
                if (
                    TITLE_PATTERN.match(next_line)
                    or ARTICLE_HEADER_PATTERN.match(next_line)
                    or AMENDMENT_ACTION_PATTERN.match(next_line)
                ):
                    break
                if next_line:
                    article_lines.append(next_line)
                index += 1

            if not article_lines:
                continue
            article_text = "\n".join(article_lines)
            status_section = article_status.get(article_label, {}).get("section_title", "")
            article_prefix = [status_section] if status_section else title_buffer
            prefixed_text = article_text if not article_prefix else "\n".join(article_prefix + [article_text])
            article_chunks = _split_large_legal_block(
                prefixed_text,
                document_id=document_id,
                source_title=source_title,
                section_title=current_title,
                article_label=article_label,
                chunk_number_start=chunk_number,
                chunk_size=chunk_size,
                overlap=overlap,
                article_status=article_status,
            )
            chunks.extend(article_chunks)
            chunk_number += len(article_chunks)
            continue

        if _looks_like_subsection_heading(line):
            current_title = f"{base_title} | {line}" if base_title else line
            title_buffer = [current_title]
            index += 1
            continue

        # Non-empty lines inside the regulation body that are not article headers become contextual chunks.
        contextual_lines = [line]
        index += 1
        while index < len(lines):
            next_line = lines[index]
            if not next_line:
                index += 1
                if contextual_lines:
                    break
                continue
            if (
                TITLE_PATTERN.match(next_line)
                or ARTICLE_HEADER_PATTERN.match(next_line)
                or AMENDMENT_ACTION_PATTERN.match(next_line)
            ):
                break
            contextual_lines.append(next_line)
            index += 1

        contextual_text = "\n".join(contextual_lines)
        contextual_chunks = _split_large_legal_block(
            contextual_text if not title_buffer else "\n".join(title_buffer + [contextual_text]),
            document_id=document_id,
            source_title=source_title,
            section_title=current_title or "PREAMBULO",
            article_label="",
            chunk_number_start=chunk_number,
            chunk_size=chunk_size,
            overlap=overlap,
            article_status=article_status,
        )
        chunks.extend(contextual_chunks)
        chunk_number += len(contextual_chunks)

    if preamble_buffer:
        preamble_text = "\n".join(preamble_buffer).strip()
        preamble_chunks = _split_large_legal_block(
            preamble_text,
            document_id=document_id,
            source_title=source_title,
            section_title="PREAMBULO",
            article_label="",
            chunk_number_start=chunk_number,
            chunk_size=chunk_size,
            overlap=overlap,
            article_status=article_status,
        )
        chunks.extend(preamble_chunks)

    return chunks


def _looks_like_subsection_heading(line: str) -> bool:
    """Identify uppercase legal subsection headings like DEFINICIONES or CRITERIOS."""

    if not line or TITLE_PATTERN.match(line) or ARTICLE_HEADER_PATTERN.match(line):
        return False
    if len(line) > 120:
        return False
    return SUBSECTION_PATTERN.match(line) is not None


def _split_large_legal_block(
    text: str,
    *,
    document_id: str,
    source_title: str,
    section_title: str,
    article_label: str,
    chunk_number_start: int,
    chunk_size: int,
    overlap: int,
    article_status: dict[str, dict],
) -> list[DocumentChunk]:
    """Split a legal block while preserving article identity."""

    paragraphs = [part.strip() for part in text.split("\n") if part.strip()]
    chunks: list[DocumentChunk] = []
    buffer = ""
    chunk_number = chunk_number_start

    for paragraph in paragraphs:
        candidate = f"{buffer}\n{paragraph}".strip() if buffer else paragraph
        if len(candidate) <= chunk_size:
            buffer = candidate
            continue

        if buffer:
            chunks.append(
                _build_chunk(
                    document_id=document_id,
                    source_title=source_title,
                    text=buffer,
                    section_title=section_title,
                    article_label=article_label,
                    chunk_number=chunk_number,
                    article_status=article_status,
                )
            )
            chunk_number += 1
            carry = buffer[-overlap:] if overlap > 0 else ""
            buffer = f"{carry}\n{paragraph}".strip()
        else:
            start = 0
            while start < len(paragraph):
                end = start + chunk_size
                slice_text = paragraph[start:end].strip()
                chunks.append(
                    _build_chunk(
                        document_id=document_id,
                        source_title=source_title,
                        text=slice_text,
                        section_title=section_title,
                        article_label=article_label,
                        chunk_number=chunk_number,
                        article_status=article_status,
                    )
                )
                chunk_number += 1
                start = max(end - overlap, start + 1)
            buffer = ""

    if buffer:
        chunks.append(
            _build_chunk(
                document_id=document_id,
                source_title=source_title,
                text=buffer,
                section_title=section_title,
                article_label=article_label,
                chunk_number=chunk_number,
                article_status=article_status,
            )
        )

    return chunks


def _build_chunk(
    *,
    document_id: str,
    source_title: str,
    text: str,
    section_title: str,
    article_label: str,
    chunk_number: int,
    article_status: dict[str, dict],
) -> DocumentChunk:
    """Create a chunk with consistent metadata."""

    status = article_status.get(article_label, {})
    resolved_section = status.get("section_title") or section_title
    normalized_text = normalize_for_search(text)
    content_type = _classify_content(
        normalized_text,
        article_label=article_label,
        section_title=normalize_for_search(resolved_section),
    )
    vigencia = status.get("vigencia", "vigente")
    priority = _assign_priority(content_type, vigencia)
    return DocumentChunk(
        chunk_id=f"{document_id}-{chunk_number:03d}",
        document_id=document_id,
        source_title=source_title,
        text=text,
        section_title=resolved_section,
        article_label=article_label,
        normalized_text=normalized_text,
        tipo_contenido=content_type,
        user_intents=_infer_user_intents(content_type, normalized_text),
        vigencia=vigencia,
        modificado_por=status.get("modificado_por", ""),
        prioridad_retrieval=priority,
        metadata={
            "chunk_number": chunk_number,
            "section_title": resolved_section,
            "article_label": article_label,
            "change_type": status.get("change_type", ""),
        },
    )


def _fallback_paragraph_chunks(
    text: str,
    *,
    document_id: str,
    source_title: str,
    chunk_size: int,
    overlap: int,
    article_status: dict[str, dict],
) -> list[DocumentChunk]:
    """Fallback splitter for non-legal or poorly structured text."""

    paragraphs = [part.strip() for part in text.split("\n\n") if part.strip()]
    chunks: list[DocumentChunk] = []
    buffer = ""
    chunk_number = 1

    for paragraph in paragraphs:
        candidate = f"{buffer}\n\n{paragraph}".strip() if buffer else paragraph
        if len(candidate) <= chunk_size:
            buffer = candidate
            continue

        if buffer:
            chunks.append(
                _build_chunk(
                    document_id=document_id,
                    source_title=source_title,
                    text=buffer,
                    section_title="",
                    article_label="",
                    chunk_number=chunk_number,
                    article_status=article_status,
                )
            )
            chunk_number += 1
            carry = buffer[-overlap:] if overlap > 0 else ""
            buffer = f"{carry}\n\n{paragraph}".strip()
        else:
            start = 0
            while start < len(paragraph):
                end = start + chunk_size
                slice_text = paragraph[start:end].strip()
                chunks.append(
                    _build_chunk(
                        document_id=document_id,
                        source_title=source_title,
                        text=slice_text,
                        section_title="",
                        article_label="",
                        chunk_number=chunk_number,
                        article_status=article_status,
                    )
                )
                chunk_number += 1
                start = max(end - overlap, start + 1)
            buffer = ""

    if buffer:
        chunks.append(
            _build_chunk(
                document_id=document_id,
                source_title=source_title,
                text=buffer,
                section_title="",
                article_label="",
                chunk_number=chunk_number,
                article_status=article_status,
            )
        )

    return chunks


def _classify_content(normalized_text: str, *, article_label: str, section_title: str) -> str:
    """Classify document evidence for citizen-oriented retrieval."""

    if not article_label:
        if "considerando" in normalized_text or "visto:" in normalized_text:
            return "considerando"
        if (
            "base legal" in normalized_text
            or "base legal" in section_title
            or "ley organica de municipalidades" in normalized_text
        ):
            return "base_legal"
    if article_label == "2" or "se entiende por" in normalized_text or "definicion" in section_title:
        return "definicion"
    if "requisit" in normalized_text or "presentar una solicitud" in normalized_text:
        return "requisito"
    if (
        ("incumpl" in normalized_text or "no cumpla" in normalized_text)
        and any(term in normalized_text for term in ("retiro", "revoc", "sancion"))
    ):
        return "sancion"
    if "sisa" in normalized_text and any(term in normalized_text for term in ("pago", "monto", "valor", "tributo")):
        return "costo"
    if "zona rigida" in normalized_text or "zonas rigidas" in normalized_text or "zona prohibida" in normalized_text:
        return "zona"
    if "horario" in normalized_text or "06:00" in normalized_text or "23.00" in normalized_text:
        return "horario"
    if "prohib" in normalized_text:
        return "prohibicion"
    if any(term in normalized_text for term in ("sancion", "revocatoria", "decomis", "incauta")):
        return "sancion"
    if "procedimiento" in normalized_text or "tramite" in normalized_text:
        return "procedimiento"
    if "disposiciones finales" in section_title:
        return "disposicion_final"
    if "incorporar" in normalized_text:
        return "incorporacion"
    return "disposicion"


def _infer_user_intents(content_type: str, normalized_text: str) -> list[str]:
    by_type = {
        "requisito": ["consulta_requisitos", "consulta_documentos"],
        "zona": ["consulta_zonas"],
        "horario": ["consulta_horarios"],
        "costo": ["consulta_costos"],
        "sancion": ["consulta_sanciones"],
        "procedimiento": ["consulta_procedimiento"],
        "definicion": ["consulta_ambulatorio"],
        "prohibicion": ["consulta_sanciones"],
    }
    intents = list(by_type.get(content_type, ["ninguno"]))
    if "autoriz" in normalized_text and "consulta_autorizacion" not in intents:
        intents.append("consulta_autorizacion")
    if "alimento" in normalized_text and "consulta_alimentos" not in intents:
        intents.append("consulta_alimentos")
    return intents


def _assign_priority(content_type: str, vigencia: str) -> int:
    if content_type in {"considerando", "base_legal"}:
        return 0
    if vigencia == "vigencia_no_verificable":
        return 0
    if vigencia in {"modificado", "derogado", "historico"}:
        return 1
    if content_type in {"definicion", "disposicion_transitoria", "disposicion_final"}:
        return 2
    if content_type in {
        "requisito",
        "prohibicion",
        "procedimiento",
        "zona",
        "costo",
        "horario",
        "sancion",
    }:
        return 3
    return 2
