from __future__ import annotations

import re

from app.utils.text_cleaner import clean_text


REPEATED_DECORATIVE_LINES = (
    re.compile(r"^MUNICIPALIDAD DISTRITAL DE PACHAC[AÁ]MAC$", re.IGNORECASE),
    re.compile(r"^DISTRITO TUR[IÍ]STICO$", re.IGNORECASE),
    re.compile(r"^[\"“]?A[NÑ]O DE .+[\"”]?$", re.IGNORECASE),
)
PAGE_NUMBER_PATTERNS = (
    re.compile(r"^\s*p[aá]gina\s+\d+\s*(?:de\s+\d+)?\s*$", re.IGNORECASE),
    re.compile(r"^\s*-\s*\d+\s*-\s*$"),
    re.compile(r"^\s*\d+\s*/\s*\d+\s*$"),
)
STRUCTURAL_START = re.compile(
    r"^(?:T[IÍ]TULO\b|ART[IÍ]CULO\b|Art[ií]culo\b|CONSIDERANDO\b|VISTO\b|"
    r"ORDENANZA\b|[A-Z]\)|\d+(?:\.\d+)?[.)])",
    re.IGNORECASE,
)


def clean_legal_document(text: str) -> str:
    """Clean a legal extraction while preserving normative article boundaries."""

    # CAMBIO FASE 7.1 — Limpiar fuentes antes de parsear articulos.
    # Motivo: evitar que encabezados y paginas contaminen la evidencia ciudadana.
    # Riesgo mitigado: solo se eliminan patrones decorativos inequívocos.
    cleaned = clean_text(text).replace("\ufeff", "")
    lines = [line.strip() for line in cleaned.splitlines()]
    filtered: list[str] = []
    decorative_seen: set[str] = set()

    for line in lines:
        if any(pattern.match(line) for pattern in PAGE_NUMBER_PATTERNS):
            continue
        if any(pattern.match(line) for pattern in REPEATED_DECORATIVE_LINES):
            key = line.casefold()
            if key in decorative_seen:
                continue
            decorative_seen.add(key)
        filtered.append(line)

    joined = _join_broken_prose_lines(filtered)
    joined = re.sub(r"\n{3,}", "\n\n", joined)
    return joined.strip()


def _join_broken_prose_lines(lines: list[str]) -> str:
    """Join obvious OCR continuations without joining legal headings or list items."""

    output: list[str] = []
    for line in lines:
        if not line:
            if output and output[-1] != "":
                output.append("")
            continue
        if not output or output[-1] == "" or STRUCTURAL_START.match(line):
            output.append(line)
            continue

        previous = output[-1]
        if (
            previous.endswith((",", ";", ":", "-"))
            and not STRUCTURAL_START.match(previous)
            and not STRUCTURAL_START.match(line)
            and not line.isupper()
        ):
            output[-1] = f"{previous} {line}".strip()
        else:
            output.append(line)
    return "\n".join(output)
