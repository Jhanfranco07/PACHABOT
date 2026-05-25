from __future__ import annotations

import logging
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from app.utils.helpers import ensure_directory, write_json


ARTICLE_HEADER_PATTERN = re.compile(
    r"^(?:Artículo|ARTÍCULO|ARTICULO)\s+([0-9]+(?:\.[0-9]+)?[A-Z]?)\s*[°º]?\s*"
    r"(?:\.\s*-?|-|:|[\"“])",
    re.MULTILINE,
)
TITLE_PATTERN = re.compile(r"^(?:TÍTULO|TITULO)\s+[IVXLC0-9]+\b.*$", re.MULTILINE)
ACTION_HEADER_PATTERN = re.compile(
    r"^(?:ARTÍCULO|ARTICULO)\s+"
    r"(VIGÉSIMO PRIMERO|VIGESIMO PRIMERO|DECIMO PRIMERO|DÉCIMO PRIMERO|"
    r"DECIMO SEGUNDO|DÉCIMO SEGUNDO|DECIMO TERCERO|DÉCIMO TERCERO|DECIMO CUARTO|"
    r"DÉCIMO CUARTO|DECIMO QUINTO|DÉCIMO QUINTO|DECIMO SEXTO|DÉCIMO SEXTO|"
    r"DECIMO SÉPTIMO|DÉCIMO SÉPTIMO|DECIMO SEPTIMO|DECIMO OCTAVO|"
    r"DÉCIMO OCTAVO|DECIMO NOVENO|DÉCIMO NOVENO|PRIMERO|SEGUNDO|TERCERO|"
    r"CUARTO|QUINTO|SEXTO|SÉPTIMO|SEPTIMO|OCTAVO|NOVENO|DÉCIMO|DECIMO|"
    r"VIGÉSIMO|VIGESIMO)\b",
    re.MULTILINE,
)


@dataclass(frozen=True, slots=True)
class ModificationRule:
    target_article: str
    modifying_provision: str
    change_type: str
    source_document: str = "ordenanza_227_2019"
    source_title: str = "Ordenanza 227-2019-MDP/C"
    verification: str = "verificado_en_fuente_disponible"


# CAMBIO FASE 7.4 — Tabla explicita de modificaciones normativas declaradas.
# Motivo: la vigencia no puede depender del parecido semantico entre documentos.
# Riesgo mitigado: solo se aplican cambios expresamente enumerados por la modificatoria.
MODIFICATION_RULES: tuple[ModificationRule, ...] = (
    ModificationRule("TITULO I", "Articulo Primero", "REEMPLAZADO"),
    ModificationRule("2", "Articulo Segundo", "REEMPLAZADO"),
    ModificationRule("5", "Articulo Tercero", "REEMPLAZADO"),
    ModificationRule("6", "Articulo Cuarto", "REEMPLAZADO"),
    ModificationRule("7", "Articulo Quinto", "REEMPLAZADO"),
    ModificationRule("7A", "Articulo Sexto", "INCORPORADO"),
    ModificationRule("8A", "Articulo Septimo", "INCORPORADO"),
    ModificationRule("16", "Articulo Octavo", "REEMPLAZADO"),
    ModificationRule("17.4", "Articulo Noveno", "REEMPLAZADO"),
    ModificationRule("21", "Articulo Decimo", "REEMPLAZADO"),
    ModificationRule("23", "Articulo Decimo Primero", "REEMPLAZADO"),
    ModificationRule("30", "Articulo Decimo Segundo", "REEMPLAZADO"),
    ModificationRule("36", "Articulo Decimo Tercero", "REEMPLAZADO"),
    ModificationRule("38", "Articulo Decimo Cuarto", "REEMPLAZADO"),
    ModificationRule("41", "Articulo Decimo Quinto", "REEMPLAZADO"),
    ModificationRule("42", "Articulo Decimo Sexto", "REEMPLAZADO"),
    ModificationRule("43", "Articulo Decimo Septimo", "REEMPLAZADO"),
    ModificationRule("50", "Articulo Decimo Octavo", "AMPLIADO"),
    ModificationRule("52", "Articulo Decimo Noveno", "REEMPLAZADO"),
    ModificationRule("54", "Articulo Vigesimo", "REEMPLAZADO"),
    ModificationRule(
        "57",
        "Articulo Vigesimo Primero",
        "INCORPORADO",
        verification="vigencia_no_verificable_texto_truncado",
    ),
)


SECTION_OVERRIDES: dict[str, str] = {
    "2": "TITULO I | DEFINICIONES",
    "5": "TITULO II | AUTORIZACION MUNICIPAL",
    "6": "TITULO III | AUTORIZACION MUNICIPAL",
    "7": "TITULO III | AUTORIZACION MUNICIPAL",
    "7A": "TITULO III | AUTORIZACION MUNICIPAL",
    "8A": "TITULO III | AUTORIZACION MUNICIPAL",
    "16": "TITULO IV | REGULACION DEL COMERCIO EN VIA PUBLICA",
    "17": "TITULO IV | ZONAS Y MODULOS",
    "17.4": "TITULO IV | ZONAS RIGIDAS",
    "21": "TITULO V | ACTIVIDADES PERMITIDAS",
    "23": "TITULO V | ALIMENTOS",
    "30": "TITULO VI | REQUISITOS Y TRAMITES",
    "36": "TITULO VI | PAGO SISA",
    "38": "TITULO VI | INCUMPLIMIENTO DE PAGO SISA",
    "41": "TITULO VII | FISCALIZACION",
    "42": "TITULO VII | FISCALIZACION",
    "43": "TITULO VII | SANCIONES",
    "50": "TITULO VIII | REVOCATORIA",
    "52": "TITULO IX | FERIAS",
    "54": "TITULO IX | REQUISITOS DE FERIA",
    "57": "TITULO IX | OBLIGACIONES Y PROHIBICIONES",
}


class LegalConsolidator:
    """Generate a traceable current-law corpus from an original and its amendment."""

    def __init__(self, logger: logging.Logger) -> None:
        self.logger = logger.getChild("legal_consolidator")

    def consolidate(
        self,
        cleaned_documents: dict[str, str],
        *,
        output_dir: Path,
    ) -> dict[str, Any]:
        """Write the modification map, consolidated articles and validation report."""

        ensure_directory(output_dir)
        base_text = cleaned_documents.get("ordenanza_108_2012", "")
        amendment_text = cleaned_documents.get("ordenanza_227_2019", "")
        base_articles = _extract_base_articles(base_text)
        amendment_articles = _extract_amendment_articles(amendment_text)
        consolidated, unverified = _apply_modifications(base_articles, amendment_articles)

        report = {
            "corpus_complete": False,
            "warnings": [
                (
                    "La fuente disponible de la Ordenanza 227-2019-MDP/C termina "
                    "truncada durante el Articulo 57."
                ),
                (
                    "No se presume el contenido faltante ni la existencia de "
                    "disposiciones posteriores no presentes en la fuente."
                ),
            ],
            "unverified_scope": [
                "Articulo 57 de la Ordenanza 227-2019-MDP/C (texto incompleto)",
                "Cualquier disposicion posterior al punto de truncamiento",
            ],
        }
        payload = {
            "corpus_id": "comercio_ambulatorio_pachacamac_consolidado",
            "source_documents": [
                "Ordenanza 108-2012-MDP/C",
                "Ordenanza 227-2019-MDP/C",
            ],
            "articles": consolidated,
            "unverified_articles": unverified,
            "validation": report,
        }
        write_json(output_dir / "modification_map.json", [asdict(rule) for rule in MODIFICATION_RULES])
        write_json(output_dir / "norma_consolidada.json", payload)
        write_json(output_dir / "corpus_validation_report.json", report)
        self.logger.warning(
            "Corpus consolidado generado con advertencia: la Ordenanza 227-2019 esta truncada en el Articulo 57."
        )
        return payload


def article_status_map(document_id: str) -> dict[str, dict[str, Any]]:
    """Return legal status and section overrides used during source chunking."""

    map_: dict[str, dict[str, Any]] = {}
    for rule in MODIFICATION_RULES:
        if rule.target_article == "TITULO I":
            continue
        label = rule.target_article
        target_label = "17" if label == "17.4" and document_id == "ordenanza_108_2012" else label
        if document_id == "ordenanza_108_2012":
            map_[target_label] = {
                "vigencia": "historico",
                "modificado_por": f"Ordenanza 227-2019-MDP/C, {rule.modifying_provision}",
                "section_title": SECTION_OVERRIDES.get(target_label, ""),
            }
        elif document_id == "ordenanza_227_2019":
            map_[label] = {
                "vigencia": (
                    "vigencia_no_verificable"
                    if rule.verification.startswith("vigencia_no_verificable")
                    else "vigente"
                ),
                "modificado_por": "",
                "section_title": SECTION_OVERRIDES.get(label, ""),
                "change_type": rule.change_type,
            }
    return map_


def _extract_base_articles(text: str) -> dict[str, dict[str, Any]]:
    if not text:
        return {}
    body_start = text.find("REGLAMENTO DEL COMERCIO AMBULATORIO")
    body = text[body_start:] if body_start >= 0 else text
    matches = list(ARTICLE_HEADER_PATTERN.finditer(body))
    articles: dict[str, dict[str, Any]] = {}
    for index, match in enumerate(matches):
        label = match.group(1).upper()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(body)
        content = body[match.start():end].strip()
        articles[label] = {
            "article_label": label,
            "source_title": "Ordenanza 108-2012-MDP/C",
            "document_id": "ordenanza_108_2012",
            "section_title": _section_before(body, match.start()),
            "text": content,
            "vigencia": "vigente",
            "modificado_por": "",
            "source_trace": ["Ordenanza 108-2012-MDP/C"],
        }
    return articles


def _extract_amendment_articles(text: str) -> dict[str, dict[str, Any]]:
    if not text:
        return {}
    starts = list(ACTION_HEADER_PATTERN.finditer(text))
    by_provision = {
        _ascii_key(rule.modifying_provision.replace("Articulo ", "")): rule
        for rule in MODIFICATION_RULES
    }
    output: dict[str, dict[str, Any]] = {}
    for index, match in enumerate(starts):
        provision_key = _ascii_key(match.group(1))
        rule = by_provision.get(provision_key)
        if rule is None or rule.target_article == "TITULO I":
            continue
        end = starts[index + 1].start() if index + 1 < len(starts) else len(text)
        block = text[match.start():end].strip()
        payload = _amendment_payload(block, rule.target_article)
        if not payload:
            continue
        vigencia = (
            "vigencia_no_verificable"
            if rule.verification.startswith("vigencia_no_verificable")
            else "vigente"
        )
        output[rule.target_article] = {
            "article_label": rule.target_article,
            "source_title": rule.source_title,
            "document_id": rule.source_document,
            "section_title": SECTION_OVERRIDES.get(rule.target_article, ""),
            "text": payload,
            "vigencia": vigencia,
            "modificado_por": "",
            "change_type": rule.change_type,
            "modifies": f"Ordenanza 108-2012-MDP/C, Articulo {rule.target_article}",
            "source_trace": [f"{rule.source_title}, {rule.modifying_provision}"],
        }
    return output


def _apply_modifications(
    base: dict[str, dict[str, Any]],
    amendments: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    consolidated = dict(base)
    unverified: list[dict[str, Any]] = []
    for rule in MODIFICATION_RULES:
        label = rule.target_article
        if label == "TITULO I":
            continue
        amendment = amendments.get(label)
        if amendment is None:
            continue
        if amendment["vigencia"] == "vigencia_no_verificable":
            unverified.append(amendment)
            continue
        if label == "17.4":
            consolidated["17"] = _merge_partial_article(consolidated.get("17"), amendment)
            consolidated[label] = amendment
        elif rule.change_type == "AMPLIADO":
            consolidated[label] = _merge_expanded_article(consolidated.get(label), amendment)
        else:
            consolidated[label] = amendment
    verified = [
        item for _, item in sorted(consolidated.items(), key=lambda value: _sort_label(value[0]))
        if item.get("vigencia") == "vigente"
    ]
    return verified, unverified


def _merge_partial_article(base: dict[str, Any] | None, amendment: dict[str, Any]) -> dict[str, Any]:
    if base is None:
        return amendment
    text = base["text"]
    text = re.sub(r"(?ms)^17\.4\..*$", amendment["text"], text).strip()
    merged = dict(base)
    merged.update(
        {
            "text": text,
            "source_title": "Ordenanza 108-2012-MDP/C modificada por Ordenanza 227-2019-MDP/C",
            "source_trace": base["source_trace"] + amendment["source_trace"],
            "modificado_por": "Ordenanza 227-2019-MDP/C, Articulo Noveno",
        }
    )
    return merged


def _merge_expanded_article(base: dict[str, Any] | None, amendment: dict[str, Any]) -> dict[str, Any]:
    if base is None:
        return amendment
    merged = dict(base)
    merged.update(
        {
            "text": f"{base['text']}\n\n{amendment['text']}".strip(),
            "source_title": "Ordenanza 108-2012-MDP/C modificada por Ordenanza 227-2019-MDP/C",
            "source_trace": base["source_trace"] + amendment["source_trace"],
            "modificado_por": "Ordenanza 227-2019-MDP/C, Articulo Decimo Octavo",
        }
    )
    return merged


def _amendment_payload(block: str, label: str) -> str:
    lines = block.splitlines()
    retained: list[str] = []
    target_wrapper = re.compile(
        rf"^art[ií]culo\s+{re.escape(label)}[°º]?\s+de\s+la\s+Ordenanza",
        re.IGNORECASE,
    )
    for index, line in enumerate(lines):
        compact = line.strip()
        if index == 0 or not compact or target_wrapper.match(compact):
            continue
        retained.append(compact)
    return "\n".join(retained).strip()


def _section_before(text: str, position: int) -> str:
    matches = list(TITLE_PATTERN.finditer(text[:position]))
    return matches[-1].group(0).strip() if matches else ""


def _ascii_key(value: str) -> str:
    replacements = str.maketrans("ÁÉÍÓÚáéíóú", "AEIOUaeiou")
    return value.translate(replacements).upper().strip()


def _sort_label(label: str) -> tuple[int, int, str]:
    match = re.match(r"([0-9]+)(?:\.([0-9]+))?([A-Z]?)", label)
    if not match:
        return (9999, 0, label)
    return (int(match.group(1)), int(match.group(2) or 0), match.group(3) or "")
