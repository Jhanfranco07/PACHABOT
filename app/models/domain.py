from __future__ import annotations

from enum import Enum


class QueryIntent(str, Enum):
    GENERAL = "general"
    DEFINICION = "consulta_definicion"
    REQUISITOS = "requisitos"
    REQUISITOS_NUEVO = "consulta_requisitos_nuevo"
    REQUISITOS_RENOVACION = "consulta_requisitos_renovacion"
    REQUISITOS_AMBIGUO = "consulta_requisitos_ambiguo"
    MODULOS = "modulos"
    PAGOS_SISA = "pagos_sisa"
    ZONAS_RIGIDAS = "zonas_rigidas"
    AUTORIZACIONES = "autorizaciones"
    RUBROS = "rubros"
    FERIAS = "ferias"
    OBLIGACIONES = "obligaciones"
    PROHIBICIONES = "prohibiciones"
    SANCIONES = "sanciones"
    REVOCACION = "revocacion"
    HORARIO = "horario"
    UBICACION = "ubicacion"
    NORMATIVA = "normativa"
    SIN_EVIDENCIA = "sin_evidencia_suficiente"
    OUT_OF_SCOPE = "out_of_scope"


class AssistantMode(str, Enum):
    GENERAL = "general"
    COMMERCE = "commerce"
