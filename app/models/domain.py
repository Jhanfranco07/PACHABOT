from __future__ import annotations

from enum import Enum


class QueryIntent(str, Enum):
    GENERAL = "general"
    REQUISITOS = "requisitos"
    MODULOS = "modulos"
    PAGOS_SISA = "pagos_sisa"
    ZONAS_RIGIDAS = "zonas_rigidas"
    AUTORIZACIONES = "autorizaciones"
    FERIAS = "ferias"
    PROHIBICIONES = "prohibiciones"
    OUT_OF_SCOPE = "out_of_scope"


class AssistantMode(str, Enum):
    GENERAL = "general"
    COMMERCE = "commerce"
