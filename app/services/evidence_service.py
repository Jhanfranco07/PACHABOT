from __future__ import annotations

import json
import re
from dataclasses import asdict
from datetime import datetime, timezone

from app.channels.schemas import IncomingChatMessage
from app.config import Settings
from app.models.schemas import (
    AnswerPayload,
    EvidenceAssessment,
    EvidenceItem,
    KnowledgeBundle,
    RoutedQuery,
)
from app.utils.helpers import ensure_directory


SOURCE_TYPES = {
    "tramites": "tramite",
    "faq": "faq",
    "chunks": "chunk_documental",
    "consolidated": "norma_consolidada",
    "normativa": "ordenanza",
}


class EvidenceService:
    """Evaluate retrieved support and optionally persist a compact RAG trace."""

    def __init__(self, settings: Settings, logger) -> None:
        self.settings = settings
        self.logger = logger.getChild("evidence_service")

    def assess(self, knowledge: KnowledgeBundle) -> EvidenceAssessment:
        items = [
            EvidenceItem(
                source=chunk.fuente or chunk.source_title,
                source_type=SOURCE_TYPES.get(chunk.knowledge_layer, chunk.knowledge_layer),
                score=round(chunk.score, 4),
                excerpt=self._excerpt(chunk.text),
                article_label=chunk.article_label,
                requires_review=chunk.requires_review,
            )
            for chunk in knowledge.chunks[: self.settings.retrieval_max_results]
        ]
        usable = [
            chunk
            for chunk in knowledge.chunks
            if not chunk.exclude_from_retrieval
            and chunk.vigencia not in {"no_verificable", "vigencia_no_verificable"}
        ]
        top_score = usable[0].score if usable else 0.0
        sufficient = bool(usable) and top_score >= self.settings.retrieval_min_score

        if not sufficient:
            return EvidenceAssessment(
                sufficient=False,
                confidence_level="low",
                warning=(
                    "No se encontro informacion suficiente en la base documental "
                    "cargada para responder con seguridad."
                ),
                items=items,
            )

        top_layer = usable[0].knowledge_layer
        if top_layer in {"tramites", "faq"} or top_score >= self.settings.retrieval_min_score + 0.25:
            confidence_level = "high"
        else:
            confidence_level = "medium"

        warning = ""
        if any(chunk.requires_review for chunk in usable):
            warning = "La evidencia utilizada tiene observaciones pendientes de validacion humana."
        return EvidenceAssessment(
            sufficient=True,
            confidence_level=confidence_level,
            warning=warning,
            items=items,
        )

    def write_trace(
        self,
        message: IncomingChatMessage,
        routed: RoutedQuery,
        knowledge: KnowledgeBundle,
        assessment: EvidenceAssessment,
        payload: AnswerPayload,
    ) -> None:
        if not self.settings.rag_debug_trace:
            return

        ensure_directory(self.settings.runtime_debug_dir)
        session = re.sub(r"[^A-Za-z0-9_.-]+", "_", f"{message.channel}_{message.session_id}")[:120]
        trace_path = self.settings.runtime_debug_dir / f"{session}.jsonl"
        trace = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "channel": message.channel,
            "session_id": message.session_id,
            "question": knowledge.original_query,
            "effective_query": knowledge.effective_query,
            "intent": routed.intent.value,
            "search_queries": knowledge.search_queries,
            "sufficient": assessment.sufficient,
            "confidence_level": assessment.confidence_level,
            "confidence_score": payload.confidence,
            "warning": assessment.warning,
            "used_llm": payload.used_llm,
            "response_origin": payload.response_origin,
            "evidence": [asdict(item) for item in assessment.items],
        }
        with trace_path.open("a", encoding="utf-8") as output:
            output.write(json.dumps(trace, ensure_ascii=False) + "\n")

    @staticmethod
    def _excerpt(text: str, limit: int = 320) -> str:
        collapsed = re.sub(r"\s+", " ", text).strip()
        return collapsed if len(collapsed) <= limit else f"{collapsed[: limit - 3]}..."
