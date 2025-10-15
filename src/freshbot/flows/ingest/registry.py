"""Registry describing available ingestion flows and their import paths."""

from __future__ import annotations

FLOW_REGISTRY = {
    "freshbot-register-ingest-item": "freshbot.flows.ingest.steps.register_ingest_item_flow",
    "freshbot-acquire-source": "freshbot.flows.ingest.steps.acquire_source_flow",
    "freshbot-docling-normalize": "freshbot.flows.ingest.steps.docling_normalize_flow",
    "freshbot-code-normalize": "freshbot.flows.ingest.steps.code_normalize_flow",
    "freshbot-doc-chunk": "freshbot.flows.ingest.steps.doc_chunk_flow",
    "freshbot-code-chunk": "freshbot.flows.ingest.steps.code_chunk_flow",
    "freshbot-summarize-document": "freshbot.flows.ingest.steps.summarize_document_flow",
    "freshbot-detect-emotions": "freshbot.flows.ingest.steps.detect_emotions_flow",
    "freshbot-build-abstractions": "freshbot.flows.ingest.steps.build_abstractions_flow",
    "freshbot-embed-chunks": "freshbot.flows.ingest.steps.embed_chunks_flow",
    "freshbot-classify-domain": "freshbot.flows.ingest.steps.classify_domain_flow",
    "freshbot-persist-ingest": "freshbot.flows.ingest.steps.persist_results_flow",
    "freshbot-ingest-pipeline": "freshbot.flows.ingest.pipeline.ingest_pipeline_flow",
    "freshbot-ingest-general": "freshbot.flows.ingest.wrappers.ingest_general_flow",
    "freshbot-ingest-law": "freshbot.flows.ingest.wrappers.ingest_law_flow",
    "freshbot-ingest-code": "freshbot.flows.ingest.wrappers.ingest_code_flow",
    "task-parse-document": "freshbot.flows.tasks.sync.parse_task_document_flow",
    "task-sync-document": "freshbot.flows.tasks.sync.sync_task_document_flow",
    "task-sync-from-kb": "freshbot.flows.tasks.sync.sync_task_document_from_kb_flow",
}

__all__ = ["FLOW_REGISTRY"]
