"""Freshbot Prefect flows for knowledge base ingestion."""

from __future__ import annotations

import base64
from typing import Dict, Optional
try:  # pragma: no cover - Prefect optional for CLI usage
    from prefect import flow, get_run_logger
except Exception:  # pragma: no cover - fallback for local tooling
    import logging

    def flow(function=None, *_, **__):  # type: ignore
        if function is None:
            def decorator(fn):
                return fn

            return decorator
        return function

    def get_run_logger():
        return logging.getLogger(__name__)

try:
    from etl.flows.flow_document_intake import document_intake_flow
    from etl.tasks.intake_models import FlowReport, IngestItem
except Exception:
    class FlowReport:
        """Placeholder `FlowReport` for unit tests."""

        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class IngestItem:
        """Placeholder `IngestItem` for unit tests."""

        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    def _missing_document_intake_flow(*_, **__):
        raise RuntimeError(
            "document_intake_flow is unavailable. Run inside the compose container "
            "or install the Intellibot ETL package."
        )

    class _DocFlowProxy:
        def with_options(self, **__):
            return _missing_document_intake_flow

        def __call__(self, *args, **kwargs):
            return _missing_document_intake_flow(*args, **kwargs)

    document_intake_flow = _DocFlowProxy()

from ..connectors.catalog import lookup as list_connectors


@flow(name="freshbot_document_ingest")
def freshbot_document_ingest(
    *,
    source_type: str,
    source_uri: str,
    display_name: str,
    content_b64: Optional[str] = None,
    extra_metadata: Optional[Dict[str, object]] = None,
    target_namespace: str = "kb",
    target_entries: Optional[str] = None,
) -> Dict[str, object]:
    """Ingest a document into the knowledge base using the Freshbot pipeline."""

    logger = get_run_logger()
    connectors = {entry["alias"]: entry for entry in list_connectors()}
    logger.info("Active connectors: %s", sorted(connectors))

    payload: Optional[bytes] = None
    if content_b64:
        payload = base64.b64decode(content_b64)
        logger.info("Decoded %s bytes of payload for %s", len(payload), display_name)

    logger.info("Delegating to legacy document intake flow")
    subflow = document_intake_flow.with_options(name="legacy_document_intake")
    report: FlowReport = subflow(
        source_type=source_type,
        source_uri=source_uri,
        display_name=display_name,
        content=payload,
        extra_metadata=extra_metadata,
        target_namespace=target_namespace,
        target_entries=target_entries,
    )

    ingest_item: IngestItem = report.ingest_item
    result = {
        "ingest_item_id": str(ingest_item.id),
        "job_id": report.job_id or (str(ingest_item.job_id) if ingest_item.job_id else None),
        "status": ingest_item.status,
        "source_uri": ingest_item.source_uri,
        "chunk_count": report.chunk_count,
        "embedding_spaces": report.embedding_spaces,
        "connectors": sorted(connectors),
    }

    logger.info(
        "Ingestion completed for %s (item=%s, chunks=%s)",
        ingest_item.display_name,
        result["ingest_item_id"],
        report.chunk_count,
    )
    return result


__all__ = ["freshbot_document_ingest"]
