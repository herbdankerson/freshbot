"""Freshbot Prefect flows for knowledge base ingestion."""

from __future__ import annotations

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

from ..connectors.catalog import lookup as list_connectors
from .ingest.pipeline import ingest_pipeline_flow


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
    is_dev: bool = False,
) -> Dict[str, object]:
    """Ingest a document into the knowledge base using the Freshbot pipeline."""

    logger = get_run_logger()
    connectors = {entry["alias"]: entry for entry in list_connectors()}
    logger.info("Active connectors: %s", sorted(connectors))

    report: FlowReport = ingest_pipeline_flow(
        source_type=source_type,
        source_uri=source_uri,
        display_name=display_name,
        content_b64=content_b64,
        extra_metadata=extra_metadata,
        target_namespace=target_namespace,
        target_entries=target_entries,
        is_dev=is_dev,
        classifier_agent=None,
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
