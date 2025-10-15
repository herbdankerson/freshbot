"""High-level ingestion flows that assemble the step-level building blocks."""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Dict, Optional

try:  # pragma: no cover - Prefect optional in some environments
    from prefect import flow, get_run_logger
    try:
        from prefect.runtime import flow_run as prefect_flow_run  # type: ignore
    except Exception:  # pragma: no cover - runtime helper optional
        prefect_flow_run = None  # type: ignore
except Exception:  # pragma: no cover - fallback shims
    import logging

    def flow(function=None, *_, **__):  # type: ignore
        if function is None:
            def decorator(fn):
                return fn

            return decorator
        return function

    def get_run_logger():
        return logging.getLogger(__name__)
    prefect_flow_run = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from etl.tasks.intake_models import FlowReport  # type: ignore
except ModuleNotFoundError as exc:  # pragma: no cover - tests without ETL
    from typing import Any

    FlowReport = Any  # type: ignore
    ETL_IMPORT_ERROR = exc
else:
    ETL_IMPORT_ERROR = None

from . import steps
from .audit import begin_flow_run, complete_flow_run
from ...pipeline.classification import is_code_extension


def _require_etl() -> None:
    if getattr(steps, "intake_tasks", None) is None:
        raise RuntimeError(
            "Freshbot ingestion flows require the Intellibot ETL package. "
            "Run inside the compose environment so 'etl' is importable."
        ) from ETL_IMPORT_ERROR


def _decode_content(content_b64: Optional[str]) -> Optional[bytes]:
    if not content_b64:
        return None
    return base64.b64decode(content_b64)


@flow(name="freshbot-ingest-pipeline")
def ingest_pipeline_flow(
    *,
    source_type: str,
    source_uri: str,
    display_name: str,
    content_b64: Optional[str],
    extra_metadata: Optional[Dict[str, object]],
    target_namespace: str,
    target_entries: Optional[str],
    is_dev: bool,
    classifier_agent: Optional[str] = None,
) -> FlowReport:
    """Complete ingestion pipeline that writes data into the ParadeDB KB."""

    logger = get_run_logger()
    _require_etl()
    metadata = dict(extra_metadata or {})
    metadata["ingest_namespace"] = target_namespace
    item = steps.register_ingest_item_flow(
        source_type=source_type,
        source_uri=source_uri,
        display_name=display_name,
        metadata=metadata,
        is_dev=is_dev,
    )

    prefect_run_id = None
    if prefect_flow_run is not None:  # pragma: no branch - guard for runtime deps
        try:
            prefect_run_id = prefect_flow_run.get_id()
        except Exception:
            prefect_run_id = None

    flow_run_record_id = begin_flow_run(
        ingest_item_id=item.id,
        flow_name="freshbot-ingest-pipeline",
        is_dev=is_dev,
        parameters={
            "source_type": source_type,
            "source_uri": source_uri,
            "display_name": display_name,
            "target_namespace": target_namespace,
            "target_entries": target_entries,
        },
        prefect_run_id=prefect_run_id,
    )

    try:
        payload_bytes = _decode_content(content_b64)
        source = steps.acquire_source_flow(item=item, content=payload_bytes)

        filename = str(source.metadata.get("filename") or display_name)
        source_path = Path(source_uri)
        if not source_path.exists():
            path_hint = source.metadata.get("path") or source.metadata.get("resolved_path")
            if isinstance(path_hint, str):
                source_path = Path(path_hint)

        if source_path.exists():
            logger.debug("Resolved source path for %s -> %s", display_name, source_path)

        if is_code_extension(filename):
            normalized = steps.code_normalize_flow(source=source)
            chunks = steps.code_chunk_flow(
                item=item,
                document=normalized,
                source_path=source_path,
            )
        else:
            normalized = steps.docling_normalize_flow(source=source)
            chunks = steps.doc_chunk_flow(item=item, document=normalized)

        item = steps.summarize_document_flow(item=item, document=normalized)
        item = steps.classify_domain_flow(
            item=item,
            document=normalized,
            chunks=chunks,
            agent=classifier_agent,
        )

        domain_key = (item.domain or "").lower()
        if domain_key == "code":
            logger.info("Skipping emotion detection for code artefact %s", item.id)
            chunk_emotions = {}
        else:
            chunk_emotions = steps.detect_emotions_flow(item=item, chunks=chunks)
        abstractions = steps.build_abstractions_flow(chunks=chunks)
        embeddings = steps.embed_chunks_flow(item=item, chunks=chunks)

        table_config = steps.intake_tasks.table_config_from_namespace(
            target_namespace,
            entries=target_entries,
        )

        report = steps.persist_results_flow(
            item=item,
            document=normalized,
            chunks=chunks,
            embeddings=embeddings,
            chunk_emotions=chunk_emotions,
            abstractions=abstractions,
            table_config=table_config,
        )
    except Exception as exc:
        complete_flow_run(
            flow_run_record_id,
            status="error",
            error=str(exc),
        )
        raise

    complete_flow_run(
        flow_run_record_id,
        status="success",
        result={
            "ingest_item_id": str(report.ingest_item.id) if getattr(report, "ingest_item", None) else str(item.id),
            "chunk_count": getattr(report, "chunk_count", len(chunks)),
            "embedding_spaces": getattr(report, "embedding_spaces", []),
        },
    )
    return report
