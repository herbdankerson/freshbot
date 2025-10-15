"""Scope-specific wrapper flows for the ingestion pipeline."""

from __future__ import annotations

from typing import Dict, Optional

try:  # pragma: no cover - Prefect optional
    from prefect import flow
except Exception:  # pragma: no cover - fallback shim
    def flow(function=None, *_, **__):  # type: ignore
        if function is None:
            def decorator(fn):
                return fn

            return decorator
        return function

try:  # pragma: no cover - optional dependency
    from etl.tasks.intake_models import FlowReport  # type: ignore
except ModuleNotFoundError as exc:  # pragma: no cover - tests without ETL
    from typing import Any

    FlowReport = Any  # type: ignore
    ETL_IMPORT_ERROR = exc
else:
    ETL_IMPORT_ERROR = None

from .pipeline import ingest_pipeline_flow


def _require_etl() -> None:
    if ETL_IMPORT_ERROR is not None:
        raise RuntimeError(
            "Freshbot ingestion flows require the Intellibot ETL package. "
            "Run inside the compose environment so 'etl' is importable."
        ) from ETL_IMPORT_ERROR


def _merge_metadata(extra: Optional[Dict[str, object]], scope: str) -> Dict[str, object]:
    payload = dict(extra or {})
    payload.setdefault("requested_scope", scope)
    return payload


@flow(name="freshbot-ingest-general")
def ingest_general_flow(
    *,
    source_type: str,
    source_uri: str,
    display_name: str,
    content_b64: Optional[str],
    extra_metadata: Optional[Dict[str, object]] = None,
    is_dev: bool = False,
) -> FlowReport:
    """Ingest general-domain artefacts into the shared KB tables."""

    _require_etl()
    metadata = _merge_metadata(extra_metadata, "general")
    return ingest_pipeline_flow(
        source_type=source_type,
        source_uri=source_uri,
        display_name=display_name,
        content_b64=content_b64,
        extra_metadata=metadata,
        target_namespace="kb",
        target_entries="kb.entries",
        is_dev=is_dev,
        classifier_agent=None,
    )


@flow(name="freshbot-ingest-law")
def ingest_law_flow(
    *,
    source_type: str,
    source_uri: str,
    display_name: str,
    content_b64: Optional[str],
    extra_metadata: Optional[Dict[str, object]] = None,
    is_dev: bool = False,
) -> FlowReport:
    """Ingest legal-domain artefacts while marking the scope for downstream tools."""

    _require_etl()
    metadata = _merge_metadata(extra_metadata, "law")
    return ingest_pipeline_flow(
        source_type=source_type,
        source_uri=source_uri,
        display_name=display_name,
        content_b64=content_b64,
        extra_metadata=metadata,
        target_namespace="kb",
        target_entries="kb.entries",
        is_dev=is_dev,
        classifier_agent=None,
    )


@flow(name="freshbot-ingest-code")
def ingest_code_flow(
    *,
    source_type: str,
    source_uri: str,
    display_name: str,
    content_b64: Optional[str],
    extra_metadata: Optional[Dict[str, object]] = None,
    is_dev: bool = False,
) -> FlowReport:
    """Ingest code artefacts and rely on the code-specific chunking path."""

    _require_etl()
    metadata = _merge_metadata(extra_metadata, "code")
    return ingest_pipeline_flow(
        source_type=source_type,
        source_uri=source_uri,
        display_name=display_name,
        content_b64=content_b64,
        extra_metadata=metadata,
        target_namespace="kb",
        target_entries="kb.entries",
        is_dev=is_dev,
        classifier_agent=None,
    )
