"""Stub Prefect flows for entry metadata handling."""

from __future__ import annotations

from typing import Any, Mapping, Optional

try:  # pragma: no cover
    from prefect import flow, get_run_logger
except ModuleNotFoundError:  # pragma: no cover - fallback for local tooling
    import logging

    def flow(function=None, *_, **__):  # type: ignore
        if function is None:
            def decorator(fn):
                return fn

            return decorator
        return function

    def get_run_logger():
        return logging.getLogger(__name__)


@flow(name="freshbot_metadata_flag", persist_result=True)
def metadata_flag_flow(
    *,
    flag: str,
    entry_id: str,
    ingest_item_id: Optional[str] = None,
    context: Optional[Mapping[str, Any]] = None,
) -> Mapping[str, Any]:
    """Placeholder flow triggered when ingestion raises a metadata flag."""

    logger = get_run_logger()
    logger.info(
        "Metadata flag flow invoked",
        extra={
            "flag": flag,
            "entry_id": entry_id,
            "ingest_item_id": ingest_item_id,
            "context": dict(context or {}),
        },
    )
    return {
        "status": "stubbed",
        "flag": flag,
        "entry_id": entry_id,
        "ingest_item_id": ingest_item_id,
        "context": dict(context or {}),
    }


__all__ = ["metadata_flag_flow"]
