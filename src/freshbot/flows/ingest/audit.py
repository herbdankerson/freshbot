"""Helpers to persist ingestion flow telemetry to ParadeDB."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, Optional
from uuid import UUID

from sqlalchemy import text

try:  # pragma: no cover - optional dependency available inside compose
    from src.my_agentic_chatbot.storage.connection import get_engine
except ModuleNotFoundError as exc:  # pragma: no cover - guard for test environments
    get_engine = None  # type: ignore
    ENGINE_IMPORT_ERROR = exc
else:  # pragma: no cover
    ENGINE_IMPORT_ERROR = None


def _require_engine() -> None:
    if get_engine is None:  # pragma: no cover - tests without storage package
        raise RuntimeError(
            "Database engine is unavailable; run inside the compose environment."
        ) from ENGINE_IMPORT_ERROR


def begin_flow_run(
    *,
    ingest_item_id: UUID,
    flow_name: str,
    is_dev: bool,
    parameters: Optional[Dict[str, Any]] = None,
    prefect_run_id: Optional[UUID] = None,
) -> UUID:
    """Insert a telemetry record for a newly-started ingestion flow."""

    _require_engine()
    engine = get_engine()
    payload = {
        "ingest_item_id": str(ingest_item_id),
        "flow_name": flow_name,
        "prefect_run_id": str(prefect_run_id) if prefect_run_id else None,
        "parameters": json.dumps(parameters or {}),
        "is_dev": is_dev,
    }
    with engine.begin() as connection:
        result = connection.execute(
            text(
                """
                INSERT INTO kb.ingest_flow_runs (
                    ingest_item_id,
                    flow_name,
                    prefect_run_id,
                    parameters,
                    is_dev
                )
                VALUES (
                    :ingest_item_id,
                    :flow_name,
                    :prefect_run_id,
                    CAST(:parameters AS JSONB),
                    :is_dev
                )
                RETURNING id
                """
            ),
            payload,
        )
        return result.scalar_one()


def complete_flow_run(
    run_id: UUID,
    *,
    status: str,
    result: Optional[Dict[str, Any]] = None,
    error: Optional[str] = None,
) -> None:
    """Update the telemetry record when the ingestion flow finishes."""

    _require_engine()
    engine = get_engine()
    payload = {
        "run_id": str(run_id),
        "status": status,
        "result": json.dumps(result or {}),
        "error": error,
        "finished_at": datetime.now(timezone.utc),
    }
    with engine.begin() as connection:
        connection.execute(
            text(
                """
                UPDATE kb.ingest_flow_runs
                SET status = :status,
                    result = CAST(:result AS JSONB),
                    error = :error,
                    finished_at = :finished_at,
                    updated_at = NOW()
                WHERE id = CAST(:run_id AS UUID)
                """
            ),
            payload,
        )


__all__ = ["begin_flow_run", "complete_flow_run"]
