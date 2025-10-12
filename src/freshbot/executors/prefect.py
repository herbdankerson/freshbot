"""HTTP client for running Prefect deployments via the API service."""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, Optional

import httpx

LOGGER = logging.getLogger(__name__)

_API_BASE_ENV = "FRESHBOT_API_BASE_URL"
_DEFAULT_API_BASE = "http://api:8000"
_DEFAULT_TIMEOUT = float(os.getenv("FRESHBOT_API_TIMEOUT", "60"))


def _resolve_api_base() -> str:
    base = os.getenv(_API_BASE_ENV, _DEFAULT_API_BASE).strip()
    if not base:
        base = _DEFAULT_API_BASE
    return base.rstrip("/")


def _normalise_response(data: Any) -> Dict[str, Any]:
    if isinstance(data, dict):
        return data
    raise RuntimeError(f"Unexpected response payload from flow runner: {data!r}")


def execute_flow(
    deployment_name: str,
    parameters: Optional[Dict[str, Any]] = None,
    wait_for_completion: bool = False,
) -> Dict[str, Any]:
    """Trigger a Prefect deployment and optionally wait for completion.

    Parameters
    ----------
    deployment_name:
        Prefect deployment name in ``<flow-name>/<deployment>`` format.
    parameters:
        Optional parameter mapping passed to the flow run.
    wait_for_completion:
        If ``True`` the call blocks until the flow run completes and returns
        final state details. Otherwise only the scheduled state metadata is
        returned.
    """

    payload = {
        "deployment_name": deployment_name,
        "parameters": parameters or {},
        "wait_for_completion": bool(wait_for_completion),
    }

    url = f"{_resolve_api_base()}/freshbot/flows/execute"
    LOGGER.info("Triggering deployment %s via %s", deployment_name, url)

    try:
        with httpx.Client(timeout=_DEFAULT_TIMEOUT) as client:
            response = client.post(url, json=payload)
        response.raise_for_status()
    except Exception as exc:  # pragma: no cover - network guard
        message = f"Failed to trigger deployment '{deployment_name}' via API"
        LOGGER.exception(message)
        raise RuntimeError(message) from exc

    try:
        data = _normalise_response(response.json())
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive
        message = "Flow runner returned non-JSON payload"
        LOGGER.exception(message)
        raise RuntimeError(message) from exc

    LOGGER.info(
        "Deployment %s completed with state %s",
        deployment_name,
        data.get("state_name"),
    )
    return data


__all__ = ["execute_flow"]
