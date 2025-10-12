"""Example Prefect flow used for connectivity smoke tests."""

from __future__ import annotations

import datetime as dt
import logging
from typing import Dict

from prefect import flow, task

LOGGER = logging.getLogger(__name__)


@task
def emit_heartbeat() -> Dict[str, str]:
    timestamp = dt.datetime.utcnow().isoformat()
    LOGGER.info("Freshbot heartbeat at %s", timestamp)
    return {"timestamp": timestamp}


@flow(name="freshbot-heartbeat")
def heartbeat_flow() -> Dict[str, str]:
    """Emit a timestamp so we can confirm Prefect deployments work."""

    return emit_heartbeat()


__all__ = ["heartbeat_flow"]
