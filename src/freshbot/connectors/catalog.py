"""Lookup utilities for Freshbot connector metadata."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from sqlalchemy import text

from freshbot.db import get_engine


def lookup(alias: Optional[str] = None) -> List[Dict[str, Any]]:
    """Return connector model entries stored in ``cfg.models``.

    Parameters
    ----------
    alias:
        Optional connector alias to filter on. When omitted all connectors with
        ``purpose = 'connector'`` are returned.
    """

    engine = get_engine()
    query = "SELECT alias, name, endpoint, default_params, config, enabled FROM cfg.models WHERE purpose = 'connector'"
    params: Dict[str, Any] = {}
    if alias:
        query += " AND alias = :alias"
        params["alias"] = alias

    with engine.connect() as connection:
        rows = connection.execute(text(query), params).mappings().all()

    connectors: List[Dict[str, Any]] = []
    for row in rows:
        if not row.get("enabled", True):
            continue
        default_params = row.get("default_params") or {}
        if isinstance(default_params, str):
            default_params = json.loads(default_params)
        config = row.get("config") or {}
        if isinstance(config, str):
            config = json.loads(config)
        connectors.append(
            {
                "alias": row["alias"],
                "name": row["name"],
                "endpoint": row.get("endpoint"),
                "default_params": default_params,
                "config": config,
            }
        )
    return connectors


__all__ = ["lookup"]
