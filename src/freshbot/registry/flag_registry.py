"""Centralised access to configurable ingest/search flags."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional

from freshbot.db.connection import psycopg_connection
from psycopg import sql


@dataclass(frozen=True)
class FlagDefinition:
    """Persisted flag metadata loaded from ParadeDB."""

    name: str
    default_value: bool
    enabled: bool
    description: Optional[str]
    metadata: Mapping[str, Any]


@lru_cache(maxsize=1)
def _load_flag_definitions() -> Dict[str, FlagDefinition]:
    """Load enabled flag definitions from the database."""

    query = sql.SQL(
        """
        SELECT
            name,
            description,
            default_value,
            enabled,
            metadata
        FROM cfg.flags
        ORDER BY name
        """
    )

    definitions: Dict[str, FlagDefinition] = {}
    with psycopg_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query)
            for name, description, default_value, enabled, metadata in cur.fetchall():
                meta_payload: Mapping[str, Any]
                if isinstance(metadata, MutableMapping):
                    meta_payload = dict(metadata)
                else:
                    meta_payload = {}
                definitions[name] = FlagDefinition(
                    name=name,
                    default_value=bool(default_value),
                    enabled=bool(enabled),
                    description=description,
                    metadata=meta_payload,
                )
    return definitions


def refresh_cache() -> None:
    """Clear the cached flag definitions."""

    _load_flag_definitions.cache_clear()  # type: ignore[attr-defined]


def list_flags(include_disabled: bool = False) -> List[FlagDefinition]:
    """Return the known flag definitions."""

    definitions = _load_flag_definitions().values()
    if include_disabled:
        return list(definitions)
    return [definition for definition in definitions if definition.enabled]


def _canonical_names() -> Dict[str, str]:
    """Return lowercase -> canonical name mapping."""

    return {definition.name.lower(): definition.name for definition in _load_flag_definitions().values()}


def resolve_flags(
    source_metadata: Optional[Mapping[str, Any]] = None,
    overrides: Optional[Mapping[str, bool]] = None,
) -> Dict[str, bool]:
    """Resolve flag values using defaults + explicit metadata overrides.

    ``source_metadata`` may contain any of:
    - ``flags``: iterable of flag names to force ``True``.
    - ``flag_overrides``: mapping of flag names to booleans.
    - Direct boolean keys matching flag names.

    ``overrides`` allows callers to programmatically set values after applying
    metadata (for example, forcing ``is_document`` for chunk entries).
    """

    definitions = _load_flag_definitions()
    name_lookup = _canonical_names()
    resolved: Dict[str, bool] = {
        name: definition.default_value
        for name, definition in definitions.items()
        if definition.enabled
    }

    if not resolved:
        return {}

    def _apply(name: str, value: bool) -> None:
        canonical = name_lookup.get(name.lower())
        if canonical and canonical in resolved:
            resolved[canonical] = bool(value)

    metadata = source_metadata or {}

    raw_flags = metadata.get("flags")
    if isinstance(raw_flags, Mapping):
        for key, value in raw_flags.items():
            _apply(str(key), bool(value))
    elif isinstance(raw_flags, Iterable) and not isinstance(raw_flags, (str, bytes)):
        for item in raw_flags:
            _apply(str(item), True)

    raw_overrides = metadata.get("flag_overrides")
    if isinstance(raw_overrides, Mapping):
        for key, value in raw_overrides.items():
            _apply(str(key), bool(value))

    # Allow direct boolean keys in metadata (e.g., {"is_note": True})
    for key, value in metadata.items():
        if isinstance(value, bool):
            _apply(str(key), value)

    if overrides:
        for key, value in overrides.items():
            _apply(str(key), value)

    return resolved


def active_flag_names(resolved_flags: Mapping[str, bool]) -> List[str]:
    """Return flag names that resolve to ``True``."""

    return sorted(name for name, value in resolved_flags.items() if value)


__all__ = [
    "FlagDefinition",
    "active_flag_names",
    "list_flags",
    "refresh_cache",
    "resolve_flags",
]
