"""CLI for loading Freshbot registry definitions into ParadeDB.

Reads YAML/JSON definition files and upserts providers, models, tools,
agents, prompts, and tool bindings into the existing cfg.* tables without
changing the schema. Designed to run inside the ``api`` container so it
reuses the installed dependencies and connection settings.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence

import yaml
from sqlalchemy import text
from sqlalchemy.engine import Connection, Engine

from freshbot.db import get_engine

LOGGER = logging.getLogger(__name__)
_ENV_PATTERN = re.compile(r"\$\{(?P<name>[A-Z0-9_]+)\}")


@dataclass
class RegistryDefinitions:
    """Container for all registry sections loaded from disk."""

    providers: Sequence[Mapping[str, Any]]
    models: Sequence[Mapping[str, Any]]
    tools: Sequence[Mapping[str, Any]]
    agents: Sequence[Mapping[str, Any]]
    agent_tools: Sequence[Mapping[str, Any]]
    prompts: Sequence[Mapping[str, Any]]
    flags: Sequence[Mapping[str, Any]]


def resolve_env_tokens(payload: Any) -> Any:
    """Recursively resolve ``${VARS}`` in strings using the current environment."""

    if isinstance(payload, str):
        def repl(match: re.Match[str]) -> str:
            env_name = match.group("name")
            try:
                return os.environ[env_name]
            except KeyError as exc:
                raise RuntimeError(
                    f"Environment variable '{env_name}' is required to resolve '{payload}'"
                ) from exc

        return _ENV_PATTERN.sub(repl, payload)

    if isinstance(payload, list):
        return [resolve_env_tokens(item) for item in payload]

    if isinstance(payload, tuple):
        return tuple(resolve_env_tokens(item) for item in payload)

    if isinstance(payload, dict):
        return {key: resolve_env_tokens(value) for key, value in payload.items()}

    return payload


def load_section(path: Path, key: str) -> Sequence[Mapping[str, Any]]:
    """Load a list of section entries from ``path`` if it exists."""

    if not path.exists():
        return []

    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    if raw is None:
        return []

    if isinstance(raw, dict):
        if key not in raw:
            raise ValueError(f"File {path} must contain top-level key '{key}'")
        records = raw.get(key) or []
    elif isinstance(raw, list):
        records = raw
    else:
        raise TypeError(f"Unsupported payload in {path}: expected list or dict, got {type(raw)!r}")

    if not isinstance(records, list):
        raise TypeError(f"Section '{key}' in {path} must be a list")

    resolved: List[Mapping[str, Any]] = []
    for entry in records:
        if not isinstance(entry, MutableMapping):
            raise TypeError(f"Items in {path} must be mappings; got {type(entry)!r}")
        resolved.append(resolve_env_tokens(dict(entry)))
    return resolved


def load_definitions(root: Path) -> RegistryDefinitions:
    """Load all registry definition files from ``root``."""

    return RegistryDefinitions(
        providers=load_section(root / "providers.yaml", "providers"),
        models=load_section(root / "models.yaml", "models"),
        tools=load_section(root / "tools.yaml", "tools"),
        agents=load_section(root / "agents.yaml", "agents"),
        agent_tools=load_section(root / "agent_tools.yaml", "agent_tools"),
        prompts=load_section(root / "prompts.yaml", "prompts"),
        flags=load_section(root / "flags.yaml", "flags"),
    )


def _json_dumps(value: Any) -> str:
    return json.dumps(value or {})


def _json_array_dumps(value: Any) -> str:
    if value is None:
        return "[]"
    return json.dumps(value)


def upsert_providers(connection: Connection, providers: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
    provider_ids: Dict[str, int] = {}
    if not providers:
        return provider_ids

    stmt = text(
        """
        INSERT INTO cfg.providers (slug, notes, display_name)
        VALUES (:slug, :notes, :display_name)
        ON CONFLICT (slug) DO UPDATE
        SET notes = EXCLUDED.notes,
            display_name = EXCLUDED.display_name,
            updated_at = NOW()
        RETURNING id, slug
        """
    )

    for entry in providers:
        params = {
            "slug": entry["slug"],
            "notes": entry.get("notes"),
            "display_name": entry.get("display_name"),
        }
        LOGGER.info("Upserting provider %s", params["slug"])
        result = connection.execute(stmt, params)
        record = result.mappings().one()
        provider_ids[record["slug"]] = record["id"]

    return provider_ids


def _fetch_existing_providers(connection: Connection) -> Dict[str, int]:
    records = connection.execute(text("SELECT id, slug FROM cfg.providers")).mappings()
    return {row["slug"]: row["id"] for row in records}


def upsert_models(
    connection: Connection,
    models: Sequence[Mapping[str, Any]],
    provider_ids: Mapping[str, int],
) -> Dict[str, int]:
    model_ids: Dict[str, int] = {}
    if not models:
        return model_ids

    stmt = text(
        """
        INSERT INTO cfg.models (
            alias,
            name,
            endpoint,
            purpose,
            dims,
            default_params,
            pricing,
            enabled,
            notes,
            provider_id,
            config,
            identifier,
            uri_template,
            version
        ) VALUES (
            :alias,
            :name,
            :endpoint,
            :purpose,
            :dims,
            CAST(:default_params AS jsonb),
            CAST(:pricing AS jsonb),
            :enabled,
            :notes,
            :provider_id,
            CAST(:config AS jsonb),
            :identifier,
            :uri_template,
            :version
        )
        ON CONFLICT (alias) DO UPDATE SET
            name = EXCLUDED.name,
            endpoint = EXCLUDED.endpoint,
            purpose = EXCLUDED.purpose,
            dims = EXCLUDED.dims,
            default_params = EXCLUDED.default_params,
            pricing = EXCLUDED.pricing,
            enabled = EXCLUDED.enabled,
            notes = EXCLUDED.notes,
            provider_id = EXCLUDED.provider_id,
            config = EXCLUDED.config,
            identifier = EXCLUDED.identifier,
            uri_template = EXCLUDED.uri_template,
            version = EXCLUDED.version,
            updated_at = NOW()
        RETURNING id, alias
        """
    )

    existing_providers = dict(provider_ids)
    if not existing_providers:
        existing_providers = _fetch_existing_providers(connection)

    for entry in models:
        provider_slug = entry.get("provider") or entry.get("provider_slug")
        if not provider_slug:
            raise ValueError(f"Model '{entry.get('alias')}' is missing provider reference")
        provider_id = existing_providers.get(provider_slug)
        if provider_id is None:
            raise ValueError(f"Provider '{provider_slug}' referenced by model '{entry.get('alias')}' is not registered")

        params = {
            "alias": entry.get("alias") or entry.get("name"),
            "name": entry.get("name") or entry.get("alias"),
            "endpoint": entry.get("endpoint"),
            "purpose": entry.get("purpose", "chat"),
            "dims": entry.get("dims"),
            "default_params": _json_dumps(entry.get("default_params")),
            "pricing": _json_dumps(entry.get("pricing")),
            "enabled": bool(entry.get("enabled", True)),
            "notes": entry.get("notes"),
            "provider_id": provider_id,
            "config": _json_dumps(entry.get("config")),
            "identifier": entry.get("identifier"),
            "uri_template": entry.get("uri_template"),
            "version": entry.get("version"),
        }
        LOGGER.info("Upserting model %s (provider=%s)", params["alias"], provider_slug)
        result = connection.execute(stmt, params)
        record = result.mappings().one()
        model_ids[record["alias"]] = record["id"]

    return model_ids


def _fetch_existing_tools(connection: Connection) -> Dict[str, int]:
    records = connection.execute(text("SELECT id, slug FROM cfg.tools")).mappings()
    return {row["slug"]: row["id"] for row in records}


def upsert_tools(connection: Connection, tools: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
    tool_ids: Dict[str, int] = {}
    if not tools:
        return tool_ids

    stmt = text(
        """
        INSERT INTO cfg.tools (slug, kind, manifest_or_ref, default_params, enabled, notes)
        VALUES (:slug, :kind, :manifest_or_ref, CAST(:default_params AS jsonb), :enabled, :notes)
        ON CONFLICT (slug) DO UPDATE SET
            kind = EXCLUDED.kind,
            manifest_or_ref = EXCLUDED.manifest_or_ref,
            default_params = EXCLUDED.default_params,
            enabled = EXCLUDED.enabled,
            notes = EXCLUDED.notes,
            updated_at = NOW()
        RETURNING id, slug
        """
    )

    for entry in tools:
        params = {
            "slug": entry["slug"],
            "kind": entry.get("kind", "native"),
            "manifest_or_ref": entry.get("manifest_or_ref", ""),
            "default_params": _json_dumps(entry.get("default_params")),
            "enabled": bool(entry.get("enabled", True)),
            "notes": entry.get("notes"),
        }
        LOGGER.info("Upserting tool %s", params["slug"])
        result = connection.execute(stmt, params)
        record = result.mappings().one()
        tool_ids[record["slug"]] = record["id"]

    return tool_ids


def _fetch_existing_agents(connection: Connection) -> Dict[str, int]:
    records = connection.execute(text("SELECT id, name FROM cfg.agents")).mappings()
    return {row["name"]: row["id"] for row in records}


def upsert_agents(connection: Connection, agents: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
    agent_ids: Dict[str, int] = {}
    if not agents:
        return agent_ids

    stmt = text(
        """
        INSERT INTO cfg.agents (
            name,
            type,
            model_alias,
            system_prompt,
            params,
            tools_profile,
            db_scope,
            enabled,
            notes
        ) VALUES (
            :name,
            :type,
            :model_alias,
            :system_prompt,
            CAST(:params AS jsonb),
            :tools_profile,
            CAST(:db_scope AS jsonb),
            :enabled,
            :notes
        )
        ON CONFLICT (name) DO UPDATE SET
            type = EXCLUDED.type,
            model_alias = EXCLUDED.model_alias,
            system_prompt = EXCLUDED.system_prompt,
            params = EXCLUDED.params,
            tools_profile = EXCLUDED.tools_profile,
            db_scope = EXCLUDED.db_scope,
            enabled = EXCLUDED.enabled,
            notes = EXCLUDED.notes,
            updated_at = NOW()
        RETURNING id, name
        """
    )

    for entry in agents:
        params = {
            "name": entry["name"],
            "type": entry.get("type", "base"),
            "model_alias": entry.get("model_alias"),
            "system_prompt": entry.get("system_prompt"),
            "params": _json_dumps(entry.get("params")),
            "tools_profile": entry.get("tools_profile"),
            "db_scope": _json_array_dumps(entry.get("db_scope", [])),
            "enabled": bool(entry.get("enabled", True)),
            "notes": entry.get("notes"),
        }
        LOGGER.info("Upserting agent %s", params["name"])
        result = connection.execute(stmt, params)
        record = result.mappings().one()
        agent_ids[record["name"]] = record["id"]

    return agent_ids


def upsert_agent_tools(
    connection: Connection,
    agent_tools: Sequence[Mapping[str, Any]],
    agent_ids: Mapping[str, int],
    tool_ids: Mapping[str, int],
) -> None:
    if not agent_tools:
        return

    stmt = text(
        """
        INSERT INTO cfg.agent_tools (agent_id, tool_id, overrides)
        VALUES (:agent_id, :tool_id, CAST(:overrides AS jsonb))
        ON CONFLICT (agent_id, tool_id) DO UPDATE SET
            overrides = EXCLUDED.overrides
        """
    )

    existing_agents = dict(agent_ids) or _fetch_existing_agents(connection)
    existing_tools = dict(tool_ids) or _fetch_existing_tools(connection)

    for entry in agent_tools:
        agent_name = entry.get("agent")
        tool_slug = entry.get("tool")
        if not agent_name or not tool_slug:
            raise ValueError("agent_tools entries require 'agent' and 'tool'")
        agent_id = existing_agents.get(agent_name)
        if agent_id is None:
            raise ValueError(f"Agent '{agent_name}' referenced in agent_tools is not registered")
        tool_id = existing_tools.get(tool_slug)
        if tool_id is None:
            raise ValueError(f"Tool '{tool_slug}' referenced in agent_tools is not registered")

        params = {
            "agent_id": agent_id,
            "tool_id": tool_id,
            "overrides": _json_dumps(entry.get("overrides")),
        }
        LOGGER.info("Binding tool %s to agent %s", tool_slug, agent_name)
        connection.execute(stmt, params)


def upsert_prompts(connection: Connection, prompts: Sequence[Mapping[str, Any]]) -> None:
    if not prompts:
        return

    stmt = text(
        """
        INSERT INTO cfg.prompts (name, content, metadata)
        VALUES (:name, :content, CAST(:metadata AS jsonb))
        ON CONFLICT (name) DO UPDATE SET
            content = EXCLUDED.content,
            metadata = EXCLUDED.metadata,
            updated_at = NOW()
        """
    )

    for entry in prompts:
        name = entry.get("name")
        if not name:
            raise ValueError("Prompt entries require a 'name'")
        params = {
            "name": name,
            "content": entry.get("content", ""),
            "metadata": _json_dumps(entry.get("metadata")),
        }
        LOGGER.info("Upserting prompt %s", name)
        connection.execute(stmt, params)


def upsert_flags(connection: Connection, flags: Sequence[Mapping[str, Any]]) -> None:
    if not flags:
        return

    stmt = text(
        """
        INSERT INTO cfg.flags (name, description, default_value, enabled, metadata)
        VALUES (:name, :description, :default_value, :enabled, CAST(:metadata AS jsonb))
        ON CONFLICT (name) DO UPDATE SET
            description = EXCLUDED.description,
            default_value = EXCLUDED.default_value,
            enabled = EXCLUDED.enabled,
            metadata = EXCLUDED.metadata,
            updated_at = NOW()
        """
    )

    for entry in flags:
        name = entry.get("name")
        if not name:
            raise ValueError("Flag entries require a 'name'")
        params = {
            "name": name,
            "description": entry.get("description"),
            "default_value": bool(entry.get("default_value", False)),
            "enabled": bool(entry.get("enabled", True)),
            "metadata": _json_dumps(entry.get("metadata")),
        }
        LOGGER.info("Upserting flag %s", name)
        connection.execute(stmt, params)


def apply_registry(definitions: RegistryDefinitions, engine: Engine, apply: bool) -> None:
    connection = engine.connect()
    transaction = connection.begin()
    try:
        provider_ids = upsert_providers(connection, definitions.providers)
        model_ids = upsert_models(connection, definitions.models, provider_ids)
        tool_ids = upsert_tools(connection, definitions.tools)
        agent_ids = upsert_agents(connection, definitions.agents)
        upsert_agent_tools(connection, definitions.agent_tools, agent_ids, tool_ids)
        upsert_prompts(connection, definitions.prompts)
        upsert_flags(connection, definitions.flags)
        if apply:
            transaction.commit()
            LOGGER.info("Registry changes committed.")
        else:
            transaction.rollback()
            LOGGER.info("Dry-run complete; no changes committed.")
    except Exception:
        transaction.rollback()
        raise
    finally:
        connection.close()


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load Freshbot registry definitions into ParadeDB.")
    parser.add_argument(
        "--registry-dir",
        default=Path("src/freshbot/registry"),
        type=Path,
        help="Path containing providers.yaml, models.yaml, etc.",
    )
    parser.add_argument(
        "--database-url",
        default=None,
        help="Optional database URL override. Defaults to DATABASE_URL env or project settings.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Persist changes instead of running a dry-run.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Log level for output.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s %(message)s")

    registry_dir: Path = args.registry_dir
    if not registry_dir.exists():
        raise SystemExit(f"Registry directory '{registry_dir}' does not exist")

    LOGGER.info("Loading registry definitions from %s", registry_dir)
    definitions = load_definitions(registry_dir)

    database_url = args.database_url
    engine = get_engine(database_url)

    apply_registry(definitions, engine, apply=args.apply)


if __name__ == "__main__":  # pragma: no cover
    main()
