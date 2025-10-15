"""Prefect flows implementing Freshbot tool executors."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

try:
    import yaml  # type: ignore
except ModuleNotFoundError as exc:  # pragma: no cover - defensive guard
    raise RuntimeError(
        "PyYAML is required to use Freshbot tool flows. Install it with 'pip install PyYAML'."
    ) from exc

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
from sqlalchemy import text as sql_text

from freshbot.db import get_engine
from freshbot.registry import get_registry

from ..gateways import chat_completion, embed_code_texts, embeddings_enabled
from .utils import merge_system_prompt


def _logger():
    """Return a Prefect run logger when available, otherwise fall back to stdlib logging."""

    try:
        return get_run_logger()
    except Exception:  # pragma: no cover - missing runtime context
        import logging

        return logging.getLogger(__name__)


@flow(name="freshbot_tool_qwen_chat", persist_result=True)
def qwen_chat_tool(
    *,
    messages: Sequence[Mapping[str, Any]],
    system_prompt: Optional[str] = None,
    gateway_alias: str = "connector_qwen_openai",
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    response_format: Optional[Mapping[str, Any]] = None,
    extra_params: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Invoke the Qwen OpenAI-compatible gateway for chat completions."""

    logger = _logger()
    payload_messages = merge_system_prompt(messages, system_prompt)
    result = chat_completion(
        payload_messages,
        gateway_alias=gateway_alias,
        temperature=temperature,
        max_tokens=max_tokens,
        response_format=response_format,
        extra_params=extra_params,
    )
    content = result.get("content")
    logger.info("Qwen chat completed via %s", result.get("gateway"))
    if content:
        logger.debug("Qwen response: %s", content)
    return result


@flow(name="freshbot_tool_gemini_chat", persist_result=True)
def gemini_chat_tool(
    *,
    messages: Sequence[Mapping[str, Any]],
    system_prompt: Optional[str] = None,
    gateway_alias: str = "connector_gemini_litellm",
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    response_format: Optional[Mapping[str, Any]] = None,
    extra_params: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Invoke Gemini via the LiteLLM gateway for chat completions."""

    logger = _logger()
    payload_messages = merge_system_prompt(messages, system_prompt)
    result = chat_completion(
        payload_messages,
        gateway_alias=gateway_alias,
        temperature=temperature,
        max_tokens=max_tokens,
        response_format=response_format,
        extra_params=extra_params,
    )
    content = result.get("content")
    logger.info("Gemini chat completed via %s", result.get("gateway"))
    if content:
        logger.debug("Gemini response: %s", content)
    return result


@flow(name="freshbot_tool_code_embed", persist_result=True)
def code_embedding_tool(
    *,
    texts: Sequence[str],
    gateway_alias: str = "connector_ollama_code_embedding",
) -> Dict[str, Any]:
    """Embed code snippets via the Ollama embedding gateway."""

    logger = _logger()
    vectors = embed_code_texts(texts, gateway_alias=gateway_alias)
    stubbed = not embeddings_enabled()
    if stubbed:
        logger.warning(
            "Returning stub embeddings for %s snippets via %s (enable real embeddings by setting FRESHBOT_ENABLE_CODE_EMBEDDINGS=1)",
            len(texts),
            gateway_alias,
        )
    else:
        logger.info("Embedded %s snippets via %s", len(texts), gateway_alias)
    dims = len(vectors[0]) if vectors else 0
    return {
        "gateway": gateway_alias,
        "dimensions": dims,
        "count": len(vectors),
        "vectors": vectors,
        "stubbed": stubbed,
    }


@flow(name="freshbot_tool_kb_search", persist_result=True)
def kb_search_tool(
    *,
    query: str,
    limit: int = 5,
    min_score: Optional[float] = None,
) -> Dict[str, Any]:
    """Hybrid search across kb.entries using ParadeDB's helper function."""

    logger = _logger()
    engine = get_engine()
    sql = sql_text(
        """
        WITH ranked AS (
            SELECT id,
                   source,
                   uri,
                   title,
                   snippet,
                   text_score,
                   vec_score,
                   score
            FROM kb.search_entries(:query, NULL, :limit)
        )
        SELECT r.id,
               r.source,
               r.uri,
               r.title,
               r.snippet,
               r.text_score,
               r.vec_score,
               r.score,
               e.summary
        FROM ranked AS r
        LEFT JOIN kb.entries AS e ON e.id = r.id
        ORDER BY r.score DESC
        """
    )
    with engine.connect() as connection:
        rows = connection.execute(sql, {"query": query, "limit": limit}).mappings().all()

    results: List[Dict[str, Any]] = []
    for row in rows:
        score = float(row["score"]) if row.get("score") is not None else None
        if min_score is not None and score is not None and score < min_score:
            continue
        results.append({
            "entry_id": str(row["id"]),
            "source": row.get("source"),
            "uri": row.get("uri"),
            "title": row.get("title"),
            "snippet": row.get("snippet"),
            "text_score": row.get("text_score"),
            "vec_score": row.get("vec_score"),
            "score": score,
            "summary": row.get("summary"),
        })
    logger.info("KB search returned %s results for '%s'", len(results), query)
    return {"query": query, "results": results}


def _pattern(value: str) -> str:
    return f"%{value.lower()}%"


@flow(name="freshbot_tool_search_tools")
def search_tools_tool(
    *,
    query: str,
    limit: int = 10,
    include_disabled: bool = False,
) -> Dict[str, Any]:
    """Search ``cfg.tools`` for matching slugs, notes, or manifests."""

    logger = _logger()
    engine = get_engine()
    sql = sql_text(
        """
        SELECT slug,
               kind,
               manifest_or_ref,
               default_params,
               enabled,
               notes
        FROM cfg.tools
        WHERE (LOWER(slug) LIKE :pattern
               OR LOWER(manifest_or_ref) LIKE :pattern
               OR LOWER(COALESCE(notes, '')) LIKE :pattern)
          AND (:include_disabled OR enabled)
        ORDER BY slug
        LIMIT :limit
        """
    )
    params = {
        "pattern": _pattern(query),
        "limit": limit,
        "include_disabled": include_disabled,
    }
    with engine.connect() as connection:
        rows = connection.execute(sql, params).mappings().all()
    logger.info("Tool search returned %s rows for '%s'", len(rows), query)
    results = []
    for row in rows:
        results.append(
            {
                "slug": row["slug"],
                "kind": row.get("kind"),
                "manifest_or_ref": row.get("manifest_or_ref"),
                "enabled": bool(row.get("enabled")),
                "notes": row.get("notes"),
                "default_params": dict(row.get("default_params") or {}),
            }
        )
    return {"query": query, "results": results}


@flow(name="freshbot_tool_search_agents")
def search_agents_tool(
    *,
    query: str,
    limit: int = 10,
    include_disabled: bool = False,
) -> Dict[str, Any]:
    """Search ``cfg.agents`` for matching names, notes, or types."""

    logger = _logger()
    engine = get_engine()
    sql = sql_text(
        """
        SELECT name,
               type,
               model_alias,
               params,
               enabled,
               notes
        FROM cfg.agents
        WHERE (LOWER(name) LIKE :pattern
               OR LOWER(COALESCE(notes, '')) LIKE :pattern
               OR LOWER(type::text) LIKE :pattern)
          AND (:include_disabled OR enabled)
        ORDER BY name
        LIMIT :limit
        """
    )
    params = {
        "pattern": _pattern(query),
        "limit": limit,
        "include_disabled": include_disabled,
    }
    with engine.connect() as connection:
        rows = connection.execute(sql, params).mappings().all()
    logger.info("Agent search returned %s rows for '%s'", len(rows), query)
    return {
        "query": query,
        "results": [
            {
                "name": row["name"],
                "type": row.get("type"),
                "model_alias": row.get("model_alias"),
                "enabled": bool(row.get("enabled")),
                "notes": row.get("notes"),
                "params": dict(row.get("params") or {}),
            }
            for row in rows
        ],
    }


@flow(name="freshbot_tool_agent_registry_snapshot")
def agent_registry_snapshot_tool() -> Dict[str, Any]:
    """Return a snapshot of configured providers, models, agents, and tools."""

    logger = _logger()
    registry = get_registry()
    summary = {
        "providers": list(registry.providers.keys()),
        "models": list(registry.models.keys()),
        "agents": list(registry.agents.keys()),
        "tools": list(registry.tools.keys()),
    }
    logger.info("Registry snapshot collected: %s", summary)
    return {
        "summary": summary,
        "providers": {slug: record.id for slug, record in registry.providers.items()},
        "models": {alias: record.provider_slug for alias, record in registry.models.items()},
        "agents": {name: record.type for name, record in registry.agents.items()},
        "tools": {slug: record.kind for slug, record in registry.tools.items()},
    }


def _load_flows_manifest() -> List[Dict[str, Any]]:
    manifest_path = Path(__file__).with_name("flows.yaml")
    if not manifest_path.exists():
        raise RuntimeError(f"Prefect flows manifest not found at {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    flows = payload.get("flows")
    if not isinstance(flows, list):
        return []
    cleaned: List[Dict[str, Any]] = []
    for entry in flows:
        if not isinstance(entry, dict):
            continue
        cleaned.append(
            {
                "callable": entry.get("callable"),
                "name": entry.get("name"),
                "description": entry.get("description"),
                "tags": entry.get("tags") or [],
                "work_pool": entry.get("work_pool"),
                "work_queue": entry.get("work_queue"),
            }
        )
    return cleaned


@flow(name="freshbot_tool_prefect_flow_catalog", persist_result=True)
def prefect_flow_catalog_tool(
    *,
    include_tags: bool = True,
    search: Optional[str] = None,
) -> Dict[str, Any]:
    """Return a catalog of Prefect deployments defined in ``flows.yaml``."""

    logger = _logger()
    manifest = _load_flows_manifest()
    logger.info("Prefect flow catalog enumerated %s entries", len(manifest))

    def _matches(entry: Mapping[str, Any]) -> bool:
        if not search:
            return True
        needle = search.lower()
        for key in ("callable", "name", "description"):
            candidate = entry.get(key)
            if isinstance(candidate, str) and needle in candidate.lower():
                return True
        for tag in entry.get("tags") or []:
            if isinstance(tag, str) and needle in tag.lower():
                return True
        return False

    catalog: List[Dict[str, Any]] = []
    for entry in manifest:
        if not _matches(entry):
            continue
        record = {
            "callable": entry.get("callable"),
            "deployment_name": entry.get("name"),
            "description": entry.get("description"),
            "work_pool": entry.get("work_pool"),
            "work_queue": entry.get("work_queue"),
        }
        if include_tags:
            record["tags"] = entry.get("tags") or []
        catalog.append(record)

    return {
        "count": len(catalog),
        "flows": catalog,
        "search": search,
        "instructions": (
            "Use 'freshbot.executors.prefect.execute_flow' with the deployment name "
            "reported here to schedule a run. Deployment identifiers follow "
            "'<flow-name>/<deployment>' format."
        ),
    }


def _fetch_tool_metadata(tool_slug: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    engine = get_engine()
    with engine.connect() as connection:
        row = connection.execute(
            sql_text(
                """
                SELECT slug,
                       kind,
                       manifest_or_ref,
                       default_params,
                       enabled,
                       notes
                FROM cfg.tools
                WHERE slug = :slug
                """
            ),
            {"slug": tool_slug},
        ).mappings().one_or_none()
        if row is None:
            raise ValueError(f"Tool '{tool_slug}' is not registered in cfg.tools")
        tool_entry = {
            "slug": row["slug"],
            "kind": row["kind"],
            "manifest_or_ref": row.get("manifest_or_ref"),
            "default_params": dict(row.get("default_params") or {}),
            "enabled": bool(row.get("enabled")),
            "notes": row.get("notes"),
        }
        doc_rows = connection.execute(
            sql_text(
                """
                SELECT id,
                       file_name,
                       source_uri,
                       summary,
                       version,
                       updated_at,
                       metadata
                FROM kb.documents
                WHERE (metadata -> 'extra' -> 'tool_slugs') ? :slug
                ORDER BY updated_at DESC
                """
            ),
            {"slug": tool_slug},
        ).mappings().all()

    docs: List[Dict[str, Any]] = []
    for row in doc_rows:
        docs.append(
            {
                "document_id": str(row["id"]),
                "file_name": row.get("file_name"),
                "source_uri": row.get("source_uri"),
                "summary": row.get("summary"),
                "version": row.get("version"),
                "updated_at": row.get("updated_at"),
                "metadata": row.get("metadata"),
            }
        )
    return tool_entry, docs


@flow(name="freshbot_tool_manifest_fetch", persist_result=True)
def tool_manifest_fetch_tool(
    *,
    tool_slug: str,
    include_docs: bool = True,
) -> Dict[str, Any]:
    """Return configuration metadata and optional documentation for ``tool_slug``."""

    logger = _logger()
    tool_entry, docs = _fetch_tool_metadata(tool_slug)
    logger.info("Retrieved manifest for tool '%s' (docs=%s)", tool_slug, bool(docs))
    payload: Dict[str, Any] = {"tool": tool_entry}
    if include_docs:
        payload["documents"] = docs
    payload["next_steps"] = (
        "Use the Prefect execution tool to run the callable referenced by "
        "`manifest_or_ref`, or consult the linked documentation for usage guidance."
    )
    return payload


@flow(name="freshbot_tool_agent_catalog", persist_result=True)
def agent_catalog_tool(
    *,
    include_disabled: bool = False,
) -> Dict[str, Any]:
    """Summarise registered agents, their thinking patterns, and tool bindings."""

    logger = _logger()
    engine = get_engine()
    with engine.connect() as connection:
        agents = connection.execute(
            sql_text(
                """
                SELECT id,
                       name,
                       type,
                       model_alias,
                       params,
                       tools_profile,
                       db_scope,
                       enabled,
                       notes
                FROM cfg.agents
                WHERE (:include_disabled OR enabled)
                ORDER BY name
                """
            ),
            {"include_disabled": include_disabled},
        ).mappings().all()
        bindings = connection.execute(
            sql_text(
                """
                SELECT a.name AS agent_name,
                       t.slug AS tool_slug,
                       COALESCE(at.overrides, '{}'::jsonb) AS overrides
                FROM cfg.agent_tools AS at
                JOIN cfg.agents AS a ON a.id = at.agent_id
                JOIN cfg.tools AS t ON t.id = at.tool_id
                WHERE (:include_disabled OR a.enabled)
                ORDER BY a.name, t.slug
                """
            ),
            {"include_disabled": include_disabled},
        ).mappings().all()

    tools_by_agent: Dict[str, List[Dict[str, Any]]] = {}
    for binding in bindings:
        tools_by_agent.setdefault(binding["agent_name"], []).append(
            {
                "tool_slug": binding["tool_slug"],
                "overrides": dict(binding.get("overrides") or {}),
            }
        )

    catalog: List[Dict[str, Any]] = []
    for agent in agents:
        params = dict(agent.get("params") or {})
        catalog.append(
            {
                "name": agent["name"],
                "type": agent.get("type"),
                "model_alias": agent.get("model_alias"),
                "tools_profile": agent.get("tools_profile"),
                "thinking_mode": params.get("strategy") or params.get("thinking_mode"),
                "default_params": params,
                "db_scope": list(agent.get("db_scope") or []),
                "enabled": bool(agent.get("enabled")),
                "notes": agent.get("notes"),
                "tools": tools_by_agent.get(agent["name"], []),
            }
        )

    logger.info("Assembled catalog for %s agents", len(catalog))
    return {
        "agents": catalog,
        "include_disabled": include_disabled,
        "instructions": (
            "Select an agent configuration and trigger the corresponding Prefect agent flow "
            "via the Prefect execution tool. Customise tool overrides as needed."
        ),
    }


@flow(name="freshbot_tool_graph_capabilities_map", persist_result=True)
def graph_capabilities_map_tool(
    *,
    limit: int = 100,
    include_nodes: bool = True,
) -> Dict[str, Any]:
    """Return a summary of tool/agent relationships stored in Neo4j."""

    logger = _logger()
    try:
        from neo4j import GraphDatabase  # type: ignore
    except ModuleNotFoundError:
        logger.warning("Neo4j driver is not installed; returning availability hint")
        return {
            "available": False,
            "message": "Neo4j driver is not installed. Install the 'neo4j' package to enable graph queries.",
            "nodes": [],
            "relationships": [],
        }

    from freshbot.flows.tasks.sync import _resolve_neo4j_connection  # type: ignore

    settings = _resolve_neo4j_connection()
    driver = GraphDatabase.driver(
        settings["uri"],
        auth=(settings["user"], settings["password"]),
    )

    relationships: List[Dict[str, Any]] = []
    nodes: List[Dict[str, Any]] = []
    with driver.session(database=settings["database"]) as session:
        rel_records = session.run(
            """
            MATCH (t:Tool)-[r]->(target)
            RETURN t.slug AS tool_slug,
                   labels(target) AS target_labels,
                   COALESCE(target.slug, target.name, target.id) AS target_id,
                   type(r) AS relationship_type
            ORDER BY tool_slug, relationship_type, target_id
            LIMIT $limit
            """,
            limit=limit,
        )
        for record in rel_records:
            relationships.append(
                {
                    "tool_slug": record["tool_slug"],
                    "target_labels": list(record["target_labels"] or []),
                    "target_id": record["target_id"],
                    "relationship": record["relationship_type"],
                }
            )

        if include_nodes:
            node_records = session.run(
                """
                MATCH (n)
                WHERE ANY(label IN labels(n) WHERE label IN ['Tool', 'Agent', 'Flow'])
                RETURN labels(n) AS labels,
                       COALESCE(n.slug, n.name, n.id) AS identifier,
                       n.description AS description
                LIMIT $limit
                """,
                limit=limit,
            )
            for record in node_records:
                nodes.append(
                    {
                        "labels": list(record["labels"] or []),
                        "identifier": record["identifier"],
                        "description": record.get("description"),
                    }
                )

    logger.info(
        "Graph capabilities map produced %s relationships (nodes=%s)",
        len(relationships),
        len(nodes),
    )
    return {
        "available": True,
        "relationships": relationships,
        "nodes": nodes,
        "instructions": (
            "Use this map to understand how tools, agents, and flows connect. "
            "Augment the graph using Neo4j tooling or rerun the sync deployment "
            "after updating task or tool metadata."
        ),
    }


__all__ = [
    "agent_registry_snapshot_tool",
    "agent_catalog_tool",
    "code_embedding_tool",
    "graph_capabilities_map_tool",
    "gemini_chat_tool",
    "kb_search_tool",
    "prefect_flow_catalog_tool",
    "qwen_chat_tool",
    "search_agents_tool",
    "search_tools_tool",
    "tool_manifest_fetch_tool",
]
