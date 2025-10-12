"""Prefect flows implementing Freshbot tool executors."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

from prefect import flow, get_run_logger
from sqlalchemy import text as sql_text

from freshbot.db import get_engine
from freshbot.registry import get_registry

from ..gateways import chat_completion, embed_code_texts, embeddings_enabled
from .utils import merge_system_prompt


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

    logger = get_run_logger()
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

    logger = get_run_logger()
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

    logger = get_run_logger()
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

    logger = get_run_logger()
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

    logger = get_run_logger()
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

    logger = get_run_logger()
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

    logger = get_run_logger()
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


__all__ = [
    "agent_registry_snapshot_tool",
    "code_embedding_tool",
    "gemini_chat_tool",
    "kb_search_tool",
    "qwen_chat_tool",
    "search_agents_tool",
    "search_tools_tool",
]
