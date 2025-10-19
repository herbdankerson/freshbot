"""Centralised hybrid search helpers backed by ParadeDB."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from sqlalchemy import text as sql_text

from src.my_agentic_chatbot.storage.connection import get_engine


def search_kb(
    query: str,
    *,
    limit: int = 10,
    scope: Optional[str] = None,
    document_id: Optional[str] = None,
    include_dev: Optional[bool] = None,
    min_score: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Run the ParadeDB hybrid search helper and return enriched entry rows."""

    if not query:
        raise ValueError("query cannot be empty")

    limit = max(1, min(limit, 50))
    params: Dict[str, Any] = {
        "query": query,
        "scope": scope,
        "limit": limit,
    }
    where_clauses: List[str] = []

    if document_id:
        where_clauses.append("e.document_id = :document_id::uuid")
        params["document_id"] = document_id

    if include_dev is True:
        where_clauses.append("d.is_dev IS TRUE")
    elif include_dev is False:
        where_clauses.append("d.is_dev IS FALSE")

    where_sql = ""
    if where_clauses:
        where_sql = "WHERE " + " AND ".join(where_clauses)

    query_sql = sql_text(
        f"""
        SELECT r.id::text AS entry_id,
               e.document_id::text AS document_id,
               e.meta,
               e.file_name,
               e.title,
               e.summary,
               e.version,
               e.updated_at,
               r.source,
               r.uri,
               r.snippet,
               r.text_score,
               r.vec_score,
               r.score,
               d.is_dev
        FROM kb.search_entries(:query, :scope, :limit) AS r
        JOIN kb.entries AS e ON e.id = r.id
        JOIN kb.documents AS d ON d.id = e.document_id
        {where_sql}
        ORDER BY r.score DESC, e.updated_at DESC
        """
    )

    engine = get_engine()
    with engine.connect() as connection:
        rows = connection.execute(query_sql, params).mappings().all()

    results: List[Dict[str, Any]] = []
    for row in rows:
        score = row.get("score")
        if min_score is not None and score is not None and float(score) < min_score:
            continue
        payload = dict(row)
        if score is not None:
            payload["score"] = float(score)
        if payload.get("text_score") is not None:
            payload["text_score"] = float(payload["text_score"])
        if payload.get("vec_score") is not None:
            payload["vec_score"] = float(payload["vec_score"])
        results.append(payload)
    return results


__all__ = ["search_kb"]
