"""Prefect flows that orchestrate task/project registry updates."""

from __future__ import annotations

import os
from dataclasses import asdict
from typing import Any, Dict, Iterable, Optional, Sequence

try:  # pragma: no cover - optional Neo4j driver import
    from neo4j import GraphDatabase
except Exception:  # pragma: no cover - allow operation without driver
    GraphDatabase = None  # type: ignore[assignment]

try:  # pragma: no cover - Prefect optional when running unit tests
    from prefect import flow
except Exception:  # pragma: no cover - fallback when Prefect unavailable
    def flow(function=None, *_, **__):  # type: ignore
        if function is None:
            def decorator(fn):
                return fn

            return decorator
        return function

from sqlalchemy import text

from etl.tasks import task_registry
from src.my_agentic_chatbot.storage.connection import get_engine


def _serialize(parsed: task_registry.ParsedTasksDocument) -> Dict[str, Any]:
    payload = asdict(parsed)
    return payload


@flow(name="task-parse-document")
def parse_task_document_flow(markdown: str) -> Dict[str, Any]:
    """Parse Markdown into structured project + task metadata."""

    parsed = task_registry.parse_task_document(markdown)
    if parsed is None:
        raise ValueError("No task registry table found in document")
    return _serialize(parsed)


@flow(name="task-sync-document")
def sync_task_document_flow(
    *,
    document_id: str,
    ingest_item_id: str,
    markdown: str,
    is_dev: bool = False,
) -> Dict[str, Any]:
    """Persist tasks/projects parsed from a Markdown document."""

    engine = get_engine()
    with engine.begin() as conn:
        parsed = task_registry.sync_task_document(
            conn,
            document_id=document_id,
            ingest_item_id=ingest_item_id,
            markdown=markdown,
            is_dev=is_dev,
        )
    if parsed is None:
        raise ValueError("Document did not contain a task registry")
    return _serialize(parsed)


@flow(name="task-sync-from-kb")
def sync_task_document_from_kb_flow(
    *,
    document_id: str,
    is_dev: bool = False,
) -> Dict[str, Any]:
    """Load a document from the KB and sync its task registry."""

    engine = get_engine()
    with engine.begin() as conn:
        row = conn.execute(
            text(
                """
                SELECT
                    id::text AS document_id,
                    ingest_item_id::text AS ingest_item_id,
                    text_full,
                    metadata
                FROM kb.documents
                WHERE id = :document_id
                """
            ),
            {"document_id": document_id},
        ).mappings().first()

        if row is None:
            raise ValueError(f"Document {document_id} not found in kb.documents")

        markdown = row["text_full"]
        metadata = row.get("metadata")
        ingest_item_id = row.get("ingest_item_id")
        if not ingest_item_id:
            ingest_item_id = document_id

        parsed = task_registry.sync_task_document(
            conn,
            document_id=row["document_id"],
            ingest_item_id=str(ingest_item_id),
            markdown=markdown or "",
            is_dev=is_dev,
        )
    if parsed is None:
        raise ValueError("Document did not contain a task registry")
    return _serialize(parsed)


def _resolve_neo4j_connection() -> Dict[str, str]:
    uri = (
        os.getenv("NEO4J_URL")
        or os.getenv("NEO4J_BOLT_URL")
        or os.getenv("NEO4J_URI")
        or "bolt://neo4j:7687"
    )
    username = os.getenv("NEO4J_USERNAME") or os.getenv("NEO4J_USER") or "neo4j"
    password = os.getenv("NEO4J_PASSWORD") or os.getenv("NEO4J_PASS")
    if not password:
        raise RuntimeError(
            "NEO4J_PASSWORD environment variable must be set for task graph sync"
        )
    database = os.getenv("NEO4J_DATABASE") or "neo4j"
    return {"uri": uri, "user": username, "password": password, "database": database}


def _fetch_task_graph(conn, *, project_key: Optional[str]) -> Dict[str, Sequence[Dict[str, Any]]]:
    filters: Dict[str, Any] = {}
    project_clause = ""
    if project_key:
        project_clause = "WHERE p.project_key = :project_key"
        filters["project_key"] = project_key

    tasks = conn.execute(
        text(
            f"""
            SELECT
                t.id::text          AS task_id,
                t.task_key          AS task_key,
                p.project_key       AS project_key,
                p.name              AS project_name,
                t.title             AS title,
                t.status            AS status,
                t.priority          AS priority,
                t.owner             AS owner,
                COALESCE(ARRAY_TO_STRING(t.tags, ','), '') AS tags,
                t.is_blocked        AS is_blocked,
                t.created_at        AS created_at,
                t.updated_at        AS updated_at
            FROM task.tasks AS t
            JOIN task.projects AS p ON p.id = t.project_id
            {project_clause}
            """
        ),
        filters,
    ).mappings().all()

    edges = conn.execute(
        text(
            f"""
            SELECT
                p.project_key       AS project_key,
                dep.task_key        AS dependency_key,
                tgt.task_key        AS target_key
            FROM task.dag_edges AS e
            JOIN task.projects AS p ON p.id = e.project_id
            JOIN task.tasks AS dep ON dep.id = e.from_task_id
            JOIN task.tasks AS tgt ON tgt.id = e.to_task_id
            {project_clause}
            """
        ),
        filters,
    ).mappings().all()

    return {"tasks": tasks, "edges": edges}


def _sync_to_neo4j(
    *,
    connection_settings: Dict[str, str],
    tasks: Iterable[Dict[str, Any]],
    edges: Iterable[Dict[str, Any]],
    purge_existing: bool,
    project_key: Optional[str],
) -> Dict[str, Any]:
    if GraphDatabase is None:
        raise RuntimeError(
            "Neo4j driver is unavailable. Install the 'neo4j' package to enable graph sync."
        )

    driver = GraphDatabase.driver(
        connection_settings["uri"],
        auth=(connection_settings["user"], connection_settings["password"]),
    )
    summary: Dict[str, Any] = {"projects": set(), "tasks": 0, "edges": 0}
    try:
        with driver.session(database=connection_settings["database"]) as session:
            if purge_existing:
                if project_key:
                    session.run(
                        """
                        MATCH (t:Task {project_key: $project_key})
                        DETACH DELETE t
                        """,
                        project_key=project_key,
                    )
                else:
                    session.run("MATCH (t:Task) DETACH DELETE t")

            for task in tasks:
                summary["projects"].add(task["project_key"])
                session.run(
                    """
                    MERGE (t:Task {project_key: $project_key, task_key: $task_key})
                    SET
                        t.task_id = $task_id,
                        t.project_name = $project_name,
                        t.title = $title,
                        t.status = $status,
                        t.priority = $priority,
                        t.owner = $owner,
                        t.tags = $tags,
                        t.is_blocked = COALESCE($is_blocked, false),
                        t.created_at = datetime($created_at),
                        t.updated_at = datetime($updated_at)
                    """,
                    task,
                )
                summary["tasks"] += 1

            for edge in edges:
                summary["projects"].add(edge["project_key"])
                session.run(
                    """
                    MATCH (dep:Task {project_key: $project_key, task_key: $dependency_key})
                    MATCH (target:Task {project_key: $project_key, task_key: $target_key})
                    MERGE (target)-[r:DEPENDS_ON]->(dep)
                    SET r.updated_at = datetime()
                    """,
                    edge,
                )
                summary["edges"] += 1
    finally:
        driver.close()

    summary["projects"] = sorted(summary["projects"])
    return summary


@flow(name="task-sync-neo4j")
def sync_task_graph_to_neo4j_flow(
    *,
    project_key: Optional[str] = None,
    purge_existing: bool = True,
) -> Dict[str, Any]:
    """Mirror task/project metadata into Neo4j for graph exploration."""

    connection_settings = _resolve_neo4j_connection()
    engine = get_engine()
    with engine.connect() as conn:
        graph_data = _fetch_task_graph(conn, project_key=project_key)
    result = _sync_to_neo4j(
        connection_settings=connection_settings,
        tasks=graph_data["tasks"],
        edges=graph_data["edges"],
        purge_existing=purge_existing,
        project_key=project_key,
    )
    return result
