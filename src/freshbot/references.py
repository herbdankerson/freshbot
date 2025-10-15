"""Helpers for managing document and entry cross references."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, DefaultDict, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Set

from freshbot.db.connection import psycopg_connection
from psycopg import sql
from psycopg.rows import dict_row


def _coerce_json(value: Any) -> Mapping[str, Any]:
    if isinstance(value, MutableMapping):
        return dict(value)
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        if isinstance(parsed, MutableMapping):
            return dict(parsed)
    return {}


def _unique(sequence: Iterable[str]) -> List[str]:
    seen: Set[str] = set()
    ordered: List[str] = []
    for item in sequence:
        candidate = item.strip()
        if not candidate:
            continue
        key = candidate.lower()
        if key in seen:
            continue
        seen.add(key)
        ordered.append(candidate)
    return ordered


def normalise_reference_ids(value: Any) -> List[str]:
    """Flatten nested reference payloads into a list of strings."""

    if value is None:
        return []

    items: List[str] = []
    if isinstance(value, str):
        items.append(value)
    elif isinstance(value, Mapping):
        for nested in value.values():
            items.extend(normalise_reference_ids(nested))
    elif isinstance(value, Iterable) and not isinstance(value, (bytes, bytearray)):
        for nested in value:
            items.extend(normalise_reference_ids(nested))
    else:
        items.append(str(value))
    return _unique(items)


def extract_reference_ids(source_metadata: Optional[Mapping[str, Any]]) -> List[str]:
    """Extract canonical reference IDs from metadata mappings."""

    if not source_metadata:
        return []
    references = source_metadata.get("references")
    return normalise_reference_ids(references)


def _derive_flags(meta: Mapping[str, Any]) -> List[str]:
    if not meta:
        return []
    flags: List[str] = []
    if isinstance(meta.get("active_flags"), Iterable) and not isinstance(meta.get("active_flags"), (str, bytes)):
        flags.extend(str(flag) for flag in meta.get("active_flags") or [])
    flag_map = meta.get("flags")
    if isinstance(flag_map, Mapping):
        for name, value in flag_map.items():
            try:
                enabled = bool(value)
            except Exception:
                enabled = False
            if enabled:
                flags.append(str(name))
    return _unique(flags)


def _build_snippet(content: Optional[str], summary: Optional[str]) -> Optional[str]:
    snippet_source = summary or content
    if not snippet_source:
        return None
    snippet = snippet_source.strip()
    if len(snippet) > 280:
        snippet = f"{snippet[:277]}…"
    return snippet


def _prepare_entry_item(row: Mapping[str, Any]) -> Dict[str, Any]:
    meta = _coerce_json(row.get("meta"))
    references = extract_reference_ids(meta)
    return {
        "kind": "entry",
        "entry_id": row.get("entry_id"),
        "document_id": row.get("document_id"),
        "file_name": row.get("file_name"),
        "title": row.get("title"),
        "snippet": _build_snippet(row.get("content"), row.get("summary")),
        "flags": _derive_flags(meta),
        "references": references,
        "is_note": row.get("is_note"),
        "is_document": row.get("is_document"),
        "updated_at": row.get("updated_at"),
        "is_dev": row.get("is_dev"),
    }


def _prepare_document_item(row: Mapping[str, Any]) -> Dict[str, Any]:
    metadata = _coerce_json(row.get("metadata"))
    references = extract_reference_ids(metadata)
    return {
        "kind": "document",
        "document_id": row.get("document_id"),
        "entry_id": None,
        "file_name": row.get("file_name"),
        "title": row.get("title"),
        "snippet": _build_snippet(row.get("text_full"), row.get("summary")),
        "flags": _derive_flags(metadata),
        "references": references,
        "is_note": False,
        "is_document": True,
        "updated_at": row.get("updated_at"),
        "is_dev": row.get("is_dev"),
    }


def fetch_items_by_ids(reference_ids: Sequence[str]) -> Dict[str, List[Dict[str, Any]]]:
    """Return documents/entries that match explicit reference IDs."""

    ids = _unique(reference_ids)
    if not ids:
        return {}

    items: DefaultDict[str, List[Dict[str, Any]]] = defaultdict(list)
    with psycopg_connection() as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                SELECT
                    id::text AS document_id,
                    file_name,
                    title,
                    summary,
                    text_full,
                    metadata,
                    is_dev,
                    updated_at
                FROM kb.documents
                WHERE id = ANY(%s)
                """,
                (ids,),
            )
            documents = cur.fetchall()
            for row in documents:
                document_id = row["document_id"]
                items[document_id].append(_prepare_document_item(row))

            cur.execute(
                """
                SELECT
                    id::text AS entry_id,
                    document_id::text AS document_id,
                    file_name,
                    title,
                    summary,
                    content,
                    meta,
                    is_note,
                    is_document,
                    is_dev,
                    updated_at
                FROM kb.entries
                WHERE id = ANY(%s)
                """,
                (ids,),
            )
            entries = cur.fetchall()
            for row in entries:
                entry_id = row["entry_id"]
                items[entry_id].append(_prepare_entry_item(row))
    return dict(items)


def fetch_backlinks(
    target_ids: Sequence[str],
    limit_per_reference: int = 10,
) -> Dict[str, List[Dict[str, Any]]]:
    """Return items that reference the provided document IDs."""

    ids = _unique(target_ids)
    if not ids:
        return {}

    backlinks: DefaultDict[str, List[Dict[str, Any]]] = defaultdict(list)

    with psycopg_connection() as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                SELECT
                    ref.value AS referenced_id,
                    e.id::text AS entry_id,
                    e.document_id::text AS document_id,
                    e.file_name,
                    e.title,
                    e.summary,
                    e.content,
                    e.meta,
                    e.is_note,
                    e.is_document,
                    e.is_dev,
                    e.updated_at
                FROM kb.entries AS e
                JOIN LATERAL jsonb_array_elements_text(e.meta -> 'references') AS ref(value) ON TRUE
                WHERE ref.value = ANY(%s)
                ORDER BY e.updated_at DESC NULLS LAST
                """,
                (ids,),
            )
            entry_rows = cur.fetchall()
            for row in entry_rows:
                referenced_id = row["referenced_id"]
                entry_item = _prepare_entry_item(row)
                bucket = backlinks[referenced_id]
                if len(bucket) < limit_per_reference:
                    bucket.append(entry_item)

            cur.execute(
                """
                SELECT
                    ref.value AS referenced_id,
                    d.id::text AS document_id,
                    d.file_name,
                    d.title,
                    d.summary,
                    d.text_full,
                    d.metadata,
                    d.is_dev,
                    d.updated_at
                FROM kb.documents AS d
                JOIN LATERAL jsonb_array_elements_text(d.metadata -> 'references') AS ref(value) ON TRUE
                WHERE ref.value = ANY(%s)
                ORDER BY d.updated_at DESC NULLS LAST
                """,
                (ids,),
            )
            document_rows = cur.fetchall()
            for row in document_rows:
                referenced_id = row["referenced_id"]
                document_item = _prepare_document_item(row)
                bucket = backlinks[referenced_id]
                if len(bucket) < limit_per_reference:
                    bucket.append(document_item)

    return dict(backlinks)


@dataclass(frozen=True)
class ReferenceResolution:
    referenced: Dict[str, List[Dict[str, Any]]]
    backlinks: Dict[str, List[Dict[str, Any]]]


def resolve_related(
    referenced_ids: Sequence[str],
    target_ids: Sequence[str],
    *,
    limit_per_reference: int = 10,
) -> ReferenceResolution:
    """Resolve both outgoing references and backlinks for search responses."""

    referenced_map = fetch_items_by_ids(referenced_ids)
    backlinks_map = fetch_backlinks(target_ids, limit_per_reference=limit_per_reference)
    return ReferenceResolution(referenced=referenced_map, backlinks=backlinks_map)


def build_related_payload(
    references: Sequence[str],
    document_id: Optional[str],
    resolution: ReferenceResolution,
) -> Dict[str, Any]:
    """Assemble the related payload for a search/document response."""

    references_map: Dict[str, List[Dict[str, Any]]] = {}
    for ref in references:
        if ref in resolution.referenced:
            references_map[ref] = resolution.referenced[ref]

    backlinks_list: List[Dict[str, Any]] = []
    if document_id and document_id in resolution.backlinks:
        backlinks_list = resolution.backlinks[document_id]

    return {
        "references": references_map,
        "backlinks": backlinks_list,
    }


__all__ = [
    "ReferenceResolution",
    "build_related_payload",
    "extract_reference_ids",
    "fetch_backlinks",
    "fetch_items_by_ids",
    "normalise_reference_ids",
    "resolve_related",
]
