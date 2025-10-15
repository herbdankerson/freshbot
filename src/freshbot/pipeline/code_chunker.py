"""Utilities for chunking source code using universal-ctags metadata.

This module wraps the legacy tool-dumpster implementation and adapts it so the
ingestion pipeline can materialise ``Chunk`` objects that mirror the structure
used elsewhere in the project. The chunker generates semantic ranges based on
ctags output, falling back to line-based splits when tags are sparse. Each
chunk carries the extracted tag metadata so downstream flows can persist it.
"""

from __future__ import annotations

import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional
from uuid import UUID, uuid4

try:  # pragma: no cover - environment validation
    from ctags import CTags, TagEntry  # type: ignore
except ImportError:  # pragma: no cover - runtime guard
    CTags = None  # type: ignore
    TagEntry = None  # type: ignore
    CTAGS_IMPORT_ERROR: Optional[Exception] = None
else:
    CTAGS_IMPORT_ERROR = None

try:  # pragma: no cover - optional dependency
    from etl.tasks import chunker  # type: ignore
    from etl.tasks.intake_models import Chunk, IngestItem, NormalizedDocument  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - tests without ETL
    from typing import Any

    chunker = None  # type: ignore
    Chunk = IngestItem = NormalizedDocument = Any  # type: ignore

LOGGER = logging.getLogger(__name__)

# Maximum lines per chunk when a tagged region exceeds the soft limit.
MAX_CHUNK_LINES_DEFAULT = 120


def _rough_tokens(text: str) -> int:
    if chunker is not None:
        return chunker.rough_tokens(text)
    return max(1, len(text) // 4 or 1)


class ChunkingError(RuntimeError):
    """Raised when ctags cannot produce semantic chunks for a file."""


def _read_lines(text: str) -> List[str]:
    return text.splitlines()


def _ctags_exec_args(source_path: Path) -> List[str]:
    return ["ctags", "--fields=+nK", "-f", "-", str(source_path)]


def _run_ctags(source_path: Path) -> str:
    if CTags is None:
        raise ChunkingError(
            "python-ctags3 is not available; falling back to generic chunking."
        )
    try:
        result = subprocess.run(
            _ctags_exec_args(source_path),
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:  # pragma: no cover - system setup required
        LOGGER.error("ctags execution failed for %s: %s", source_path, exc)
        raise ChunkingError(
            "ctags execution failed; ensure universal-ctags is installed and in PATH."
        ) from exc
    if not result.stdout.strip():
        LOGGER.warning("ctags produced no output for %s", source_path)
    return result.stdout


def _parse_ctags(output: str) -> List[Dict[str, object]]:
    if CTags is None:
        raise ChunkingError("ctags support unavailable")
    if not output:
        return []

    temp_file = tempfile.NamedTemporaryFile(delete=False, mode="w", encoding="utf-8")
    temp_file.write(output)
    temp_file.close()

    entries: List[Dict[str, object]] = []
    try:
        tag_reader = CTags(temp_file.name)
        entry = TagEntry()
        while tag_reader.next(entry):
            try:
                entries.append(
                    {
                        "name": entry["name"],
                        "line": int(entry["lineNumber"]),
                        "kind": entry["kind"],
                    }
                )
            except Exception:  # pragma: no cover - defensive
                continue
    finally:
        try:
            os.unlink(temp_file.name)
        except FileNotFoundError:
            pass
    return entries


def _apply_line_bounds(tags: List[Dict[str, object]], line_total: int) -> None:
    tags.sort(key=lambda item: item["line"])
    for index, tag in enumerate(tags):
        end_line = line_total
        if index + 1 < len(tags):
            end_line = int(tags[index + 1]["line"]) - 1
        tag["end_line"] = max(tag["line"], end_line)


def _split_large_block(lines: List[str], start: int, end: int, max_lines: int) -> Iterable[str]:
    for cursor in range(start, end + 1, max_lines):
        chunk_end = min(cursor + max_lines - 1, end)
        yield "\n".join(lines[cursor - 1 : chunk_end])


def _build_tag_chunks(
    *,
    lines: List[str],
    tags: List[Dict[str, object]],
    max_chunk_lines: int,
) -> List[Dict[str, object]]:
    if not tags:
        return [
            {
                "start_line": 1,
                "end_line": len(lines),
                "tags": [],
            }
        ]

    definitions: List[Dict[str, object]] = []
    last_end = 0
    for tag in tags:
        start_line = int(tag["line"])
        end_line = int(tag.get("end_line") or start_line)
        if start_line > last_end + 1:
            definitions.append(
                {
                    "start_line": last_end + 1,
                    "end_line": start_line - 1,
                    "tags": [],
                }
            )
        definitions.append(
            {
                "start_line": start_line,
                "end_line": end_line,
                "tags": [tag],
            }
        )
        last_end = end_line
    if last_end < len(lines):
        definitions.append(
            {
                "start_line": last_end + 1,
                "end_line": len(lines),
                "tags": [],
            }
        )

    chunks: List[Dict[str, object]] = []
    for definition in definitions:
        start_line = int(definition["start_line"])
        end_line = int(definition["end_line"])
        block_lines = max(1, end_line - start_line + 1)
        if block_lines > max_chunk_lines:
            for segment in _split_large_block(lines, start_line, end_line, max_chunk_lines):
                chunks.append(
                    {
                        "content": segment,
                        "tags": list(definition["tags"]),
                    }
                )
        else:
            chunks.append(
                {
                    "content": "\n".join(lines[start_line - 1 : end_line]),
                    "tags": list(definition["tags"]),
                }
            )
    return chunks


def chunk_code_document(
    *,
    item: IngestItem,
    document: NormalizedDocument,
    source_path: Path,
    max_chunk_lines: int = MAX_CHUNK_LINES_DEFAULT,
) -> List[Chunk]:
    """Generate semantic code chunks for the supplied ingest artefact."""

    text = document.text or ""
    lines = _read_lines(text)
    if not lines:
        raise ChunkingError(f"Document {source_path} is empty; cannot chunk.")

    ctags_output = _run_ctags(source_path)
    tag_entries = _parse_ctags(ctags_output)
    _apply_line_bounds(tag_entries, len(lines))

    chunk_defs = _build_tag_chunks(lines=lines, tags=tag_entries, max_chunk_lines=max_chunk_lines)

    chunks: List[Chunk] = []
    for index, definition in enumerate(chunk_defs):
        content = str(definition["content"])
        tags = list(definition["tags"])
        heading = "Code"
        if tags:
            tag = tags[0]
            heading = str(tag.get("name") or heading)
        chunk = Chunk(
            id=uuid4(),
            ingest_item_id=item.id,
            document_id=None,
            chunk_index=index,
            heading_path=[heading],
            kind="code",
            text=content,
            token_count=_rough_tokens(content),
            overlap_tokens=0,
            ner_entities=[],
            summary=None,
            metadata={
                "ctags": tags,
                "source_type": item.source_type,
            },
        )
        chunks.append(chunk)
    LOGGER.debug("ctags chunked %s into %d chunk(s)", source_path, len(chunks))
    return chunks
