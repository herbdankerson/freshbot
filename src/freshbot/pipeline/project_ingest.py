"""Purpose-specific wrappers around the Freshbot ingestion flow."""

from __future__ import annotations

import base64
import mimetypes
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from freshbot.flows.ingestion import freshbot_document_ingest


@dataclass(frozen=True)
class NamespaceBinding:
    """Describe where ingestion results should be stored."""

    namespace: str
    entries_table: Optional[str]
    category: str

    def flow_kwargs(self) -> Dict[str, Optional[str]]:
        return {
            "target_namespace": self.namespace,
            "target_entries": self.entries_table,
        }


PROJECT_CODE_BINDING = NamespaceBinding(
    namespace="project_code",
    entries_table="project_code.entries",
    category="code",
)

PROJECT_DOCS_BINDING = NamespaceBinding(
    namespace="project_docs",
    entries_table="project_docs.entries",
    category="docs",
)


_LANGUAGE_MAP = {
    ".py": "python",
    ".md": "markdown",
    ".rst": "rst",
    ".yml": "yaml",
    ".yaml": "yaml",
    ".json": "json",
    ".sql": "sql",
    ".js": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".jsx": "javascript",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".cs": "csharp",
    ".cpp": "cpp",
    ".c": "c",
    ".rb": "ruby",
    ".php": "php",
}


def ingest_path(
    path: Path,
    *,
    binding: NamespaceBinding,
    display_name: Optional[str] = None,
    extra_metadata: Optional[Mapping[str, Any]] = None,
    source_root: Optional[Path] = None,
    source_type: str = "document",
) -> Dict[str, Any]:
    """Ingest ``path`` using the provided namespace binding.

    The helper reads the file, attaches standard metadata, and calls the
    legacy Prefect flow so callers do not have to remember base64 handling or
    table overrides.
    """

    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)

    payload = resolved.read_bytes()
    metadata = _compose_metadata(
        resolved,
        payload_length=len(payload),
        binding=binding,
        source_root=source_root,
        extra_metadata=extra_metadata,
    )

    encoded = base64.b64encode(payload).decode("ascii")

    return freshbot_document_ingest(
        source_type=source_type,
        source_uri=str(resolved),
        display_name=display_name or resolved.name,
        content_b64=encoded,
        extra_metadata=metadata,
        **binding.flow_kwargs(),
    )


def ingest_project_code(
    path: Path,
    *,
    display_name: Optional[str] = None,
    extra_metadata: Optional[Mapping[str, Any]] = None,
    source_root: Optional[Path] = None,
) -> Dict[str, Any]:
    """Ingest a code artefact into the ``project_code`` namespace."""

    return ingest_path(
        path,
        binding=PROJECT_CODE_BINDING,
        display_name=display_name,
        extra_metadata=extra_metadata,
        source_root=source_root,
    )


def ingest_project_docs(
    path: Path,
    *,
    display_name: Optional[str] = None,
    extra_metadata: Optional[Mapping[str, Any]] = None,
    source_root: Optional[Path] = None,
) -> Dict[str, Any]:
    """Ingest a documentation artefact into the ``project_docs`` namespace."""

    return ingest_path(
        path,
        binding=PROJECT_DOCS_BINDING,
        display_name=display_name,
        extra_metadata=extra_metadata,
        source_root=source_root,
    )


def _compose_metadata(
    path: Path,
    *,
    payload_length: int,
    binding: NamespaceBinding,
    source_root: Optional[Path],
    extra_metadata: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    mime_type, _ = mimetypes.guess_type(path.name)
    language = _LANGUAGE_MAP.get(path.suffix.lower())

    source_block: Dict[str, Any] = {
        "filename": path.name,
        "path": str(path),
        "relative_path": _compute_relative(path, source_root),
        "content_length": payload_length,
        "content_type": mime_type or "application/octet-stream",
        "category": binding.category,
    }
    if language:
        source_block["language"] = language

    metadata: Dict[str, Any] = {
        "source": source_block,
        "freshbot": {
            "namespace": binding.namespace,
            "entries_table": binding.entries_table,
            "category": binding.category,
        },
    }

    if extra_metadata:
        metadata = _merge_metadata(metadata, extra_metadata)

    return metadata


def _compute_relative(path: Path, source_root: Optional[Path]) -> Optional[str]:
    if not source_root:
        return None
    try:
        relative = path.relative_to(source_root.resolve())
    except ValueError:
        return None
    return str(relative)


def _merge_metadata(base: Mapping[str, Any], extra: Mapping[str, Any]) -> Dict[str, Any]:
    """Deep-merge ``extra`` into ``base`` without mutating inputs."""

    result: Dict[str, Any] = {}

    def _merge(dst: Dict[str, Any], src: Mapping[str, Any]) -> None:
        for key, value in src.items():
            if key in dst and isinstance(dst[key], dict) and isinstance(value, Mapping):
                nested: Dict[str, Any] = dict(dst[key])
                _merge(nested, value)
                dst[key] = nested
            elif isinstance(value, Mapping):
                dst[key] = dict(value)
            else:
                dst[key] = value

    _merge(result, base)
    _merge(result, extra)
    return result


__all__ = [
    "NamespaceBinding",
    "PROJECT_CODE_BINDING",
    "PROJECT_DOCS_BINDING",
    "ingest_path",
    "ingest_project_code",
    "ingest_project_docs",
]
