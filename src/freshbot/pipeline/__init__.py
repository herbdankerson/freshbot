"""Freshbot ingestion-oriented pipeline helpers."""

from .ingestion import (
    DEFAULT_CLASSIFIER_TOOL,
    DEFAULT_GEMINI_TOOL,
    classify_document,
    detect_emotions,
    run_chat_tool,
    summarize_chunks,
    summarize_document,
)
from .project_ingest import (
    PROJECT_CODE_BINDING,
    PROJECT_DOCS_BINDING,
    NamespaceBinding,
    ingest_path,
    ingest_project_code,
    ingest_project_docs,
)

__all__ = [
    "DEFAULT_CLASSIFIER_TOOL",
    "DEFAULT_GEMINI_TOOL",
    "classify_document",
    "detect_emotions",
    "run_chat_tool",
    "summarize_chunks",
    "summarize_document",
    "NamespaceBinding",
    "PROJECT_CODE_BINDING",
    "PROJECT_DOCS_BINDING",
    "ingest_path",
    "ingest_project_code",
    "ingest_project_docs",
]
