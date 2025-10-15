"""Composable Prefect flows that implement the ingestion building blocks.

Each flow wraps a single logical step so higher-level pipelines can assemble
them in different orders. The docstrings intentionally read like tool
descriptions so they can double as human-facing documentation.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional

try:  # pragma: no cover - Prefect optional in some environments
    from prefect import flow
except Exception:  # pragma: no cover - fallback shims
    def flow(function=None, *_, **__):  # type: ignore
        if function is None:
            def decorator(fn):
                return fn

            return decorator
        return function

try:  # pragma: no cover - optional dependency
    from etl.tasks import intake_tasks  # type: ignore
    from etl.tasks.intake_models import (  # type: ignore
        AcquiredSource,
        Chunk,
        ChunkEmbedding,
        FlowReport,
        IngestItem,
        NormalizedDocument,
    )
except ModuleNotFoundError as exc:  # pragma: no cover - runtime guard for tests
    intake_tasks = None  # type: ignore
    ETL_IMPORT_ERROR = exc

    from typing import Any

    AcquiredSource = Chunk = ChunkEmbedding = FlowReport = IngestItem = NormalizedDocument = Any  # type: ignore
else:  # pragma: no cover
    ETL_IMPORT_ERROR = None

if TYPE_CHECKING:  # pragma: no cover - hints only
    from etl.tasks.intake_tasks import TableConfig
elif intake_tasks is not None:  # pragma: no cover - runtime import
    from etl.tasks.intake_tasks import TableConfig  # type: ignore
else:  # pragma: no cover - fallback when ETL missing
    from typing import Any

    TableConfig = Any  # type: ignore

from ...pipeline.classification import classify_domain, is_code_extension
from ...pipeline.code_chunker import ChunkingError, chunk_code_document


def _require_etl() -> None:
    if intake_tasks is None:
        raise RuntimeError(
            "Freshbot ingestion flows require the Intellibot ETL package. "
            "Run inside the compose environment so 'etl' is importable."
        ) from ETL_IMPORT_ERROR


@flow(name="freshbot-register-ingest-item")
def register_ingest_item_flow(
    *,
    source_type: str,
    source_uri: str,
    display_name: str,
    metadata: Optional[Dict[str, object]] = None,
    is_dev: bool = False,
) -> IngestItem:
    """Register the ingest artefact and persist its bookkeeping metadata."""

    _require_etl()

    payload = dict(metadata or {})
    payload["is_dev"] = is_dev
    item = intake_tasks.register_ingest_item.fn(
        source_type=source_type,
        source_uri=source_uri,
        display_name=display_name,
        metadata=payload,
    )
    item.is_dev = is_dev
    return item


@flow(name="freshbot-acquire-source")
def acquire_source_flow(
    *,
    item: IngestItem,
    content: Optional[bytes] = None,
) -> AcquiredSource:
    """Fetch or decode the raw bytes for the ingest artefact."""

    _require_etl()
    return intake_tasks.acquire_source.fn(item, content)


@flow(name="freshbot-docling-normalize")
def docling_normalize_flow(*, source: AcquiredSource) -> NormalizedDocument:
    """Run Docling to obtain Markdown, plain text, and structured metadata."""

    _require_etl()
    return intake_tasks.docling_normalize.fn(source)


@flow(name="freshbot-code-normalize")
def code_normalize_flow(*, source: AcquiredSource) -> NormalizedDocument:
    """Decode source bytes into a NormalizedDocument without Docling."""

    text = source.content.decode("utf-8", errors="replace")
    metadata = {
        "language": "code",
        "source": dict(source.metadata),
    }
    return NormalizedDocument(
        ingest_item_id=source.ingest_item_id,
        markdown=text,
        text=text,
        metadata=metadata,
    )


@flow(name="freshbot-doc-chunk")
def doc_chunk_flow(*, item: IngestItem, document: NormalizedDocument) -> List[Chunk]:
    """Chunk Markdown/HTML documents with Docling-derived structure."""

    _require_etl()
    return intake_tasks.chunk_and_ner.fn(item, document)


@flow(name="freshbot-code-chunk")
def code_chunk_flow(
    *,
    item: IngestItem,
    document: NormalizedDocument,
    source_path: Path,
    max_chunk_lines: int = 120,
) -> List[Chunk]:
    """Chunk source code using universal-ctags for semantic boundaries."""

    _require_etl()
    try:
        return chunk_code_document(
            item=item,
            document=document,
            source_path=source_path,
            max_chunk_lines=max_chunk_lines,
        )
    except ChunkingError:
        # Fall back to generic chunking so ingestion still succeeds.
        return intake_tasks.chunk_and_ner.fn(item, document)


@flow(name="freshbot-summarize-document")
def summarize_document_flow(*, item: IngestItem, document: NormalizedDocument) -> IngestItem:
    """Create a Gemini Flash synopsis for the full document."""

    _require_etl()
    return intake_tasks.summarize_document.fn(item, document)


@flow(name="freshbot-detect-emotions")
def detect_emotions_flow(*, item: IngestItem, chunks: List[Chunk]) -> Dict[int, List[Dict[str, object]]]:
    """Attach chunk-level emotion signals for downstream routing."""

    _require_etl()
    return intake_tasks.detect_emotions.fn(item, chunks)


@flow(name="freshbot-build-abstractions")
def build_abstractions_flow(*, chunks: List[Chunk]) -> List[str]:
    """Generate coarse abstractions used when evidence budgets overflow."""

    _require_etl()
    return intake_tasks.build_budgeted_abstractions.fn(chunks)


@flow(name="freshbot-embed-chunks")
def embed_chunks_flow(*, item: IngestItem, chunks: List[Chunk]) -> List[ChunkEmbedding]:
    """Produce embeddings for the supplied chunk list."""

    _require_etl()
    return intake_tasks.embed_chunks.fn(item, chunks)


@flow(name="freshbot-classify-domain")
def classify_domain_flow(
    *,
    item: IngestItem,
    document: NormalizedDocument,
    chunks: List[Chunk],
    agent: Optional[str] = None,
) -> IngestItem:
    """Decide whether the artefact is general, law, or code."""

    _require_etl()
    filename = (
        str(document.metadata.get("source", {}).get("filename"))
        if isinstance(document.metadata, dict)
        else None
    )
    filename = filename or item.display_name

    if is_code_extension(filename):
        domain, confidence, raw = "code", 1.0, {"source": "extension"}
    else:
        domain, confidence, raw = classify_domain(
            filename=filename,
            chunks=chunks,
            agent=agent,
        )
    updated = item.copy()
    updated.domain = domain
    updated.domain_confidence = confidence
    updated.metadata.setdefault("classification", {})
    updated.metadata["classification"].update(raw)
    return updated


@flow(name="freshbot-persist-ingest")
def persist_results_flow(
    *,
    item: IngestItem,
    document: NormalizedDocument,
    chunks: List[Chunk],
    embeddings: List[ChunkEmbedding],
    chunk_emotions: Dict[int, List[Dict[str, object]]],
    abstractions: List[str],
    table_config: Optional[Any] = None,
) -> FlowReport:
    """Write chunks, embeddings, and metadata to ParadeDB."""

    _require_etl()
    return intake_tasks.persist_results.fn(
        item,
        document,
        chunks,
        embeddings,
        chunk_emotions,
        abstractions,
        table_config=table_config,
    )
