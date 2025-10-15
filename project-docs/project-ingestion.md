# Project Ingestion Overview

> **Run inside Docker.** Execute all ingestion commands from a compose container (for example `docker compose exec prefect-worker python ...`). Host-side runs cannot reach Prefect, ParadeDB, or the embedding gateways and will fall back to stubbed behaviour.

## Storage Model

Project artefacts now land in the primary `kb.*` tables. Dev-specific content is flagged with `is_dev = TRUE` instead of being written to separate schemas. Legacy schemas (`project_code.*`, `project_docs.*`) are dropped automatically by the schema migration so fresh deployments never see the stale tables. Companion tables provide audit metadata:

- `kb.dev_documents_meta` – source bookkeeping, classification provenance, ingest metadata.
- `kb.ingest_flow_runs` – one row per Prefect ingestion run with parameters, status, and runtime telemetry.

```sql
SELECT is_dev, COUNT(*) FROM kb.documents GROUP BY 1;
SELECT document_id, source_uri, file_name, domain, domain_confidence, classifier_source
FROM kb.dev_documents_meta
ORDER BY updated_at DESC
LIMIT 10;
```

This keeps the schema uncluttered while allowing agents to exclude development material with a simple `WHERE is_dev = FALSE`.

## Flow Building Blocks

Every ingestion stage ships as an individual Prefect flow so we can remix them as needed:

| Flow Name | Description |
|-----------|-------------|
| `freshbot-register-ingest-item` | Persist ingest bookkeeping and mark `is_dev`. |
| `freshbot-acquire-source` | Load raw bytes from disk or remote URIs. |
| `freshbot-docling-normalize` | Convert documents via Docling. |
| `freshbot-code-normalize` | Decode source files without Docling. |
| `freshbot-doc-chunk` | Chunk Docling output with heading-aware splits. |
| `freshbot-code-chunk` | Chunk code using universal-ctags, with line-based fallback. |
| `freshbot-classify-domain` | Extension-first domain classifier with Qwen fallback. |
| `freshbot-summarize-document` | Gemini Flash document synopsis (no chunk summaries). |
| `freshbot-embed-chunks` | Generate chunk embeddings (general/legal/code). |
| `freshbot-detect-emotions` | Optional sentiment tagging for downstream routing. |
| `freshbot-build-abstractions` | Coarse abstractions used by retrieval. |
| `freshbot-persist-ingest` | Write documents, chunks, embeddings, and metadata back to ParadeDB. |

The compiled pipelines expose convenient entry points:

- `freshbot-ingest-pipeline` – raw lego box; accepts namespace, entries table, and `is_dev`.
- `freshbot-ingest-general` – normal documents (`kb.*`, Docling chunking).
- `freshbot-ingest-law` – legal documents (`kb.*`, Docling chunking, legal flagging).
- `freshbot-ingest-code` – source code (ctags chunking, code embeddings).
- `freshbot_document_ingest` – legacy wrapper that now delegates to the modular pipeline.

All flows are registered in `freshbot/src/freshbot/flows/flows.yaml` so Prefect deployments stay in sync.

## Wrapper Helpers

`freshbot.pipeline.project_ingest` exposes convenience wrappers that set the right metadata and mark artefacts as dev-only:

- `ingest_project_code(path, source_root=...)`
- `ingest_project_docs(path, source_root=...)`

Both call `freshbot_document_ingest` with `is_dev=True`, store the relative path, tag the source as `code`/`docs`, and copy any extra metadata you provide.

## Classification & Chunking Rules

1. **Domain detection**: extensions determine obvious code files. Everything else is classified by Qwen 3 using the first 10 chunks plus the filename.
2. **Chunking**:
   - Code → universal-ctags, with metadata stored on each chunk (`chunk.metadata["ctags"]`).
   - General/Law → Docling sections + sentence-aware splitting.
3. **Summaries**: only document-level Gemini summaries are produced (chunk summaries are skipped).

## Running a Dev Ingest

```bash
docker compose exec prefect-worker python - <<'PY'
from pathlib import Path
from freshbot.pipeline import project_ingest

root = Path("/workspace/freshbot")
project_ingest.ingest_project_code(root / "src/freshbot/flows/ingest/steps.py", source_root=root)
project_ingest.ingest_project_docs(root / "README.md", source_root=root)
PY
```

Verify the results:

```sql
SELECT file_name, is_dev, domain
FROM kb.documents
WHERE is_dev
ORDER BY updated_at DESC
LIMIT 5;

SELECT document_id,
       domain,
       domain_confidence,
       classifier_source,
       extra->'classification'
FROM kb.dev_documents_meta
ORDER BY updated_at DESC
LIMIT 5;

SELECT flow_name,
       status,
       started_at,
       finished_at,
       parameters
FROM kb.ingest_flow_runs
ORDER BY started_at DESC
LIMIT 5;
```

## Supporting Scripts

- `python -m freshbot.devtools.registry_loader --apply` keeps `cfg.*` tables aligned with the registry definitions.
- `python -m freshbot.devtools.table_loader --table kb.documents --rows /path/to/dev.yaml` offers schema-aware bulk upserts when flows are overkill.
- `python -m freshbot.devtools.prefect_loader` registers or updates Prefect deployments in one shot.
- `python intellibot/ops/scripts/migrate.py` reapplies the schema (drops legacy schemas, ensures dev columns/meta) whenever ParadeDB is reset.

Always run these scripts inside the compose environment so DNS (`paradedb`) and credentials resolve correctly.
