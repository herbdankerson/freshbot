# Project-Focused Ingestion Snapshot

> **Run inside docker**: Execute all ingestion commands from a running compose container (e.g. `docker compose exec prefect-worker python ...`). Host-side runs cannot reach Prefect, ParadeDB, or the TEI gateways and will fall back to stubbed embeddings.

## Database Tables
- `project_code.documents` – mirror of `kb.documents` for source code artefacts; linked tables: `project_code.chunks`, `project_code.chunk_embeddings`, `project_code.document_embeddings`.
- `project_docs.documents` – mirror of `kb.documents` for documentation; linked tables: `project_docs.chunks`, `project_docs.chunk_embeddings`, `project_docs.document_embeddings`.
- Embedding columns use `vector(1024)` to stay compatible with ParadeDB and hnsw indexes. Indices are created automatically by the pipeline when embeddings are written.

## Test Ingests (2025-10-12)
| Target | Ingest Item | Source | Chunks | Embedding Spaces | Notes |
|--------|-------------|--------|--------|------------------|-------|
| `project_code` | `3f3b24d3-34b9-42f8-b4a3-59e484956bec` | `src/my_agentic_chatbot/runtime_config.py` | 3 | `emb-general` | Live run from inside `prefect-worker` with TEI online; embeddings contain non-zero values (e.g. `[-0.0032, -0.0155, -0.0203, …]`). |
| `project_docs` | `f5029438-682f-4538-975a-bb1314a6a147` | `project-docs/task.md` | 7 | `emb-general` | Live run via docker with TEI; embeddings populated (sample `[-0.0171, 0.0079, -0.0155, …]`). |

During the container run we mounted the Freshbot package inside `prefect-worker` (`PYTHONPATH=/tmp/freshbot:/tmp/freshbot/src:/app/src:/app`) so `etl.tasks.model_clients` could import `src.freshbot.*`. Install or copy the package into the container before invoking the flow if it is missing.

Column counts confirm the resulting inserts across the alternate namespaces:

```sql
SELECT COUNT(*) FROM project_code.documents;           -- 2
SELECT COUNT(*) FROM project_code.chunks;              -- 4
SELECT COUNT(*) FROM project_code.chunk_embeddings;    -- 4
SELECT COUNT(*) FROM project_code.entries;             -- 3
SELECT COUNT(*) FROM project_docs.documents;           -- 2
SELECT COUNT(*) FROM project_docs.chunk_embeddings;    -- 8
SELECT COUNT(*) FROM project_docs.entries;             -- 7
```

## Agent Type Sanity Checks
Quick calls against the new agent types (with the same gateway fallback) produce stubbed responses while remote LLMs are offline:

- `freshbot-base-agent` → `"Stubbed response (gateway unavailable)"`
- `freshbot-thinking-agent` → steps `[]`, final `"Stubbed response (gateway unavailable)"`
- `freshbot-deep-thinking` → best path `[]`, final `"Stubbed response (gateway unavailable)"`

Once Qwen is reachable these entrypoints will surface real plans thanks to the same runtime wiring.

The dedicated `project_code.entries` and `project_docs.entries` tables were created with `CREATE TABLE ... LIKE kb.entries INCLUDING ALL` so downstream agents can read project-specific responses without polluting the primary KB.

## Wrapper helpers
- Code artefacts: `freshbot.pipeline.ingest_project_code(path, source_root=...)`
- Documentation artefacts: `freshbot.pipeline.ingest_project_docs(path, source_root=...)`

Both wrappers add file metadata (`source.filename`, `source.relative_path`, `source.category`, `source.language`) and point the flow at the correct namespace/entries table. They return the same payload as `freshbot_document_ingest`. Use `pip install -e /workspace/freshbot` inside the compose container so the ETL tasks can import the package without manual copying.

## Supporting scripts
- `python -m freshbot.devtools.registry_loader --apply` keeps ParadeDB’s `cfg.*` tables aligned with the definitions under `src/freshbot/registry/`. It performs schema-aware upserts and must run from a compose container so the correct dependencies and environment variables are loaded.
- `python -m freshbot.devtools.table_loader --table project_code.entries --rows path/to/rows.yaml` is the generic seeding path for any table; it introspects the live schema, validates the YAML payload, and inserts or updates rows while preserving existing UUIDs.
- `python -m freshbot.devtools.prefect_loader` (experimental) will register Prefect deployments straight from code. Use it once the flows are packaged so workers stay in sync with the repo.

These utilities currently read from YAML snapshots; longer-term we plan to manage configuration purely through ParadeDB rows and scripted edits instead of editing files directly.

## Follow-ups
1. Keep the TEI (`tei-gte-large`, `tei-legal`) services running in the compose stack; future host-side runs will break back to stub vectors.
2. Backfill historic artefacts by re-running `freshbot_document_ingest` with `target_namespace='project_code'`/`'project_docs'` for each repository batch.
