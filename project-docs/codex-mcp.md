# Codex MCP Server

Codex MCP exposes ParadeDB-backed tools for reading, searching, and editing project code/documentation namespaces. The server runs inside docker compose so it always shares the same ParadeDB instance and filesystem mounts as the ingestion pipeline.

## Service summary
- **Container**: `codex-mcp` (defined in `intellibot/docker-compose.yml`).
- **Port**: `8105` → HTTP + MCP transport at `http://localhost:8105/mcp/`.
- **Mounts**: binds the repo root into `/workspace` and `/app` so filesystem rewrites land on the host checkout.
- **Environment**:
  - `DATABASE_URL` / `FRESHBOT_DATABASE_URL`: ParadeDB DSN (`postgresql://agent:agentpass@paradedb:5432/agentdb`).
  - `CODEX_MCP_SCOPES`: allowed schemas (`project_code,project_docs` by default).
  - `CODEX_CODE_ROOTS` (optional): semicolon/comma separated additional roots for locating files.
  - `CODEX_MCP_PUBLIC_ENDPOINT` (optional): manifest endpoint override.

## Tools
`codex_mcp` registers four FastMCP tools:

| Tool | Purpose |
|------|---------|
| `list_documents` | Returns document ids + metadata for a schema (`project_code` by default). Filters via `search` on `file_name`/`source_uri`. |
| `get_document` | Fetches compiled text and chunk metadata for a document id. |
| `search_entries` | Full-text search (`plainto_tsquery`) against chunk entries per scope. Optional `document_id` to narrow to a single document. |
| `update_chunk` | Replaces chunk text, clears related embeddings, updates ParadeDB records, and rewrites the backing file. |

Each tool accepts an optional `scope` argument (`project_code`, `project_docs`, `kb`) so the agent can target the relevant namespace.

## Editing workflow
1. Use `list_documents`/`search_entries` to locate the desired document and chunk index.
2. Call `get_document` to review the compiled text and chunk breakdown.
3. Issue `update_chunk` with the new text. The helper:
   - Updates `{scope}.chunks`, `{scope}.entries`, and `{scope}.documents`.
   - Clears `{scope}.chunk_embeddings` + `{scope}.document_embeddings` for the affected rows so downstream jobs can re-embed.
   - Rebuilds the compiled document text and writes it back to the resolved filesystem path (preferring `source.path`/`source_uri`).
4. Re-run the ingestion wrapper (inside compose) to re-embed if needed.

All edits should be executed from inside the compose container so the DSN and filesystem mounts resolve correctly.
