# Codex MCP Server

Codex MCP exposes ParadeDB-backed tools for reading, searching, and editing project code/documentation namespaces. The server runs inside docker compose so it always shares the same ParadeDB instance and filesystem mounts as the ingestion pipeline.

## Service summary
- **Container**: `codex-mcp` (defined in `intellibot/docker-compose.yml`).
- **Port**: `8105` → HTTP + MCP transport at `http://localhost:8105/mcp/`.
- **Mounts**: binds the repo root into `/workspace` and `/app` so filesystem rewrites land on the host checkout.
- **Environment**:
  - `DATABASE_URL` / `FRESHBOT_DATABASE_URL`: ParadeDB DSN (`postgresql://agent:agentpass@paradedb:5432/agentdb`).
  - `CODEX_MCP_SCOPES`: logical scopes (`kb,kb_dev,kb_all` by default).
  - `CODEX_CODE_ROOTS` (optional): semicolon/comma separated additional roots for locating files.
  - `CODEX_MCP_PUBLIC_ENDPOINT` (optional): manifest endpoint override.
  - `CODEX_UPLOAD_ROOT` (optional): base directory for uploaded files (defaults to the first `CODEX_CODE_ROOTS` entry or `/workspace`).
- **Profiles**: All legacy MCP servers and the shared Cloudflare tunnel now sit behind the `mcp-legacy`/`network` compose profile (also accessible via `intellibot/ops/scripts/network_stack.sh`). Keep it idle for local-only work; start it when you need the older toolboxes or remote exposure (for example `docker compose --profile network up -d legal-mcp` from the `intellibot/` root).
- **Health check**: `curl http://localhost:8105/.well-known/mcp.json` should return the FastMCP manifest if the server is up.

## Tools
Codex MCP now exposes a full project knowledge and task-management surface. Tools fall into two groups:

### Knowledge base tools
| Tool | Purpose |
|------|---------|
| `list_scopes` | Enumerate logical scopes with schema + `is_dev` handling. |
| `list_documents` | Return document identifiers and metadata for a scope (optionally filter by filename/source URI). |
| `get_document` | Fetch compiled text, chunk metadata, dev flags, and filesystem paths. |
| `search_entries` | Hybrid BM25/pg_search across chunk entries; optional `document_id` narrows results. |
| `update_chunk` | Rewrite a chunk (dev docs only), clear embeddings, update ParadeDB rows, and rewrite the backing file. |
| `upload_document` | Write UTF-8 content under the configured upload root and return ingest metadata. |
| `ingest_document` | Run the Freshbot ingestion pipeline against a filesystem path (flags + references supported). |

### Task & project tools
| Tool | Purpose |
|------|---------|
| `list_projects` | List task projects with aggregate counts, statuses, doc linkage, and metadata. |
| `fetch_project` | Fetch a single project with planning document references and task counts. |
| `create_project` | Provision a new project and attach planning metadata (plan/DAG paths, tags). |
| `update_project` | Edit project metadata, tags, status, and planning document references. |
| `list_tasks` | Filter tasks by project, key, status, owner, or tags. |
| `list_open_tasks` | Return non-complete, non-archived tasks (auto-marks `is_blocked` based on DAG edges). |
| `list_next_tasks` | Surface next actionable tasks (dependencies satisfied). |
| `current_task` | Return the highest-priority actionable task across (or within) a project. |
| `fetch_task` | Hydrate a task with dependencies, dependents, activity log, document link, and metadata. |
| `search_tasks` | Full-text search over task titles, keys, owners, notes, and tags. |
| `search_task_documents` | Search knowledge base documents linked to tasks (descriptions, completion notes, DAGs). |
| `update_task` | Update task status/owner/priority/notes/title/description/blocked flag and append an activity log entry. |
| `add_task` | Create a task (auto-create project if missing), persist dependencies/tags, and ingest Markdown docs. |
| `assign_tasks_to_project` | Move tasks into a project, log the reassignment, and refresh task docs. |
| `set_task_dependencies` | Replace a task’s dependency list, rebuild DAG edges, and refresh task docs. |
| `mark_task_complete` | Enforce dependency checks, capture a required summary, ingest completion notes, and update docs. |
| `delete_task` | Archive a task, remove DAG edges, ingest the updated doc, and log the archive reason. |
| `upload_project_plan` | Parse plan markdown (`TASK-A -> TASK-B`), sync `task.dag_edges`, ingest the plan doc, and mirror to Neo4j. |
| `update_project_plan` | Re-ingest plan markdown changes, reapply dependency edges, and mirror to Neo4j. |
| `materialize_goap_plan` | Execute the GOAP planner to generate tasks, ingest a plan note, and sync the project DAG. |
| `sync_task_document` | Re-run the task registry parser for a KB document (inline Prefect flow). |

Scopes remain logical views over the shared `kb.*` tables:

- `kb` → production docs only (`is_dev = false`).
- `kb_dev` → development artefacts only.
- `kb_all` → both production and dev rows.

Writes are automatically constrained to dev artefacts (`is_dev = true`) regardless of scope. Task documents live under `intellibot/project-docs/tasks`, DAGs under `intellibot/project-docs/task-dags`.

### Prefect interaction
- `add_task`, `mark_task_complete`, and `delete_task` call `_prefect_ingest_document` which schedules `freshbot_document_ingest/freshbot-document-ingest` (override via `CODEX_TASK_DOC_DEPLOYMENT`). Each run records telemetry to `kb.ingest_flow_runs`, updates embeddings, and appends task activity entries with `prefect_run_id`.
- `ingest_document` wraps the same deployment for arbitrary filesystem paths, applying `cfg.flags`, reference extraction, and optional background execution.
- `upload_project_plan`/`update_project_plan` schedule both the ingest deployment (override `CODEX_DAG_DOC_DEPLOYMENT`) and `task-sync-neo4j/freshbot-task-graph-sync` (override `CODEX_TASK_GRAPH_DEPLOYMENT`) so ParadeDB and Neo4j stay aligned.
- `sync_task_document` invokes the `task-sync-from-kb` Prefect flow inline (no remote scheduling) to parse/regenerate project/task rows from an existing KB document.

Monitor Prefect runs via `prefect deployment ls` / `prefect flow-run ls` or by querying `kb.ingest_flow_runs`. Graph sync attempts are recorded in `task.graph_sync_runs` (project key, deployment, flow run id, status, message) for alerting and audit trails. Neo4j mirroring can be verified with `MATCH (t:Task)-[:DEPENDS_ON]->(d:Task) RETURN t.task_key, d.task_key`.

## Editing workflow
1. Use `list_documents`/`search_entries` to locate the desired document and chunk index.
2. Call `get_document` to review the compiled text and chunk breakdown.
3. Issue `update_chunk` with the new text. The helper:
   - Updates `{scope}.chunks`, `{scope}.entries`, and `{scope}.documents`.
   - Clears `{scope}.chunk_embeddings` + `{scope}.document_embeddings` for the affected rows so downstream jobs can re-embed.
   - Rebuilds the compiled document text and writes it back to the resolved filesystem path (preferring `source.path`/`source_uri`).
4. Re-run the ingestion wrapper (inside compose) to re-embed if needed.

All edits should be executed from inside the compose container so the DSN and filesystem mounts resolve correctly.

## HTTP convenience routes
For quick smoke-tests (or simple curl usage) the server exposes REST-style wrappers:

- `GET /tools/list-scopes`
- `GET /tools/list-documents?scope=kb_dev&limit=5`
- `GET /tools/document/{document_id}?scope=kb_all`
- `GET /tools/search-entries?scope=kb&query=freshbot&limit=5`
- `GET /tools/list-projects`
- `GET /tools/list-tasks?project_key=DEV-TASK-SYSTEM-20241001`
- `GET /tools/list-open-tasks?project_key=DEV-TASK-SYSTEM-20241001`
- `GET /tools/list-next-tasks?project_key=DEV-TASK-SYSTEM-20241001`
- `GET /tools/current-task?project_key=DEV-TASK-SYSTEM-20241001`
- `GET /tools/search-tasks?query=prefect`
- `GET /tools/search-task-documents?query=registry&task_key=TASK-EXAMPLE`
- `POST /tools/update-chunk` with JSON body `{ "scope": "kb_dev", "document_id": "…", "chunk_index": 0, "new_text": "…" }`
- `POST /tools/upload-document` with JSON `{ "relative_path": "freshbot/dev-notes.md", "content": "…" }`
- `POST /tools/update-task` with JSON `{ "task_key": "TASK-TASK-MANAGER-POC", "status": "complete" }`
- `POST /tools/add-task` with JSON `{ "project_key": "DEV-TASK-SYSTEM-20241001", "title": "Sync docs" }`
- `POST /tools/mark-task-complete` with JSON `{ "task_key": "TASK-TASK-MANAGER-POC", "summary": "Verified ingestion end-to-end." }`
- `POST /tools/delete-task` with JSON `{ "task_key": "TASK-TASK-MANAGER-POC" }`
- `POST /tools/upload-dag` with JSON `{ "project_key": "DEV-TASK-SYSTEM-20241001", "content": "- TASK-A -> TASK-B" }`
- `POST /tools/materialize-goap-plan` with JSON `{ "project_key": "PROJECT-GOAP", "goal": "Ship release" }`
- `POST /tools/sync-task-document` with JSON `{ "document_id": "<uuid>", "is_dev": true }`

These routes call the same logic as the MCP tools, so results mirror what an MCP client would see. Use `CODEX_UPLOAD_ROOT`/`CODEX_CODE_ROOTS` to control where uploaded files land; the default is `/workspace`, which maps to the host repo root under compose.

## Exposing Codex through the shared FastMCP gateway
The stack already fronts MCP services with lightweight Nginx proxies so that a single public endpoint can multiplex multiple servers (see `intellibot/project-docs/fastMCP.md`). Codex slots into that pattern without any new infrastructure—create a small proxy wrapper and point `CODEX_MCP_PUBLIC_ENDPOINT` at the aggregated path.

1. **Create the proxy config.** Add `ops/mcp/codex/nginx.conf` and `ops/mcp/codex/manifest.json` (mirroring the existing Neo4j folders). Use the snippet below as a starting point:

   ```nginx
   # ops/mcp/codex/nginx.conf
   worker_processes  1;
   events { worker_connections  1024; }

   http {
     server {
       listen 8080;

       location = /.well-known/mcp.json {
         proxy_pass http://codex-mcp:8105/.well-known/mcp.json;
         proxy_set_header Host $host;
       }

       location /codex/mcp/ {
         proxy_pass http://codex-mcp:8105/mcp/;
         proxy_http_version 1.1;
         proxy_set_header Host $host;
         proxy_set_header Connection "";
         proxy_read_timeout 3600;
       }
     }
   }
   ```

   ```json
   // ops/mcp/codex/manifest.json
   {
     "name": "codex",
     "transport": "http",
     "endpoint": "/codex/mcp/",
     "manifest": "/.well-known/mcp.json"
   }
   ```

   The proxy directory naming keeps everything self-discoverable—the relay container mounts each `manifest.json` into `/etc/nginx/manifests/<name>.json` and places the server block from `nginx.conf` in front of the upstream MCP instance.

2. **Mount the new proxy.** Update `intellibot/docker-compose.yml` (service `mcp-relay` or whichever shared gateway you are running) to bind-mount the new files, following the same pattern as `ops/mcp/neo4j-*`. Rebuild or restart the gateway container with the network profile active: `docker compose --profile network up -d mcp-relay` (or just `intellibot/ops/scripts/network_stack.sh restart mcp-relay`).

3. **Publish the aggregated URL.** Set `CODEX_MCP_PUBLIC_ENDPOINT` (either in `.env` or the compose service definition) to the proxy path, for example `https://fastmcp.example.com/codex/mcp/`. MCP clients will then pick up the proxied URL from the manifest. The raw `http://localhost:8105/mcp/` endpoint continues to work for local debugging.

Once these steps are complete, Codex appears alongside the other toolboxes that the shared FastMCP gateway serves—no additional compose services are required, and new wrappers can be added simply by dropping more config folders into `ops/mcp/`.

> 🔐 **Local-only default:** With the `network` profile idle, `cloudflared` (the shared MCP tunnel) stays down, leaving only the OpenWebUI tunnel (`cloudflared-openwebui`) for remote access. Bring the general tunnel back with `docker compose --profile network up -d cloudflared` or `intellibot/ops/scripts/network_stack.sh up cloudflared` if you decide to publish Codex or the legacy servers externally.
