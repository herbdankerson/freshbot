# Freshbot Phase 0 Setup

This document captures the workflow for managing connectors, tools, agents, and Prefect flows using the new Freshbot registry utilities that live inside the existing `api` container.

## Quick Checklist (as of 2025-10-11)

1. `docker compose exec api python -m freshbot.devtools.registry_loader --registry-dir src/freshbot/registry --apply`
2. `docker compose exec api python -m freshbot.devtools.prefect_loader --flows-spec src/freshbot/flows/flows.yaml --apply`
3. Ensure the dedicated worker service is running: `docker compose up -d prefect-worker`
   - Tail with `docker compose logs -f prefect-worker` to confirm it connected to the `freshbot-process/freshbot-default` queue.
   - The service already exports `PREFECT_RESULTS_PERSIST_BY_DEFAULT=true` and `PREFECT_LOCAL_STORAGE_PATH=/tmp/prefect-results`, so Prefect keeps flow payloads across restarts. For manual debugging you can stop the service and run `docker compose exec api prefect worker start --pool freshbot-process --work-queue freshbot-default` with the same env.
4. From the API container, trigger a smoke run:

```bash
docker compose exec api python - <<'PY'
from freshbot.executors import prefect
print(prefect.execute_flow("freshbot_tool_qwen_chat/freshbot-tool-qwen-chat"))
PY
```

   Confirm `/freshbot/flows/execute` reaches the registered deployment and returns a scheduled (or completed) state. The Prefect slug uses an underscore (`freshbot_tool_qwen_chat/freshbot-tool-qwen-chat`).
   - If the request takes longer than the default 60s HTTP timeout, export `FRESHBOT_API_TIMEOUT=120` (or higher) for the CLI call.
   - Qwen runs now honour a 240s connector timeout (`connector_qwen_openai` in `models.yaml`). If a deployment still fails, inspect the proxy logs to ensure the upstream model responds before the four-minute window.
   - Recent smoke runs: `freshbot_tool_qwen_chat` → `COMPLETED` (`flow_run_id=888b79f8-d3c2-412f-9b8d-59a3fa3e9d4c`, content “Prefect offers robust scheduling…”); `freshbot_tool_gemini_chat` → `COMPLETED` (`flow_run_id=1518d32c-6814-4b19-b676-1af4a8c2bd50`, content “Hi there.”); `freshbot_document_ingest` → `COMPLETED` (`flow_run_id=4cd3e684-6f0c-434b-9d87-43fe70045754`, ingest item `738496ed-c365-4333-b1f4-6b0388541689`); `freshbot_tool_kb_search` → `COMPLETED` (`flow_run_id=26387b21-0ac1-4103-af78-204ca185e267`) returning the new “Freshbot live ingestion” snippet; `freshbot_tool_code_embed` currently reports `COMPLETED` with `stubbed=True` (`flow_run_id=e9f16c18-f520-44f2-bd12-d65a9b10f11d`); metadata flag stubs land via `freshbot_metadata_flag` (latest run `649f102e-27be-40af-b847-f7ba65ef9192`).

Mark these off whenever the compose stack restarts so Prefect deployments and workers stay in sync with the registry.

## Registry Layout

Registry definitions live under `src/freshbot/registry/` and are split by entity type:

- `providers.yaml` – provider rows for `cfg.providers`
- `models.yaml` – extended metadata for connectors persisted in `cfg.models`
- `tools.yaml` – tool manifests and default params stored in `cfg.tools`
- `agents.yaml` – agent definitions for `cfg.agents`
- `agent_tools.yaml` – bindings between agents and tools
- `prompts.yaml` – optional prompt templates referenced by agents

Values may include `${ENV_VAR}` tokens; the loader resolves them before writing to the database. Only the sections present in the directory are processed—omitted files leave existing rows untouched.

### Container environment

The `api` service now exports `PYTHONPATH=/app` so both Prefect workers and ad-hoc CLI runs can import the legacy `etl` package without reaching into `sys.path` at runtime. When invoking the tooling outside Docker, mirror this by running `export PYTHONPATH=$PWD` from the repo root.

### Gateway schemas

Connector rows that represent gateways (e.g., the Ollama encoder) should place request/response schemas and any prompt metadata inside their `default_params`. The runtime loader exposes these through `freshbot.gateways.registry.GatewayConfig`, and helper modules (such as `freshbot.gateways.ollama`) validate payloads before dispatching to the remote service. Keep the schema descriptive—list required keys and note optional overrides so downstream tools can rely on the metadata instead of hard-coding arguments.

## Loading the Registry

Run the loader from inside the running `api` container so it reuses the installed dependencies and environment variables:

```bash
docker compose exec api python -m freshbot.devtools.registry_loader --registry-dir src/freshbot/registry           # dry run
docker compose exec api python -m freshbot.devtools.registry_loader --registry-dir src/freshbot/registry --apply  # commit changes
```

Optional flags:

- `--database-url postgresql+psycopg://...` overrides the default ParadeDB DSN.
- `--log-level DEBUG` emits verbose information while applying changes.

The loader wraps its operations in a single transaction; when run without `--apply` the transaction is rolled back, providing a preview of pending changes.

## Prefect Flow Registration

Store Prefect flow modules in `src/freshbot/flows/` and describe deployments in `src/freshbot/flows/flows.yaml`. Each entry names the import path of a Prefect `@flow` callable plus deployment metadata.

Register or preview deployments via:

```bash
docker compose exec api python -m freshbot.devtools.prefect_loader                     # dry run
docker compose exec api python -m freshbot.devtools.prefect_loader --apply             # apply all deployments
docker compose exec api python -m freshbot.devtools.prefect_loader --apply --flow freshbot-heartbeat-deployment
```

The loader builds a Prefect deployment for each spec and applies it through the Prefect API (available at `http://prefect:4200/api`). Cron schedules and work queue names are honoured when present.

> **Heads-up:** provide a work-pool either inline in `flows.yaml` or by exporting `FRESHBOT_PREFECT_WORK_POOL`. The loader refuses to apply deployments without a resolved pool so you do not accidentally register flows to a pool with no worker.

## Runtime Helpers

- `freshbot.connectors.catalog.lookup()` fetches active connector entries from `cfg.models` so agents can enumerate available sources.
- `freshbot.executors.prefect.execute_flow()` wraps `prefect.deployments.run_deployment` for trigger-and-wait semantics.
- `freshbot.gateways.embed_code_texts()` resolves the Ollama embedding gateway defined in ParadeDB and enforces its schema for code embeddings. The helper runs in **stub mode** (returns zero vectors) unless `FRESHBOT_ENABLE_CODE_EMBEDDINGS=1` is set before starting the API/worker containers.
- `freshbot.pipeline.ingestion` wraps ingestion-time LLM calls (classification, summaries, emotion signals) and routes them through the registry-backed `tool_qwen_chat`/`tool_gemini_chat` definitions.
- `freshbot_metadata_flag` is a stub Prefect deployment invoked whenever ingestion emits `meta_flags` (for example, `needs_metadata`); it currently logs and returns immediately.
- The `execute_flow` helper now calls the running API service (`POST /freshbot/flows/execute`) instead of importing Prefect directly. Set `FRESHBOT_API_BASE_URL` (default `http://api:8000`) and optional `FRESHBOT_API_TIMEOUT` to point at the correct container.
- The FastAPI endpoint responds with HTTP 404 when Prefect reports `ObjectNotFound`; make sure deployments have been applied before invoking tools or agents through the executor.

These utilities integrate with the registry definitions: `tools.yaml` references the Prefect executor while `models.yaml` captures connector metadata that agents can inspect.

## Generic table loader

Use the standalone loader when you need to seed any table (agents, tools, bespoke registries) outside the curated Freshbot YAMLs:

```bash
docker compose exec api python -m freshbot.devtools.table_loader --table cfg.tools --rows-file src/freshbot/registry/custom_tools.yaml --conflict slug --apply
```

The loader reads YAML/JSON, validates keys against the live schema, and issues idempotent inserts when `--conflict` columns are supplied. Keep reusable row bundles under `src/freshbot/registry/` so they travel with the codebase.

## Next Steps

1. Flesh out real Prefect DAGs under `src/freshbot/flows/` and add matching entries to `flows.yaml`.
2. Extend registry YAML files with additional agents/tools as the architecture grows.
3. Wire the upcoming agent executor to pull connector metadata via `freshbot.connectors.catalog.lookup()` and decide which Prefect flows to launch using `freshbot.executors.prefect.execute_flow()`.


## Agent and Tool Deployments

Run the registry loader before applying Prefect deployments so the `cfg.*` tables have the latest agent/tool metadata:

```bash
docker compose exec api python -m freshbot.devtools.registry_loader --registry-dir src/freshbot/registry --apply
```

Then register the flows:

```bash
docker compose exec api python -m freshbot.devtools.prefect_loader --flows-spec src/freshbot/flows/flows.yaml --apply
```

Key deployments:

- `freshbot-agent-planner`, `freshbot-agent-tool-user`, `freshbot-agent-responder`, `freshbot-agent-auditor`, `freshbot-agent-code-executor`, `freshbot-agent-custom`
- `freshbot-tool-qwen-chat`, `freshbot-tool-gemini-chat`, `freshbot-tool-code-embed`, `freshbot-tool-kb-search`, `freshbot-tool-search-agents`, `freshbot-tool-search-tools`, `freshbot-tool-registry-snapshot`

Start the worker service for the `freshbot-process` pool and `freshbot-default` queue before triggering deployments:

```bash
docker compose up -d prefect-worker
```

Stop it with `docker compose stop prefect-worker` if you need to run a one-off CLI worker manually.

With a worker running you can exercise the runtime helpers locally, e.g.:

```python
from freshbot.executors import invoke_tool, invoke_agent

plan = invoke_agent("freshbot-planner", payload={"objective": "summarise the latest ingest", "context": {}})
chat = invoke_tool("tool_qwen_chat", payload={"messages": [{"role": "user", "content": "ping"}]})
```
