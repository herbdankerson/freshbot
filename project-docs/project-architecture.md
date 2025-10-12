# Freshbot Project Architecture

## Runtime Concept
- Registry-driven execution: agents, models, and tools live in ParadeDB under the `cfg` schema. Each row binds runtime Python entrypoints to configurable metadata (prompts, defaults, scopes) and is loaded from versioned YAML in `src/freshbot/registry/`.
- Agents are "base" compositions that wrap a thinking framework. Planner-style agents (Tree-of-Thought, Monte-Carlo Tree Search, etc.) simply swap the entrypoint referenced in `cfg.agents.params.entrypoint` and adjust defaults. Tool-capable agents discover connectors and tool definitions via `cfg.agent_tools` bindings at runtime.
- Connectors encode access to external systems (databases, LLM gateways, graph stores) and are stored as `purpose = 'connector'` rows in `cfg.models`. Tools can depend on connectors, and agents inherit tool overrides for payload shaping.
- Executors mediate side effects. The Prefect executor (`freshbot.executors.prefect:execute_flow`) launches orchestration flows, while additional executors can be added under `src/freshbot/executors/`.

## Source Layout (target)
```text
freshbot/
├── docs/
├── project-docs/
│   └── project-architecture.md  # this document
├── searxng/
├── src/freshbot/
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── base.py               # shared AgentConfig + runtime helpers
│   │   ├── loader.py             # fetch agent configs from registry/DB
│   │   └── frameworks/
│   │       ├── __init__.py
│   │       ├── vanilla.py        # simple message loop
│   │       ├── tot.py            # Tree-of-Thought planner
│   │       └── mcts.py           # MCTS-style planner (future)
│   ├── connectors/
│   ├── db/
│   ├── executors/
│   ├── flows/
│   ├── gateways/
│   ├── pipeline/
│   └── registry/
└── tests/
    └── ...
```
- `src/freshbot/agents/` (to be created) will host the base agent runtime plus optional thinking frameworks. Each framework exposes a callable that the registry points to via `params.entrypoint`.
- `src/freshbot/registry/` already mirrors the database state; `devtools/registry_loader.py` keeps DB and YAML in sync.
- Tests dedicated to agent plumbing will live under `tests/agents/`, mirroring the module layout once implementations land.

## Registry Tables & Live State
### `cfg.agents`
Key columns: `name`, `type` (`base`, `planner`, `custom`), `model_alias`, JSON `params` (entrypoint, defaults), `tools_profile`, `db_scope`.

Freshbot-specific rows (live database snapshot 2025-10-11):
| name | type | model_alias | entrypoint | notes |
|------|------|-------------|-----------|-------|
| `freshbot-planner` | planner | planner | `freshbot.flows.agents:planner_agent_flow` | Planner orchestrating tool usage. |
| `freshbot-tool-user` | base | planner | `freshbot.flows.agents:tool_user_agent_flow` | Executes plan steps with tool bindings. |
| `freshbot-responder` | base | responder | `freshbot.flows.agents:responder_agent_flow` | Crafts final replies. |
| `freshbot-auditor` | base | cheap-worker | `freshbot.flows.agents:auditor_agent_flow` | Runs compliance/quality pass. |
| `freshbot-code-executor` | base | *(null)* | `freshbot.flows.agents:code_executor_agent_flow` | Handles python execution, no model alias. |
| `freshbot-custom-agent` | custom | planner | `freshbot.flows.agents:custom_agent_flow` | Template for bespoke payloads. |

Legacy rows (`planner`, `responder`, etc.) from the pre-Freshbot pipeline are still present but inert for the modular architecture. They can remain until the migration is officially complete.

### `cfg.agent_tools`
Binds agents to the tools they can call plus override payloads. Current highlights:
- `freshbot-planner` → Prefect executor (default deployment `freshbot-heartbeat-deployment`), connector catalog, search helpers (`tool_search_agents`, `tool_search_tools`, `tool_kb_search` with `limit=5`), and a registry snapshot tool.
- `freshbot-tool-user` → chat, KB search (`limit=10`), code embedding.
- `freshbot-responder` → chat tools with tuned temperatures.
- `freshbot-auditor` → deterministic chat tool for audits.
- `freshbot-custom-agent` → default chat tool.

### `cfg.models`
Columns cover aliases, purpose (`chat`, `connector`, `embedding`, etc.), endpoint, default params, and provider metadata.

Connector inventory already aligned with Freshbot:
| alias | endpoint | description |
|-------|----------|-------------|
| `connector_paradedb` | `postgresql://agent:agentpass@paradedb:5432/agentdb` | ParadeDB warehouse connector (database scope access). |
| `connector_qwen_openai` | `http://qwen-openai-proxy:5000` | Qwen OpenAI-compatible gateway (primary reasoning LLM). |
| `connector_gemini_litellm` | `http://litellm:4000` | Gemini via LiteLLM router for fallback reasoning. |
| `connector_neo4j` | `bolt://neo4j:7687` | Neo4j graph connector. |
| `connector_ollama_code_embedding` | `http://ollama:11434/api/embeddings` | Code embedding via Ollama (manutic/nomic-embed-code). |

Embedding and chat model aliases (`emb-code`, `planner`, `responder`, etc.) continue to exist for backwards compatibility; Freshbot agents reference the connector aliases via `params.gateway_alias`.

### `cfg.tools`
Important columns: `slug`, `kind` (`native`, `http`, `mcp`), `manifest_or_ref` (Python callable or remote manifest), JSON `default_params`.

Freshbot toolchain (all enabled):
| slug | callable | role |
|------|----------|------|
| `prefect_flow_executor` | `freshbot.executors.prefect:execute_flow` | Launch Prefect deployments on demand. |
| `connector_catalog` | `freshbot.connectors.catalog:lookup` | Expose available connectors and scopes. |
| `tool_qwen_chat` | `freshbot.flows.tools:qwen_chat_tool` | Primary chat interface. |
| `tool_gemini_chat` | `freshbot.flows.tools:gemini_chat_tool` | Gemini fallback chat. |
| `tool_code_embed` | `freshbot.flows.tools:code_embedding_tool` | Generate code embeddings. |
| `tool_kb_search` | `freshbot.flows.tools:kb_search_tool` | Hybrid ParadeDB KB search. |
| `tool_search_agents` | `freshbot.flows.tools:search_agents_tool` | Discover agent registry entries. |
| `tool_search_tools` | `freshbot.flows.tools:search_tools_tool` | Discover tool registry entries. |
| `tool_registry_snapshot` | `freshbot.flows.tools:agent_registry_snapshot_tool` | Debugging view of registry content. |

Legacy HTTP/MCP tools (`search-toolbox`, `docling`, `neo4j_*`, etc.) remain from Intellibot. They are not referenced by the Freshbot agents and can be archived later.

## Alignment & Seeding Status
- ParadeDB already contains the Freshbot agent/tool/model rows; no additional seeding is required before implementing the new agent frameworks.
- Registry YAML under `src/freshbot/registry/` mirrors the active Freshbot entries. Running `python -m freshbot.devtools.registry_loader --apply` will reconcile changes if YAML is updated.
- The presence of legacy rows does not interfere with Freshbot because planner/tool-user agents only reference the `freshbot-*` namespace. Any cleanup can happen once documentation and migration plans are finalised.

## Implementation Checklist
1. Stand up `src/freshbot/agents/` with a base class handling:
   - Loading agent configuration from ParadeDB (via `registry.snapshot.get_registry()`).
   - Delegating to framework-specific planners (vanilla, ToT, MCTS).
   - Tool routing using `freshbot.executors.runtime.invoke_tool`.
2. Provide framework shims (e.g., `tot.py`) that expose callables referenced by `cfg.agents.params.entrypoint`.
3. Extend tests (`tests/agents/`) to cover registry resolution, planner branching, and tool invocation overrides.
4. Keep connectors discoverable through the `connector_catalog` tool so planning frameworks can dynamically request access to sources.
5. For ingestion, use the namespace wrappers (`freshbot.pipeline.ingest_project_code/docs`) so metadata stays aligned with the ParadeDB schemas.

With this structure Freshbot stays self-contained: all runtime decisions flow from database configuration, and code changes live inside the new `src/freshbot/agents/` module family.

## Project Namespaces
- `project_code.*` – code-first mirror of the KB schema (documents, chunks, `vector(1024)` embeddings) populated by running `freshbot_document_ingest(..., target_namespace='project_code')` and pointing `target_entries` at `project_code.entries`.
- `project_docs.*` – documentation mirror using the same layout; triggered with `target_namespace='project_docs'` and `target_entries='project_docs.entries'`.

Both namespaces reuse `kb.embedding_spaces` and now maintain their own `entries` tables (created with `CREATE TABLE project_code.entries (LIKE kb.entries INCLUDING ALL)` and the equivalent for documentation) so agents can search code/doc material without polluting the main KB. The ingestion flow automatically builds hnsw indexes inside each schema the first time embeddings are written.
