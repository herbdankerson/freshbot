# Tool: materialize_goap_plan

Execute the GOAP planner to create tasks, ingest a plan document, and sync the
project DAG in one call. This wraps the `freshbot.agents.frameworks.goap`
runtime and the Codex task repository helpers.

- **Entrypoint:** `materialize_goap_plan`
- **Repository:** `codex_mcp.repository.materialize_goap_plan`
- **Prefect Deployments:** `freshbot_document_ingest/freshbot-document-ingest` (plan note),
  `task-sync-neo4j/freshbot-task-graph-sync` (triggered via DAG sync)

## Parameters

| Name | Type | Required | Description |
| --- | --- | --- | --- |
| `project_key` | `string` | Yes | Project that will receive the generated tasks |
| `goal` | `string \| object` | Yes | Desired end state for the planner (free text or structured payload) |
| `world_state` | `object` | No | Current facts exposed to the planner |
| `actions` | `object[]` | No | Optional action definitions to send to the planner |
| `toolbox` | `object[]` | No | Optional tool descriptors forwarded to the planner |
| `metadata` | `object` | No | Additional planner hints (stored with each task) |
| `agent_name` | `string` | No | Planner agent to invoke (defaults to `freshbot-goap-planner`) |
| `task_prefix` | `string` | No | Override for generated task keys (defaults to `project_key`) |

## Behaviour

1. Calls the GOAP planner service and normalises the returned steps.
2. Creates one task per step using `add_task`, embedding GOAP metadata.
3. Writes a Markdown plan note under `intellibot/project-docs/goap/plans/` and ingests it
   with flags `["is_plan", "is_note", "is_dev"]`.
4. Builds the dependency edges and invokes `upload_project_plan` to refresh ParadeDB and Neo4j.
5. Records a `planning_run` activity entry on each task and updates the project metadata with
   the last plan run ID/agent.

The response includes the generated plan document path/ID, DAG document path,
and a summary of the created tasks.

## Example

```json
{
  "name": "materialize_goap_plan",
  "arguments": {
    "project_key": "PROJECT-GOAP",
    "goal": "Ship release 1.2",
    "world_state": {"env": "staging"},
    "metadata": {"priority": "critical"}
  }
}
```
