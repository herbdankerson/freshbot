# Tool: create_project

## Overview

The `create_project` tool provisions a new project record inside the task manager,
including human-friendly metadata and references to planning documents. It is the
authoritative way to initialise a project before tasks are added or reassigned.

## Parameters

| Name | Type | Required | Description |
| --- | --- | --- | --- |
| `project_key` | string | yes | Unique slug for the project. Normalised to uppercase before insertion. |
| `name` | string | no | Display name shown in dashboards. Defaults to the normalised `project_key`. |
| `description` | string | no | Short explanation of the project's scope. Stored on the project record. |
| `status` | string | no | Initial status (`active`, `archived`, etc.). Defaults to `active`. |
| `priority` | string | no | Optional priority label for sorting. |
| `tags` | array[string] | no | Project tags (stored in `task.projects.tags`). |
| `metadata` | object | no | Additional JSON metadata to merge into the project record. |
| `plan_document` | string | no | Relative path to the authoritative project plan markdown. Stored in metadata as `plan_document`. |
| `dag_document` | string | no | Relative path to the DAG/ordering markdown. Stored in metadata as `dag_document`. |

## Behaviour

- Ensures the project key is unique; requests fail with a clear error if a project already
  exists for the provided key.
- Persists the new row in `task.projects` with populated metadata, tags, and planning
  document references.
- Automatically stamps metadata with `managed_via=codex-mcp` so downstream tooling can
  differentiate manually created projects from automated imports.
- Returns the newly created project via `fetch_project`, including task counts (all zero)
  and the stored planning references for immediate verification.

## Notes

- Use `create_project` before calling `assign_tasks_to_project` to keep planning metadata
  and task assignments in sync.
- To update planning documents or metadata later, call `update_project`.
