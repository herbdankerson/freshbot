# Tool: assign_tasks_to_project

## Overview

The `assign_tasks_to_project` tool bulk-moves tasks into a project, logging the change
and regenerating each task's documentation. Use it to consolidate tasks under a new or
renamed project without hand-editing numerous records.

## Parameters

| Name | Type | Required | Description |
| --- | --- | --- | --- |
| `project_key` | string | yes | Target project slug (case-insensitive). Created on the fly if it does not yet exist. |
| `task_keys` | array[string] | yes | Task keys to reassign. Each is normalised to uppercase. |
| `project_name` | string | no | Display name to use when auto-creating the target project. Ignored if the project already exists. |

## Behaviour

- Ensures the destination project exists (creating it with the supplied `project_name`
  when necessary).
- Moves each task by updating `task.tasks.project_id`, recording the previous project
  key in metadata (`previous_project_key`) for auditability.
- Logs a `task_reassigned` activity entry per task and immediately refreshes the task's
  markdown/documentation via the ingest pipeline so the project heading is correct.
- Returns both the refreshed project snapshot (`fetch_project`) and the updated task
  payloads for confirmation.

## Notes

- The tool preserves existing dependencies; use `set_task_dependencies` afterwards if a
  new DAG ordering is required.
- When moving a large set of tasks, consider batching calls to keep Prefect ingest runs
  manageable.
