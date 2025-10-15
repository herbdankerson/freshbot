# Tool: set_task_dependencies

## Overview

The `set_task_dependencies` tool replaces a task’s dependency list with an explicit
ordering. It rewrites the DAG edges, logs the update, and regenerates the task document
so downstream planners see the new prerequisites immediately.

## Parameters

| Name | Type | Required | Description |
| --- | --- | --- | --- |
| `task_key` | string | yes | Task identifier (case-insensitive) to update. |
| `dependencies` | array[string] | yes | Ordered list of task keys that must complete before the target task. Duplicates and self-references are ignored. |

## Behaviour

- Normalises the task key and dependency keys to uppercase, removing duplicates and
  self-dependencies.
- Validates that every dependency task exists before applying changes.
- Clears existing DAG edges for the task, then inserts the new dependency edges.
- Logs a `task_dependencies_updated` activity entry detailing the previous and new
  dependency sets.
- Refreshes the task’s markdown via the ingest pipeline so the dependency list in the
  documentation stays current.
- Returns the refreshed task payload (matching `fetch_task`) for immediate inspection.

## Notes

- To remove all dependencies, pass an empty list (`[]`); the tool deletes the edges and
  marks the task unblocked if no other constraints exist.
- Combine with `assign_tasks_to_project` when reorganising large plans: move tasks first,
  then update dependencies to reflect the new project DAG.
