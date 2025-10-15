# Tool: fetch_project

## Overview

The `fetch_project` tool retrieves a single project's metadata, planning references, and
aggregated task counts. It is the quickest way to confirm project state before adding
tasks, updating DAGs, or planning work.

## Parameters

| Name | Type | Required | Description |
| --- | --- | --- | --- |
| `project_key` | string | yes | Project slug (case-insensitive) to fetch. |

## Behaviour

- Normalises the project key and loads the full project snapshot (`task.projects` row +
  metadata).
- Aggregates task counts (`total`, `complete`, `blocked`, `active`) from `task.tasks`
  for the specified project.
- Returns planning references (`plan_document`, `dag_document`) if they were set via
  `create_project` or `update_project`.

## Notes

- Use `fetch_project` prior to task reassignment or DAG uploads to validate that the
  project is active and to surface existing planning docs.
- For project listings across the workspace, continue using `list_projects`.
