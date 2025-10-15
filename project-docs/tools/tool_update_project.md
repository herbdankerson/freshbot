# Tool: update_project

## Overview

The `update_project` tool edits an existing project's metadata: display name,
description, status, priority, tags, and planning document references. Use it to keep
project records aligned with the latest DAG and plan files without touching the
database manually.

## Parameters

| Name | Type | Required | Description |
| --- | --- | --- | --- |
| `project_key` | string | yes | Project slug (case-insensitive) to update. |
| `name` | string | no | New project display name. |
| `description` | string | no | Updated short description. |
| `status` | string | no | New status label (`active`, `paused`, `archived`, etc.). |
| `priority` | string | no | Optional priority marker for dashboards. |
| `tags` | array[string] | no | Replacement tag list. Pass an empty array to clear tags. |
| `metadata` | object | no | JSON fragment merged into existing metadata. Keys overwrite previous values. |
| `plan_document` | string | no | Replace the stored plan document path (metadata `plan_document`). |
| `dag_document` | string | no | Replace the stored DAG path (metadata `dag_document`). |

## Behaviour

- Normalises the project key and validates that the project exists before applying
  updates.
- Only fields supplied are touched; omitted parameters leave existing values unchanged.
- Metadata updates are merged using `jsonb ||`, so you can update individual keys without
  rewriting the entire metadata payload.
- Returns the refreshed project payload (via `fetch_project`) so clients can confirm the
  change along with up-to-date task counts.

## Notes

- Use this tool whenever project planning files move or when changing project level
  status/priority—avoid direct SQL updates to keep auditability intact.
- Tags are treated as a replacement list; pass the full desired set on each call.
