# Tool: ingest_document

## Overview

The `ingest_document` MCP tool uploads a local file into the knowledge base using the
Freshbot ingestion pipeline. It mirrors the behaviour of the Prefect deployment used
for automated ingest runs, so classification, summarisation, embeddings, and metadata
tagging all execute exactly as they do for batch ETL jobs.

## Parameters

| Name | Type | Required | Description |
| --- | --- | --- | --- |
| `path` | string | yes | Absolute or relative path to the file that should be ingested. Must be visible to the Prefect worker (usually within `/workspace`). |
| `display_name` | string | no | Optional human-friendly label for the document; defaults to the filename. |
| `flags` | array[string] | no | Additional ingest flags (e.g., `is_note`, `workspace_file`). These augment the defaults stored in `cfg.flags`. |
| `flag_overrides` | object | no | Explicit flag assignments (`{"is_note": true}`) that take precedence over defaults and list-based flags. |
| `references` | array[string] | no | UUIDs that the document should reference. The ingestion pipeline records them in document and chunk metadata so downstream searches surface linked notes. |
| `metadata` | object | no | Extra metadata merged into the ingest payload (e.g., `{"document_kind": "design"}`). |
| `is_dev` | boolean | no | Override the `is_dev` target; defaults to the namespace binding’s value. |
| `wait_for_completion` | boolean | no | When `true` (default) the tool blocks until the Prefect flow completes and returns resolved document identifiers. If `false`, the flow is scheduled and the response includes the `flow_run_id` so callers can poll for completion. |
| `source_type` | string | no | Optional override for the ingest source type; defaults to `code_document` or `document` depending on file extension. |

## Behaviour

- Chooses the appropriate namespace binding (`code` vs `docs`) using the same
  extension heuristics as the ingestion pipeline.
- Loads all flag definitions from `cfg.flags`, applies list/override inputs, and records
  the resolved values (including automatically forcing `workspace_file` for files inside
  the repository).
- Normalises supplied references, merges them with any metadata-specified references,
  and stores the union in both document-level and chunk-level metadata.
- Calls the Prefect deployment `freshbot_document_ingest/freshbot-document-ingest` via
  the API server, respecting the `wait_for_completion` option.
- Returns the resolved `document_id`, `ingest_item_id`, flow identifiers, the final
  metadata payload sent to Prefect, and the resolved flag/reference sets so planners can
  immediately follow up with search or update operations.

## Notes

- The tool does **not** modify the source file. For edits, update the file on disk and
  re-run ingestion or use `update_chunk` for targeted text patches.
- Large batches can be processed by toggling `wait_for_completion=false` and tracking
  the resulting `flow_run_id` via the Prefect flow catalogue tools.
