# Documentation Guide

This guide tracks the major documentation assets across the Freshbot and Intellibot projects so you can quickly find reference material after the upcoming Codex CLI upgrade.

## Quick Table of Contents

| Area | Document | Path | Summary |
| --- | --- | --- | --- |
| Freshbot | Project Architecture | `project-docs/project-architecture.md` | High-level runtime layout, registry description, and now an MCP-services section pointing at `codex-mcp`.
| Freshbot | Project Ingestion | `project-docs/project-ingestion.md` | ParadeDB namespace snapshots, ingestion flow notes, and container-only run instructions.
| Freshbot | Codex MCP | `project-docs/codex-mcp.md` | Service summary, exposed tools, and editing workflow for the new FastMCP server.
| Freshbot | Documentation Guide | `project-docs/documentation-guide.md` | (This file) master index for docs.
| Intellibot | Project Overview | `intellibot/project-docs/project.md` | Legacy architecture outline, agent roles, and service inventory.
| Intellibot | FastMCP Playbook | `intellibot/project-docs/fastMCP.md` | How older FastMCP helpers were staged and exercised.
| Intellibot | Runtime Config | `intellibot/project-docs/runtime-config.md` | Environment variables, service URLs, and configuration expectations for the original stack.
| Intellibot | Ingestion Flow | `intellibot/project-docs/ingestion-flow.md` | Step-by-step pipeline walkthrough plus SQL snippets.
| Intellibot | Qwen Notes | `intellibot/project-docs/qwen.md` | Gateway quirks and tuning guidance for the Qwen proxies.
| Intellibot | SearXNG Notes | `intellibot/project-docs/searxng.md` | Mapping between SearXNG engines, planning notes, and output schema captures.
| Intellibot | Task Log | `intellibot/project-docs/task.md` | Rolling task journal from earlier sessions.
| Intellibot | Instructions | `intellibot/project-docs/instructions.md` | Prior directives for the legacy pipeline; kept for historical context.
| Intellibot | Scratchpad | `intellibot/project-docs/planning-scratchpad-DO-NOT-READ-OR-USE.md` | Archived brainstorming (marked as do-not-use).

## File Tree (Docs Focus)

```
freshbot/
  project-docs/
    project-architecture.md
    project-ingestion.md
    codex-mcp.md
    documentation-guide.md  ← you are here
intellibot/
  project-docs/
    fastMCP.md
    ingestion-flow.md
    instructions.md
    planning-scratchpad-DO-NOT-READ-OR-USE.md
    project.md
    qwen.md
    runtime-config.md
    searxng.md
    task.md
```

## Notes

- Freshbot doc set is now the source of truth for the modular architecture and new MCP services.
- Intellibot docs remain valuable for legacy pipelines or when you need historical context; check this guide for quick summaries before diving into the longer files.
- When we upgrade the Codex CLI (see commands below), re-run `codex mcp list` to confirm the `freshbot_codex` server appears; the instructions for that service live in `project-docs/codex-mcp.md`.
