# Freshbot Planning Outline

## Goals
- Centralize search and research tooling for Freshbot’s agent workflows.
- Prototype SearXNG interface improvements without touching production sources.

## Immediate Tasks
1. Inventory existing integrations (SearXNG, Startpage proxy, custom APIs).
2. Define schema for search results table (fields, sorting behavior, metadata badges).
3. Identify Google-access strategy (Custom Search API vs. Startpage proxy).

## Open Questions
- Do we host a dedicated SearXNG instance for Freshbot or reuse the shared container?
- What authentication/limiting do we need for production usage?
- Which downstream consumers (UI, LLM agents) will rely on this repo’s artifacts?

## Next Steps
- Draft architecture diagram for Freshbot search modules.
- Collect constraints from product/design stakeholders.
- Plan milestone backlog and link to tracking issues once the remote is provisioned.
