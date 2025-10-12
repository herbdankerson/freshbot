# Freshbot

Freshbot now hosts the ParadeDB-backed runtime, registry tooling, and ingestion helpers that used to sit inside the Intellibot monolith. Use this repository for database-sourced agent/tool execution, registry loaders, and related planning docs; legacy file-config workflows stay in `intellibot/`.

## Local installation

Install the package in editable mode so compose containers (or virtual environments) can import `freshbot` without manual copies:

```bash
pip install -e .[dev]
```

The repo uses a `src/` layout and the new `pyproject.toml` matches the Intellibot packaging so both projects can coexist inside the same environment.
