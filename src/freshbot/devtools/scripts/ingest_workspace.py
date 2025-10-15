"""CLI helper to ingest the active workspace into the knowledge base.

This utility walks one or more root directories, filters out build artefacts,
and submits each file to the configured Freshbot ingestion pipeline.  It is
primarily meant for developer environments so the MCP tooling can vector-search
the live codebase without manual uploads.

Key behaviours:
* Automatically tags every artefact with ``document_kind=workspace_file`` and
  ``is_editable=True`` metadata so downstream tools can scope searches.
* Distinguishes code vs documentation via ``is_code_extension`` so ctags
  chunking only runs for source files.
* Respects ``--dry-run`` and ``--limit`` flags to sanity-check what would be
  ingested before committing to a long run.
* Skips obvious non-text artefacts (images, archives, compiled meshes) and
  ignores directories such as ``.git`` or ``__pycache__``.

Usage examples:

    poetry run python -m freshbot.devtools.scripts.ingest_workspace \\
        --root freshbot/src --root mcp/codex-mcp/src --root intellibot/src

    PYTHONPATH=freshbot/src:intellibot:intellibot/src \\
        FRESHBOT_DATABASE_URL=postgresql+psycopg://agent:agentpass@localhost:5432/agentdb \\
        python -m freshbot.devtools.scripts.ingest_workspace --dry-run
"""

from __future__ import annotations

import argparse
import os
import sys
import types
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable, Iterable, Iterator, List, Sequence

# Standard skip lists tuned for repo-style workspaces.
SKIP_DIR_NAMES = {
    ".git",
    ".idea",
    ".vscode",
    ".trash",
    "__pycache__",
    "node_modules",
    "dist",
    "build",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "venv",
    ".tox",
}

def _bootstrap_sys_path() -> None:
    """Ensure editable checkouts are importable without virtualenv tweaks."""

    candidates = [
        Path("freshbot/src"),
        Path("intellibot"),
        Path("intellibot/src"),
        Path("mcp/codex-mcp/src"),
        Path("agent-framework/src"),
        Path("/app/src"),
        Path("/app"),
        Path("/workspace/freshbot/src"),
        Path("/workspace/intellibot/src"),
        Path("/workspace/intellibot"),
        Path("/etl"),
    ]
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.exists():
            if str(resolved) not in sys.path:
                sys.path.append(str(resolved))


@dataclass(frozen=True)
class IngestTarget:
    path: Path
    is_code: bool


@lru_cache(maxsize=1)
def _ingest_functions() -> tuple[Callable, Callable]:
    """Import ingestion helpers lazily after path bootstrap."""

    import importlib

    from freshbot.flows.ingest import steps as ingest_steps

    if getattr(ingest_steps, "intake_tasks", None) is None:
        ingest_steps = importlib.reload(ingest_steps)

    from freshbot.pipeline.project_ingest import (
        ingest_project_code,
        ingest_project_docs,
    )

    return ingest_project_code, ingest_project_docs


def _ensure_namespace_package() -> None:
    """Expose ``src.*`` namespace packages expected by ETL modules."""

    candidates = [
        Path("freshbot/src").resolve(),
        Path("intellibot/src").resolve(),
        Path("mcp/codex-mcp/src").resolve(),
        Path("/workspace/freshbot/src"),
        Path("/workspace/intellibot/src"),
        Path("/app/src"),
    ]
    namespace_paths = [str(path) for path in candidates if path.exists()]
    if not namespace_paths:
        return

    existing = sys.modules.get("src")
    if existing is None:
        src_module = types.ModuleType("src")
        src_module.__path__ = []  # type: ignore[attr-defined]
        sys.modules["src"] = src_module
    else:
        src_module = existing
        if not hasattr(src_module, "__path__"):
            src_module.__path__ = []  # type: ignore[attr-defined]

    for path in namespace_paths:
        if path not in src_module.__path__:  # type: ignore[attr-defined]
            src_module.__path__.append(path)  # type: ignore[attr-defined]


SKIP_FILE_SUFFIXES = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".ico",
    ".svg",
    ".mp4",
    ".mp3",
    ".wav",
    ".ogg",
    ".zip",
    ".tar",
    ".gz",
    ".bz2",
    ".xz",
    ".7z",
    ".pdf",
    ".pyc",
    ".so",
    ".dll",
    ".dylib",
    ".pkl",
    ".bin",
}

# Prime namespace packages before importing Freshbot modules that expect them.
_bootstrap_sys_path()
_ensure_namespace_package()

from freshbot.pipeline.classification import is_code_extension


# Default workspace roots relative to the script repository.
DEFAULT_ROOTS = [
    Path("."),
]


def _should_skip(path: Path) -> bool:
    if path.suffix.lower() in SKIP_FILE_SUFFIXES:
        return True
    if path.name.startswith(".") and path.suffix == "":
        # Hidden dotfiles like .DS_Store
        return True
    return False


def _iter_files(roots: Sequence[Path]) -> Iterator[Path]:
    for root in roots:
        resolved_root = root.resolve()
        if not resolved_root.exists():
            continue
        for current_dir, dirnames, filenames in os.walk(resolved_root):
            # Mutate dirnames in-place so os.walk skips ignored directories.
            dirnames[:] = [
                name
                for name in dirnames
                if name not in SKIP_DIR_NAMES
                and not name.startswith(".")
                and not name.endswith(".egg-info")
            ]
            for filename in filenames:
                path = Path(current_dir, filename)
                if any(part in SKIP_DIR_NAMES for part in path.parts):
                    continue
                if _should_skip(path):
                    continue
                yield path


def _build_targets(paths: Iterable[Path]) -> List[IngestTarget]:
    targets: List[IngestTarget] = []
    for path in paths:
        try:
            if path.stat().st_size == 0:
                print(f"[SKIP] empty file :: {path}")
                continue
        except FileNotFoundError:
            continue
        is_code = is_code_extension(path.name)
        targets.append(IngestTarget(path=path, is_code=is_code))
    return targets


def _ingest_target(target: IngestTarget, *, source_root: Path, dry_run: bool) -> None:
    extra_metadata = {
        "document_kind": "workspace_file",
        "is_editable": True,
        "workspace": {
            "relative_path": None,
        },
    }

    try:
        relative = target.path.resolve().relative_to(source_root)
    except ValueError:
        relative = target.path.name
    extra_metadata["workspace"]["relative_path"] = str(relative)

    if dry_run:
        kind = "code" if target.is_code else "doc"
        print(f"[DRY-RUN] {kind:4s} :: {target.path}")
        return

    binding_kwargs = {
        "display_name": str(target.path.name),
        "extra_metadata": extra_metadata,
        "source_root": source_root,
    }
    ingest_project_code, ingest_project_docs = _ingest_functions()

    if target.is_code:
        ingest_project_code(target.path, **binding_kwargs)
    else:
        ingest_project_docs(target.path, **binding_kwargs)
    print(f"[INGESTED] {target.path}")


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingest workspace files into the Freshbot knowledge base.",
    )
    parser.add_argument(
        "--root",
        action="append",
        dest="roots",
        help="Directory to ingest (can be supplied multiple times). Defaults to project root.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of files to ingest.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List files that would be ingested without calling the pipeline.",
    )
    parser.add_argument(
        "--start-with",
        dest="start_with",
        help="Skip files lexically before the provided path (useful for resume).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])

    _bootstrap_sys_path()
    _ensure_namespace_package()

    roots = [Path(root) for root in (args.roots or DEFAULT_ROOTS)]
    workspace_root = Path(".").resolve()
    paths = sorted(_iter_files(roots), key=lambda p: str(p))
    if args.start_with:
        start_path = Path(args.start_with).resolve()
        paths = [path for path in paths if path.resolve() >= start_path]

    if args.limit is not None:
        paths = paths[: max(args.limit, 0)]

    targets = _build_targets(paths)
    print(f"Discovered {len(targets)} ingest targets.")

    for target in targets:
        _ingest_target(target, source_root=workspace_root, dry_run=args.dry_run)

    mode = "DRY-RUN" if args.dry_run else "INGEST"
    print(f"{mode} complete for {len(targets)} files.")
    return 0


if __name__ == "__main__":
    os.environ.setdefault(
        "FRESHBOT_DATABASE_URL",
        "postgresql+psycopg://agent:agentpass@paradedb:5432/agentdb",
    )
    sys.exit(main())
