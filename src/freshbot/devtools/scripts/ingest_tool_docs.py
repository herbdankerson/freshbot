"""Ingest tool README markdown into the knowledge base with tool metadata."""

from __future__ import annotations

import argparse
import logging
import os
import sys
import types
from pathlib import Path
from typing import Iterable

CURRENT_PATH = Path(__file__).resolve()
ROOT_CANDIDATES = [CURRENT_PATH.parents[5]]
try:
    ROOT_CANDIDATES.append(ROOT_CANDIDATES[0].parent)
except IndexError:
    pass
for candidate in ROOT_CANDIDATES:
    if candidate is not None and (candidate / "intellibot").exists():
        REPO_ROOT = candidate
        break
else:  # pragma: no cover - defensive path resolution
    REPO_ROOT = CURRENT_PATH.parents[5]

SRC_PACKAGE_PATHS = [
    REPO_ROOT / "freshbot" / "src",
    REPO_ROOT / "intellibot" / "src",
]
for path in SRC_PACKAGE_PATHS:
    sys.path.insert(0, str(path))

if "src" not in sys.modules:
    src_module = types.ModuleType("src")
    src_module.__path__ = [str(path) for path in SRC_PACKAGE_PATHS]  # type: ignore[attr-defined]
    sys.modules["src"] = src_module

from freshbot.pipeline.project_ingest import ingest_project_docs

LOGGER = logging.getLogger("freshbot.devtools.ingest_tool_docs")


def _tool_doc_root() -> Path:
    env_root = (os.environ.get("TOOL_DOC_ROOT") or "").strip()
    candidates = []
    if env_root:
        candidates.append(Path(env_root).expanduser())
    candidates.append(REPO_ROOT / "intellibot" / "project-docs" / "tools")
    candidates.append(REPO_ROOT / "freshbot" / "project-docs" / "tools")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Tool documentation directory not found in known locations.")


def _discover_docs(include: Iterable[str] | None = None) -> Iterable[Path]:
    doc_root = _tool_doc_root()
    if not doc_root.exists():
        raise FileNotFoundError(f"Tool documentation directory missing: {doc_root}")
    allow = {name.strip() for name in include or [] if name.strip()}
    for path in sorted(doc_root.glob("*.md")):
        slug = path.stem
        if allow and slug not in allow:
            continue
        yield slug, path


def _ingest_doc(slug: str, path: Path, dry_run: bool = False) -> None:
    LOGGER.info("Ingesting tool doc %s from %s", slug, path)
    if dry_run:
        return
    extra_metadata = {
        "extra": {
            "tool_slugs": [slug],
            "doc_type": "tool",
            "tool_doc": True,
        },
        "category": "tool",
        "source_tags": ["tooling", "codex"],
    }
    ingest_project_docs(
        path,
        display_name=f"Tool Doc: {slug}",
        extra_metadata=extra_metadata,
        source_root=_tool_doc_root(),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest tool documentation into the KB.")
    parser.add_argument(
        "--slug",
        action="append",
        help="Specific tool slug(s) to ingest (default: ingest all tool docs).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only list the documents that would be ingested.",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    any_docs = False
    for slug, path in _discover_docs(args.slug):
        any_docs = True
        _ingest_doc(slug, path, dry_run=args.dry_run)
    if not any_docs:
        LOGGER.warning("No tool documentation found for ingestion.")


if __name__ == "__main__":
    main()
