"""Run smoke tests against the core tool flows."""

from __future__ import annotations

import argparse
import json
import logging
from typing import Any, Dict

from freshbot.flows.tools import (
    agent_catalog_tool,
    kb_search_tool,
    prefect_flow_catalog_tool,
    search_agents_tool,
    search_tools_tool,
    tool_manifest_fetch_tool,
    graph_capabilities_map_tool,
)

LOGGER = logging.getLogger("freshbot.devtools.tool_smoke_tests")


def _run_flow(flow, *args, **kwargs) -> Any:
    """Execute a Prefect flow function in local mode via its `.fn` attribute."""

    try:
        fn = flow.fn
    except AttributeError:  # pragma: no cover - defensive
        fn = flow
    return fn(*args, **kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test core tool flows.")
    parser.add_argument(
        "--tool-slug",
        default="tool_kb_search",
        help="Tool slug used for manifest checks (default: tool_kb_search).",
    )
    parser.add_argument(
        "--search-query",
        default="prefect",
        help="Query string for search-based tests (default: 'prefect').",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Maximum number of records retrieved during tests (default: 5).",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    results: Dict[str, Any] = {}
    LOGGER.info("Running Prefect flow catalog test")
    results["prefect_flow_catalog"] = _run_flow(
        prefect_flow_catalog_tool,
        include_tags=True,
        search=None,
    )

    LOGGER.info("Running tool manifest fetch test for %s", args.tool_slug)
    results["tool_manifest"] = _run_flow(
        tool_manifest_fetch_tool,
        tool_slug=args.tool_slug,
        include_docs=True,
    )

    LOGGER.info("Running agent catalog test")
    results["agent_catalog"] = _run_flow(
        agent_catalog_tool,
        include_disabled=False,
    )

    LOGGER.info("Running tool search test")
    results["tool_search"] = _run_flow(
        search_tools_tool,
        query=args.search_query,
        limit=args.limit,
        include_disabled=False,
    )

    LOGGER.info("Running agent search test")
    results["agent_search"] = _run_flow(
        search_agents_tool,
        query=args.search_query,
        limit=args.limit,
        include_disabled=False,
    )

    LOGGER.info("Running KB search test")
    results["kb_search"] = _run_flow(
        kb_search_tool,
        query=args.search_query,
        limit=args.limit,
    )

    LOGGER.info("Running graph capabilities map test")
    try:
        results["graph_map"] = _run_flow(
            graph_capabilities_map_tool,
            limit=args.limit,
            include_nodes=True,
        )
    except Exception as exc:  # pragma: no cover - provide visibility during dev
        LOGGER.exception("Graph capabilities map failed")
        results["graph_map"] = {"error": str(exc)}

    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

