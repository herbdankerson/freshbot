"""Monte-Carlo Tree Search inspired agent type."""

from __future__ import annotations

import json
from typing import Any, Dict, Mapping, Optional

from .base import AgentExecutionError, assemble_messages, invoke_llm, load_runtime


def _parse_tree(content: Optional[str]) -> Dict[str, Any]:
    if not content:
        return {"error": "empty response"}
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return {"raw_text": content}


def run_deep_thinking_agent(
    *,
    objective: str,
    context: Optional[Mapping[str, Any]] = None,
    branching_factor: int = 3,
    rollouts: int = 6,
    max_depth: int = 3,
    agent_name: str = "freshbot-deep-thinking",
    include_toolbox: bool = True,
    temperature: Optional[float] = None,
    gateway_alias: Optional[str] = None,
) -> Dict[str, Any]:
    """Run a deep-thinking pass that approximates MCTS style exploration."""

    if not objective:
        raise AgentExecutionError("Deep thinking agent requires a non-empty objective.")
    if branching_factor < 1 or rollouts < 1 or max_depth < 1:
        raise AgentExecutionError("branching_factor, rollouts, and max_depth must be positive integers.")

    runtime = load_runtime(agent_name)
    payload = {
        "objective": objective,
        "context": context or {},
        "branching_factor": branching_factor,
        "rollouts": rollouts,
        "max_depth": max_depth,
        "format": {
            "tree": "list of nodes: {id, depth, thought, action, value_estimate, parent_id}",
            "best_path": "ordered list of node ids describing the recommended strategy",
            "final_answer": "concise synthesis of the selected path",
        },
        "guidance": [
            "Simulate multiple reasoning rollouts before committing to a plan.",
            "Use the provided tools when they improve evidence gathering.",
            "Explain why the best path was selected.",
        ],
    }

    messages = [
        {
            "role": "user",
            "content": json.dumps(payload, ensure_ascii=False),
        }
    ]
    assembled = assemble_messages(
        messages=messages,
        runtime=runtime,
        context={"mode": "mcts"},
        include_toolbox=include_toolbox,
        extra_tool_descriptions=[
            "Describe exploration vs exploitation at each depth.",
            "Return JSON matching the requested format.",
        ],
    )
    response = invoke_llm(
        runtime,
        messages=assembled,
        temperature=temperature,
        response_format={"type": "json_object"},
        gateway_alias=gateway_alias,
    )
    parsed = _parse_tree(response.get("content"))
    return {
        "agent": runtime.agent.name,
        "objective": objective,
        "tree": parsed.get("tree", []),
        "best_path": parsed.get("best_path", []),
        "final_answer": parsed.get("final_answer") or parsed.get("raw_text"),
        "raw": response,
    }


__all__ = ["run_deep_thinking_agent"]
