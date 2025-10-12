"""Sequential reasoning agent type."""

from __future__ import annotations

import json
from typing import Any, Dict, Mapping, Optional, Sequence

from .base import AgentExecutionError, assemble_messages, invoke_llm, load_runtime


def _parse_response(content: Optional[str]) -> Dict[str, Any]:
    if not content:
        return {"error": "empty response"}
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return {"raw_text": content}


def run_thinking_agent(
    *,
    objective: str,
    context: Optional[Mapping[str, Any]] = None,
    prior_insights: Optional[Sequence[str]] = None,
    max_steps: int = 5,
    agent_name: str = "freshbot-thinking-agent",
    include_toolbox: bool = True,
    temperature: Optional[float] = None,
    gateway_alias: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute a sequential thinking pass that returns structured steps."""

    if not objective:
        raise AgentExecutionError("Thinking agent requires a non-empty objective.")

    runtime = load_runtime(agent_name)
    payload = {
        "objective": objective,
        "context": context or {},
        "max_steps": max_steps,
        "prior_insights": list(prior_insights or []),
        "format": {
            "steps": "list of ordered reasoning steps with {step, thought, optional_tool, expected_outcome}",
            "final_answer": "concise resolution of the objective",
        },
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
        context={"mode": "sequential-planning"},
        include_toolbox=include_toolbox,
        extra_tool_descriptions=[
            "When tools are listed, reference them by slug under 'optional_tool'.",
            "Always reason step-by-step before final_answer.",
        ],
    )
    response = invoke_llm(
        runtime,
        messages=assembled,
        temperature=temperature,
        response_format={"type": "json_object"},
        gateway_alias=gateway_alias,
    )
    parsed = _parse_response(response.get("content"))
    return {
        "agent": runtime.agent.name,
        "objective": objective,
        "steps": parsed.get("steps", []),
        "final_answer": parsed.get("final_answer") or parsed.get("raw_text"),
        "raw": response,
    }


__all__ = ["run_thinking_agent"]
