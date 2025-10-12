"""Core helpers for Freshbot agent types."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence

from freshbot.gateways import chat_completion
from freshbot.registry import AgentRecord, RegistrySnapshot, ToolRecord, get_registry


@dataclass(frozen=True)
class ToolSummary:
    """Describes a tool available to an agent."""

    slug: str
    description: str
    notes: Optional[str]
    overrides: Dict[str, Any]


@dataclass(frozen=True)
class AgentRuntime:
    """Cached metadata needed to execute an agent call."""

    agent: AgentRecord
    gateway_alias: str
    system_prompt: Optional[str]
    default_temperature: Optional[float]
    default_max_tokens: Optional[int]
    tool_summaries: List[ToolSummary]

    def format_toolbox(self) -> str:
        if not self.tool_summaries:
            return "No registered tools for this agent."
        lines: List[str] = ["Registered tools:"]
        for tool in self.tool_summaries:
            overrides = ", ".join(f"{k}={v!r}" for k, v in tool.overrides.items())
            detail = f" ({overrides})" if overrides else ""
            notes = f" — {tool.notes}" if tool.notes else ""
            lines.append(f"- {tool.slug}: {tool.description}{detail}{notes}")
        return "\n".join(lines)


class AgentExecutionError(RuntimeError):
    """Raised when agent execution fails due to invalid configuration."""


def _resolve_gateway(agent: AgentRecord) -> str:
    params_gateway = None
    if isinstance(agent.params, Mapping):
        params_gateway = agent.params.get("gateway_alias")
    if params_gateway:
        return str(params_gateway)
    if agent.model_alias:
        return str(agent.model_alias)
    return "connector_qwen_openai"


def _resolve_defaults(agent: AgentRecord) -> Dict[str, Any]:
    defaults: Dict[str, Any] = {}
    if isinstance(agent.params, Mapping):
        defaults = dict(agent.params.get("defaults") or {})
    return defaults


def _collect_tool_summaries(snapshot: RegistrySnapshot, agent: AgentRecord) -> List[ToolSummary]:
    summaries: List[ToolSummary] = []
    for slug, overrides in agent.tool_bindings.items():
        tool: Optional[ToolRecord] = snapshot.tools.get(slug)
        if not tool:
            continue
        description = tool.default_params.get("description") if isinstance(tool.default_params, Mapping) else None
        summaries.append(
            ToolSummary(
                slug=slug,
                description=str(description or tool.notes or "No description supplied."),
                notes=tool.notes,
                overrides=dict(overrides or {}),
            )
        )
    return summaries


def load_runtime(agent_name: str) -> AgentRuntime:
    """Resolve registry metadata for ``agent_name``."""

    snapshot = get_registry()
    agent = snapshot.require_agent(agent_name)
    if not agent.enabled:
        raise AgentExecutionError(f"Agent '{agent_name}' is disabled.")
    defaults = _resolve_defaults(agent)
    tool_summaries = _collect_tool_summaries(snapshot, agent)
    return AgentRuntime(
        agent=agent,
        gateway_alias=_resolve_gateway(agent),
        system_prompt=agent.system_prompt,
        default_temperature=_coerce_optional_float(defaults.get("temperature")),
        default_max_tokens=_coerce_optional_int(defaults.get("max_tokens")),
        tool_summaries=tool_summaries,
    )


def _coerce_optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
        raise AgentExecutionError(f"Invalid float default: {value!r}") from exc


def _coerce_optional_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
        raise AgentExecutionError(f"Invalid integer default: {value!r}") from exc


def _ensure_messages(messages: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for item in messages:
        if not isinstance(item, Mapping):
            raise AgentExecutionError("Each message must be a mapping with role/content fields.")
        role = item.get("role")
        content = item.get("content")
        if role is None:
            raise AgentExecutionError(f"Message missing role: {item!r}")
        normalized.append({"role": str(role), "content": content})
    return normalized


def _build_context_message(context: Optional[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
    if not context:
        return None
    return {
        "role": "system",
        "content": json.dumps({"context": context}, ensure_ascii=False),
    }


def _build_tool_message(toolbox_text: str) -> Dict[str, Any]:
    return {
        "role": "system",
        "content": toolbox_text,
    }


def invoke_llm(
    runtime: AgentRuntime,
    *,
    messages: Sequence[Mapping[str, Any]],
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    response_format: Optional[Mapping[str, Any]] = None,
    extra_params: Optional[Mapping[str, Any]] = None,
    gateway_alias: Optional[str] = None,
) -> Dict[str, Any]:
    """Call the configured gateway and return the raw response."""

    payload_messages = _ensure_messages(messages)
    resolved_temperature = temperature if temperature is not None else runtime.default_temperature
    resolved_max_tokens = max_tokens if max_tokens is not None else runtime.default_max_tokens
    target_alias = gateway_alias or runtime.gateway_alias
    return chat_completion(
        payload_messages,
        gateway_alias=target_alias,
        temperature=resolved_temperature,
        max_tokens=resolved_max_tokens,
        response_format=response_format,
        extra_params=extra_params,
    )


def assemble_messages(
    *,
    messages: Sequence[Mapping[str, Any]],
    runtime: AgentRuntime,
    context: Optional[Mapping[str, Any]] = None,
    include_toolbox: bool = True,
    extra_tool_descriptions: Optional[Iterable[str]] = None,
) -> List[Dict[str, Any]]:
    """Attach optional context/tool annotations to the conversation."""

    assembled: List[Dict[str, Any]] = []
    if runtime.system_prompt:
        assembled.append({"role": "system", "content": runtime.system_prompt})

    context_message = _build_context_message(context)
    if context_message:
        assembled.append(context_message)

    if include_toolbox:
        toolbox_lines: List[str] = [runtime.format_toolbox()] if runtime.tool_summaries else []
        if extra_tool_descriptions:
            toolbox_lines.extend(str(item) for item in extra_tool_descriptions if item)
        if toolbox_lines:
            assembled.append(_build_tool_message("\n".join(toolbox_lines)))

    assembled.extend(_ensure_messages(messages))
    return assembled


def run_base_agent(
    *,
    messages: Sequence[Mapping[str, Any]],
    agent_name: str = "freshbot-base-agent",
    context: Optional[Mapping[str, Any]] = None,
    include_toolbox: bool = True,
    extra_tool_descriptions: Optional[Iterable[str]] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    response_format: Optional[Mapping[str, Any]] = None,
    gateway_alias: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute the Base agent type against the configured gateway."""

    runtime = load_runtime(agent_name)
    assembled_messages = assemble_messages(
        messages=messages,
        runtime=runtime,
        context=context,
        include_toolbox=include_toolbox,
        extra_tool_descriptions=extra_tool_descriptions,
    )
    result = invoke_llm(
        runtime,
        messages=assembled_messages,
        temperature=temperature,
        max_tokens=max_tokens,
        response_format=response_format,
        gateway_alias=gateway_alias,
    )
    return {
        "agent": runtime.agent.name,
        "content": result.get("content"),
        "raw": result,
    }


__all__ = [
    "AgentExecutionError",
    "AgentRuntime",
    "ToolSummary",
    "assemble_messages",
    "invoke_llm",
    "load_runtime",
    "run_base_agent",
]
