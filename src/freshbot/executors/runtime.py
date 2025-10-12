"""Runtime helpers for invoking Freshbot tools and agents from registry metadata."""

from __future__ import annotations

import importlib
import inspect
from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, MutableMapping, Optional

from freshbot.registry import AgentRecord, RegistrySnapshot, ToolRecord, get_registry

_METADATA_KEYS = {"description", "input_schema", "output_schema", "sources", "notes"}
_EXECUTE_FLOW_PARAMS = {"deployment_name", "parameters", "wait_for_completion"}


@dataclass(frozen=True)
class InvocationResult:
    """Container for execution results plus metadata."""

    slug: str
    result: Any
    metadata: Dict[str, Any]


def _load_callable(path: str) -> Callable[..., Any]:
    if ":" not in path:
        raise ValueError(f"Callable reference '{path}' must be in 'module:object' format")
    module_name, attr = path.split(":", 1)
    module = importlib.import_module(module_name)
    try:
        return getattr(module, attr)
    except AttributeError as exc:  # pragma: no cover - defensive
        raise AttributeError(f"Module '{module_name}' does not define '{attr}'") from exc


def _collect_metadata(record_defaults: Mapping[str, Any]) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {}
    for key in _METADATA_KEYS:
        if key in record_defaults:
            metadata[key] = record_defaults[key]
    return metadata


def _merge_parameters(
    callable_obj: Callable[..., Any],
    params: Dict[str, Any],
) -> Dict[str, Any]:
    signature = inspect.signature(callable_obj)
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
        return params
    filtered: Dict[str, Any] = {}
    for name in signature.parameters:
        if name in params:
            filtered[name] = params[name]
    return filtered


def _load_snapshot() -> RegistrySnapshot:
    return get_registry()


def invoke_tool(
    slug: str,
    *,
    payload: Optional[Mapping[str, Any]] = None,
    agent: Optional[str] = None,
    extra_overrides: Optional[Mapping[str, Any]] = None,
) -> InvocationResult:
    """Invoke the tool identified by ``slug`` with optional payload."""

    snapshot = _load_snapshot()
    if slug not in snapshot.tools:
        raise RuntimeError(f"Tool '{slug}' is not registered")
    tool: ToolRecord = snapshot.tools[slug]
    callable_ref = tool.manifest_or_ref
    func = _load_callable(callable_ref)

    params: Dict[str, Any] = {}
    params.update(tool.default_params or {})
    if agent:
        agent_record = snapshot.require_agent(agent)
        params.update(agent_record.tool_overrides(slug))
    if extra_overrides:
        params.update(extra_overrides)

    metadata = _collect_metadata(params)

    if callable_ref == "freshbot.executors.prefect:execute_flow":
        runtime_params = {key: params.get(key) for key in _EXECUTE_FLOW_PARAMS if key in params}
        runtime_params.setdefault("parameters", dict(payload or {}))
    else:
        runtime_params = dict(params)
        payload_dict = dict(payload or {})
        runtime_params.update(payload_dict)

    runtime_params = _merge_parameters(func, runtime_params)
    result = func(**runtime_params)
    return InvocationResult(slug=slug, result=result, metadata=metadata)


def invoke_agent(
    name: str,
    *,
    payload: Mapping[str, Any],
    overrides: Optional[Mapping[str, Any]] = None,
) -> InvocationResult:
    """Execute an agent flow referenced by ``name`` with the provided payload."""

    snapshot = _load_snapshot()
    agent: AgentRecord = snapshot.require_agent(name)
    if not isinstance(agent.params, MutableMapping):
        raise RuntimeError(f"Agent '{name}' params do not define an entrypoint")
    entrypoint = agent.params.get("entrypoint")
    if not entrypoint:
        raise RuntimeError(f"Agent '{name}' is missing an 'entrypoint' parameter")

    func = _load_callable(str(entrypoint))
    params: Dict[str, Any] = dict(agent.params.get("defaults") or {})
    params.update(payload)
    if overrides:
        params.update(dict(overrides))
    params.setdefault("agent_name", agent.name)
    runtime_params = _merge_parameters(func, params)
    result = func(**runtime_params)
    metadata = _collect_metadata(agent.params if isinstance(agent.params, Mapping) else {})
    return InvocationResult(slug=name, result=result, metadata=metadata)


__all__ = ["InvocationResult", "invoke_agent", "invoke_tool"]
