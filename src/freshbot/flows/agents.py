"""Prefect flows that execute Freshbot agents."""

from __future__ import annotations

import json
from contextlib import redirect_stdout
from io import StringIO
from typing import Any, Dict, Mapping, Optional, Sequence

try:  # pragma: no cover
    from prefect import flow, get_run_logger
except ModuleNotFoundError:  # pragma: no cover - fallback for local execution
    import logging

    def flow(function=None, *_, **__):  # type: ignore
        if function is None:
            def decorator(fn):
                return fn

            return decorator
        return function

    def get_run_logger():
        return logging.getLogger(__name__)

from freshbot.registry import AgentRecord, get_registry

from ..gateways import chat_completion
from ..prompts import fetch_prompt
from .utils import merge_system_prompt


def _load_agent(name: str) -> AgentRecord:
    registry = get_registry()
    return registry.require_agent(name)


def _resolve_system_prompt(agent: AgentRecord, prompt_ref: Optional[str], inline_prompt: Optional[str]) -> Optional[str]:
    if inline_prompt:
        return inline_prompt
    if prompt_ref:
        return fetch_prompt(prompt_ref)
    params_prompt = agent.params.get("prompt_ref") if isinstance(agent.params, dict) else None
    if params_prompt:
        resolved = fetch_prompt(str(params_prompt))
        if resolved:
            return resolved
    if agent.system_prompt:
        return agent.system_prompt
    return None


def _resolve_gateway(agent: AgentRecord, override: Optional[str]) -> str:
    if override:
        return override
    params_gateway = None
    if isinstance(agent.params, dict):
        params_gateway = agent.params.get("gateway_alias")
    if params_gateway:
        return str(params_gateway)
    if agent.model_alias:
        return str(agent.model_alias)
    return "connector_qwen_openai"


def _invoke_agent_chat(
    agent: AgentRecord,
    *,
    messages: Sequence[Mapping[str, Any]],
    system_prompt: Optional[str],
    gateway_alias: str,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    response_format: Optional[Mapping[str, Any]] = None,
    extra_params: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    payload_messages = merge_system_prompt(messages, system_prompt)
    result = chat_completion(
        payload_messages,
        gateway_alias=gateway_alias,
        temperature=temperature,
        max_tokens=max_tokens,
        response_format=response_format,
        extra_params=extra_params,
    )
    result.setdefault("agent", agent.name)
    return result


@flow(name="freshbot_agent_planner")
def planner_agent_flow(
    *,
    objective: str,
    context: Optional[Mapping[str, Any]] = None,
    available_tools: Optional[Sequence[Mapping[str, Any]]] = None,
    agent_name: str = "freshbot-planner",
    prompt_ref: Optional[str] = None,
    gateway_alias: Optional[str] = None,
    temperature: Optional[float] = None,
) -> Dict[str, Any]:
    """Generate a structured execution plan for downstream agents."""

    logger = get_run_logger()
    agent = _load_agent(agent_name)
    system_prompt = _resolve_system_prompt(agent, prompt_ref, None)
    messages = [
        {
            "role": "user",
            "content": json.dumps(
                {
                    "objective": objective,
                    "context": context or {},
                    "available_tools": available_tools or [],
                },
                ensure_ascii=False,
            ),
        }
    ]
    extras = agent.params.get("request_overrides") if isinstance(agent.params, dict) else None
    result = _invoke_agent_chat(
        agent,
        messages=messages,
        system_prompt=system_prompt,
        gateway_alias=_resolve_gateway(agent, gateway_alias),
        temperature=temperature,
        extra_params=extras if isinstance(extras, Mapping) else None,
    )
    logger.info("Planner produced plan for objective '%s'", objective)
    return {
        "agent": agent.name,
        "objective": objective,
        "plan": result.get("content"),
        "raw": result,
    }


@flow(name="freshbot_agent_tool_user")
def tool_user_agent_flow(
    *,
    task: str,
    inputs: Optional[Mapping[str, Any]] = None,
    agent_name: str = "freshbot-tool-user",
    prompt_ref: Optional[str] = None,
    gateway_alias: Optional[str] = None,
    temperature: Optional[float] = None,
) -> Dict[str, Any]:
    """Guide the tool user agent to execute a specific task."""

    logger = get_run_logger()
    agent = _load_agent(agent_name)
    system_prompt = _resolve_system_prompt(agent, prompt_ref, None)
    messages = [
        {
            "role": "user",
            "content": json.dumps(
                {
                    "task": task,
                    "inputs": inputs or {},
                },
                ensure_ascii=False,
            ),
        }
    ]
    result = _invoke_agent_chat(
        agent,
        messages=messages,
        system_prompt=system_prompt,
        gateway_alias=_resolve_gateway(agent, gateway_alias),
        temperature=temperature,
    )
    logger.info("Tool user agent completed task '%s'", task)
    return {
        "agent": agent.name,
        "task": task,
        "instructions": result.get("content"),
        "raw": result,
    }


@flow(name="freshbot_agent_responder")
def responder_agent_flow(
    *,
    data_packet: Mapping[str, Any],
    agent_name: str = "freshbot-responder",
    prompt_ref: Optional[str] = None,
    gateway_alias: Optional[str] = None,
    temperature: Optional[float] = None,
) -> Dict[str, Any]:
    """Produce a final user-facing response."""

    logger = get_run_logger()
    agent = _load_agent(agent_name)
    system_prompt = _resolve_system_prompt(agent, prompt_ref, None)
    messages = [
        {
            "role": "user",
            "content": json.dumps(data_packet, ensure_ascii=False),
        }
    ]
    result = _invoke_agent_chat(
        agent,
        messages=messages,
        system_prompt=system_prompt,
        gateway_alias=_resolve_gateway(agent, gateway_alias),
        temperature=temperature,
    )
    logger.info("Responder agent drafted reply for packet keys=%s", list(data_packet.keys()))
    return {
        "agent": agent.name,
        "response_text": result.get("content"),
        "raw": result,
    }


@flow(name="freshbot_agent_auditor")
def auditor_agent_flow(
    *,
    response_packet: Mapping[str, Any],
    requirements: Optional[Sequence[str]] = None,
    agent_name: str = "freshbot-auditor",
    prompt_ref: Optional[str] = None,
    gateway_alias: Optional[str] = None,
    temperature: Optional[float] = None,
) -> Dict[str, Any]:
    """Evaluate whether a response satisfies the provided requirements."""

    logger = get_run_logger()
    agent = _load_agent(agent_name)
    system_prompt = _resolve_system_prompt(agent, prompt_ref, None)
    payload = {
        "response": response_packet,
        "requirements": requirements or [],
    }
    messages = [
        {
            "role": "user",
            "content": json.dumps(payload, ensure_ascii=False),
        }
    ]
    result = _invoke_agent_chat(
        agent,
        messages=messages,
        system_prompt=system_prompt,
        gateway_alias=_resolve_gateway(agent, gateway_alias),
        temperature=temperature,
    )
    logger.info("Auditor agent evaluated response; content length=%s", len(response_packet))
    return {
        "agent": agent.name,
        "verdict": result.get("content"),
        "raw": result,
    }


@flow(name="freshbot_agent_code_executor")
def code_executor_agent_flow(
    *,
    code: str,
    globals_override: Optional[Mapping[str, Any]] = None,
    locals_override: Optional[Mapping[str, Any]] = None,
    capture_stdout: bool = True,
) -> Dict[str, Any]:
    """Execute Python code in a constrained sandbox for quick utilities."""

    logger = get_run_logger()
    safe_globals: Dict[str, Any] = {"__builtins__": {"print": print, "len": len, "range": range}}
    if globals_override:
        safe_globals.update(dict(globals_override))
    safe_locals: Dict[str, Any] = {}
    if locals_override:
        safe_locals.update(dict(locals_override))
    stdout_buffer = StringIO()
    try:
        if capture_stdout:
            with redirect_stdout(stdout_buffer):
                exec(code, safe_globals, safe_locals)
        else:
            exec(code, safe_globals, safe_locals)
        logger.info("Code executor ran %s characters of code", len(code))
        return {
            "agent": "freshbot-code-executor",
            "stdout": stdout_buffer.getvalue() if capture_stdout else None,
            "globals": {k: v for k, v in safe_globals.items() if not k.startswith("__")},
            "locals": safe_locals,
        }
    except Exception as exc:  # pragma: no cover - defensive logging of runtime errors
        logger.exception("Code execution failed")
        return {
            "agent": "freshbot-code-executor",
            "error": str(exc),
            "stdout": stdout_buffer.getvalue() if capture_stdout else None,
        }


@flow(name="freshbot_agent_custom")
def custom_agent_flow(
    *,
    agent_name: str,
    messages: Sequence[Mapping[str, Any]],
    system_prompt: Optional[str] = None,
    gateway_alias: Optional[str] = None,
    temperature: Optional[float] = None,
    response_format: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Execute a custom agent definition with optional prompt overrides."""

    logger = get_run_logger()
    agent = _load_agent(agent_name)
    prompt = _resolve_system_prompt(agent, None, system_prompt)
    result = _invoke_agent_chat(
        agent,
        messages=messages,
        system_prompt=prompt,
        gateway_alias=_resolve_gateway(agent, gateway_alias),
        temperature=temperature,
        response_format=response_format,
    )
    logger.info("Custom agent %s executed with %s messages", agent_name, len(messages))
    return {
        "agent": agent.name,
        "content": result.get("content"),
        "raw": result,
    }


__all__ = [
    "auditor_agent_flow",
    "code_executor_agent_flow",
    "custom_agent_flow",
    "planner_agent_flow",
    "responder_agent_flow",
    "tool_user_agent_flow",
]
