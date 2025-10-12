"""Unit tests for Freshbot executor runtime helpers."""

from __future__ import annotations

from typing import Any, Dict

import sys
import types


def _install_prefect_stub() -> None:
    if "prefect" in sys.modules:
        return

    prefect_module = types.ModuleType("prefect")

    deployments_module = types.ModuleType("prefect.deployments")

    def _run_deployment(*args: Any, **kwargs: Any) -> None:  # pragma: no cover - defensive stub
        raise RuntimeError("Prefect stub 'run_deployment' should not be executed in tests")

    deployments_module.run_deployment = _run_deployment

    states_module = types.ModuleType("prefect.states")

    class _State:  # pragma: no cover - sanity placeholder
        def __init__(self) -> None:
            self.type = types.SimpleNamespace(value="COMPLETED")
            self.name = "Completed"
            self.state_details = types.SimpleNamespace(flow_run_id=None)

        def result(self) -> Dict[str, Any]:
            return {}

    states_module.State = _State

    prefect_module.deployments = deployments_module
    prefect_module.states = states_module

    sys.modules["prefect"] = prefect_module
    sys.modules["prefect.deployments"] = deployments_module
    sys.modules["prefect.states"] = states_module


_install_prefect_stub()

import pytest

from freshbot.executors import runtime as runtime_module
from freshbot.registry import AgentRecord, RegistrySnapshot, ToolRecord


def echo_tool(text: str, suffix: str = "!") -> str:
    return f"{text}{suffix}"


def sample_agent(agent_name: str, objective: str, pre: str = "") -> Dict[str, Any]:
    return {"agent": agent_name, "objective": objective, "prefill": pre}


def _make_snapshot() -> RegistrySnapshot:
    tool_record = ToolRecord(
        id=1,
        slug="tool_echo",
        kind="python",
        manifest_or_ref="tests.test_freshbot_runtime:echo_tool",
        default_params={
            "suffix": "!",
            "description": "Echo tool",
            "input_schema": {"type": "object"},
        },
        enabled=True,
        notes=None,
    )
    agent_record = AgentRecord(
        id=1,
        name="agent_echo",
        type="planner",
        model_alias=None,
        system_prompt=None,
        params={
            "entrypoint": "tests.test_freshbot_runtime:sample_agent",
            "defaults": {"pre": "seed"},
            "description": "Echo agent",
        },
        tools_profile=None,
        enabled=True,
        notes=None,
        tool_bindings={
            "tool_echo": {"suffix": "?"}
        },
    )
    return RegistrySnapshot(
        providers={},
        models={},
        tools={tool_record.slug: tool_record},
        agents={agent_record.name: agent_record},
    )


def test_invoke_tool_merges_registry_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    snapshot = _make_snapshot()
    monkeypatch.setattr(runtime_module, "_load_snapshot", lambda: snapshot)

    result = runtime_module.invoke_tool("tool_echo", payload={"text": "hello"})

    assert result.result == "hello!"
    assert result.metadata == {
        "description": "Echo tool",
        "input_schema": {"type": "object"},
    }


def test_invoke_tool_applies_agent_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    snapshot = _make_snapshot()
    monkeypatch.setattr(runtime_module, "_load_snapshot", lambda: snapshot)

    result = runtime_module.invoke_tool("tool_echo", payload={"text": "hi"}, agent="agent_echo")

    assert result.result == "hi?"


def test_invoke_agent_passes_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    snapshot = _make_snapshot()
    monkeypatch.setattr(runtime_module, "_load_snapshot", lambda: snapshot)

    result = runtime_module.invoke_agent("agent_echo", payload={"objective": "demo"})

    assert result.result == {"agent": "agent_echo", "objective": "demo", "prefill": "seed"}
    assert result.metadata == {"description": "Echo agent"}
