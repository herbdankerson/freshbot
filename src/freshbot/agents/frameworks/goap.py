"""Goal-Oriented Action Planner integration for Freshbot.

This module exposes a reusable client for the external GOAP planning service and a
`goap_planner` helper that can be registered as an agent entrypoint. The planner
accepts an objective, the current world state, and a catalogue of available actions,
delegates the search to the GOAP service, and normalises the resulting action plan
into a structured payload that downstream orchestration flows can consume.

Every public function includes enough metadata to audit planner behaviour end-to-end:
  * Requests capture the caller-supplied goal, world state, toolbox, and action set.
  * Responses record the structured steps, dependency information, and raw service
    payload so we can persist artefacts and reconstruct DAGs.
The module is intentionally standalone so that other subsystems (task sync, MCP tools,
tests) can reuse the client without importing Prefect- or repository-specific code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence
from urllib.parse import urljoin
import logging

import httpx

from freshbot.registry import get_registry
from freshbot.registry.snapshot import AgentRecord, ModelRecord, RegistrySnapshot

LOGGER = logging.getLogger(__name__)


class GOAPPlannerError(RuntimeError):
    """Raised when GOAP planner execution fails."""


def _as_dict(mapping: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    if isinstance(mapping, MutableMapping):
        return dict(mapping)
    return {}


def _as_list(sequence: Optional[Iterable[Any]]) -> List[Any]:
    if sequence is None:
        return []
    if isinstance(sequence, (list, tuple)):
        return list(sequence)
    return [sequence]


def _coerce_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
        raise GOAPPlannerError(f"Invalid cost value: {value!r}") from exc


def _coerce_str(value: Any, *, fallback: str) -> str:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return fallback


@dataclass(frozen=True)
class GOAPActionDefinition:
    """Describes an action available to the planner."""

    name: str
    description: Optional[str] = None
    parameters: Mapping[str, Any] = field(default_factory=dict)
    preconditions: Sequence[Mapping[str, Any]] = field(default_factory=list)
    effects: Sequence[Mapping[str, Any]] = field(default_factory=list)
    cost: Optional[float] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_payload(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "name": self.name,
            "parameters": dict(self.parameters or {}),
            "preconditions": [dict(item or {}) for item in self.preconditions],
            "effects": [dict(item or {}) for item in self.effects],
        }
        if self.description:
            payload["description"] = self.description
        if self.cost is not None:
            payload["cost"] = self.cost
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload


@dataclass(frozen=True)
class GOAPPlanStep:
    """Represents a single step in the GOAP-generated plan."""

    index: int
    action_id: str
    action: str
    description: Optional[str]
    parameters: Mapping[str, Any]
    preconditions: Sequence[Mapping[str, Any]]
    effects: Sequence[Mapping[str, Any]]
    cost: Optional[float]
    depends_on: Sequence[str]
    metadata: Mapping[str, Any]
    raw: Mapping[str, Any]

    def to_payload(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "action_id": self.action_id,
            "action": self.action,
            "description": self.description,
            "parameters": dict(self.parameters or {}),
            "preconditions": [dict(item or {}) for item in self.preconditions],
            "effects": [dict(item or {}) for item in self.effects],
            "cost": self.cost,
            "depends_on": list(self.depends_on or []),
            "metadata": dict(self.metadata or {}),
        }

    @classmethod
    def from_payload(cls, index: int, payload: Mapping[str, Any]) -> "GOAPPlanStep":
        """Construct a plan step from a raw planner payload mapping."""

        action_name = _coerce_str(
            payload.get("action") or payload.get("name"),
            fallback=f"action_{index}",
        )
        action_id = _coerce_str(
            payload.get("action_id") or payload.get("id") or payload.get("identifier"),
            fallback=action_name,
        )
        depends_on = _as_list(payload.get("depends_on") or payload.get("requires"))
        return cls(
            index=index,
            action_id=action_id,
            action=action_name,
            description=payload.get("description"),
            parameters=_as_dict(payload.get("parameters")),
            preconditions=[_as_dict(item) for item in _as_list(payload.get("preconditions"))],
            effects=[_as_dict(item) for item in _as_list(payload.get("effects"))],
            cost=_coerce_float(payload.get("cost")),
            depends_on=[str(dep) for dep in depends_on if str(dep).strip()],
            metadata=_as_dict(payload.get("metadata")),
            raw=dict(payload),
        )


@dataclass(frozen=True)
class GOAPPlanResult:
    """Normalised representation of the planner response."""

    steps: Sequence[GOAPPlanStep]
    metadata: Mapping[str, Any]
    raw_response: Mapping[str, Any]

    def to_payload(self) -> Dict[str, Any]:
        return {
            "steps": [step.to_payload() for step in self.steps],
            "metadata": dict(self.metadata or {}),
            "raw_response": dict(self.raw_response or {}),
        }


class GOAPClient:
    """HTTP client for the external GOAP planner service."""

    def __init__(
        self,
        *,
        base_url: str,
        plan_path: str = "/plan",
        timeout_seconds: float = 30.0,
        headers: Optional[Mapping[str, str]] = None,
    ) -> None:
        if not base_url:
            raise GOAPPlannerError("GOAP planner base_url must be configured")
        self._base_url = base_url.rstrip("/")
        self._plan_path = plan_path or "/plan"
        self._timeout = timeout_seconds
        self._headers = dict(headers or {})

    @classmethod
    def from_model(
        cls,
        model: ModelRecord,
        *,
        params: Optional[Mapping[str, Any]] = None,
    ) -> "GOAPClient":
        """Construct a client using registry model metadata."""

        config_sources: List[Mapping[str, Any]] = []
        config = _as_dict(model.default_params)
        if config:
            config_sources.append(config)

        param_sources: List[Mapping[str, Any]] = []
        if isinstance(params, Mapping):
            param_sources.append(dict(params))
            defaults_section = params.get("defaults")
            if isinstance(defaults_section, Mapping):
                param_sources.append(dict(defaults_section))
            service_section = params.get("service")
            if isinstance(service_section, Mapping):
                param_sources.append(dict(service_section))

        def _lookup(keys: Sequence[str], fallback: Any = None) -> Any:
            for source in param_sources + config_sources:
                for key in keys:
                    if key in source and source[key] is not None:
                        return source[key]
            return fallback

        plan_path = _lookup(["plan_path", "path"], fallback="/plan")
        timeout = _lookup(["timeout_seconds", "timeout"], fallback=30)

        headers: Dict[str, str] = {}
        for source in param_sources + config_sources:
            value = source.get("headers")
            if isinstance(value, Mapping):
                headers.update({str(key): str(val) for key, val in value.items()})

        return cls(
            base_url=model.endpoint or "",
            plan_path=str(plan_path),
            timeout_seconds=float(timeout),
            headers=headers or None,
        )

    def plan(
        self,
        *,
        goal: Any,
        world_state: Optional[Mapping[str, Any]] = None,
        actions: Optional[Sequence[GOAPActionDefinition | Mapping[str, Any]]] = None,
        toolbox: Optional[Sequence[Mapping[str, Any]]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> GOAPPlanResult:
        """Request a plan from the GOAP service and normalise the response."""

        url = urljoin(f"{self._base_url}/", self._plan_path.lstrip("/"))
        payload_actions: List[Dict[str, Any]] = []
        for entry in actions or []:
            if isinstance(entry, GOAPActionDefinition):
                payload_actions.append(entry.to_payload())
            elif isinstance(entry, Mapping):
                payload_actions.append(dict(entry))
            else:  # pragma: no cover - defensive
                raise GOAPPlannerError(f"Unsupported action entry: {type(entry)!r}")

        request_body = {
            "goal": goal,
            "world_state": world_state or {},
            "actions": payload_actions,
            "toolbox": list(toolbox or []),
            "metadata": dict(metadata or {}),
        }

        LOGGER.info(
            "Dispatching GOAP plan request",
            extra={
                "url": url,
                "action_count": len(payload_actions),
                "has_world_state": bool(world_state),
            },
        )
        try:
            response = httpx.post(
                url,
                json=request_body,
                headers=self._headers or None,
                timeout=self._timeout,
            )
        except httpx.HTTPError as exc:  # pragma: no cover - network failure handling
            raise GOAPPlannerError(f"GOAP planner request failed: {exc}") from exc

        if response.status_code >= 400:
            raise GOAPPlannerError(
                f"GOAP planner returned {response.status_code}: {response.text}"
            )

        try:
            data = response.json()
        except ValueError as exc:  # pragma: no cover - defensive
            raise GOAPPlannerError("GOAP planner returned invalid JSON") from exc

        plan_steps = data.get("plan") or data.get("steps")
        if not isinstance(plan_steps, list) or not plan_steps:
            raise GOAPPlannerError("GOAP planner response did not include any steps")

        steps: List[GOAPPlanStep] = []
        for index, raw_step in enumerate(plan_steps, start=1):
            if not isinstance(raw_step, Mapping):
                raise GOAPPlannerError(f"GOAP plan step {index} is not a mapping")
            action_name = _coerce_str(
                raw_step.get("action") or raw_step.get("name"),
                fallback=f"action_{index}",
            )
            action_id = _coerce_str(
                raw_step.get("action_id")
                or raw_step.get("id")
                or raw_step.get("identifier"),
                fallback=action_name,
            )
            depends_on = _as_list(raw_step.get("depends_on") or raw_step.get("requires"))
            steps.append(
                GOAPPlanStep(
                    index=index,
                    action_id=action_id,
                    action=action_name,
                    description=raw_step.get("description"),
                    parameters=_as_dict(raw_step.get("parameters")),
                    preconditions=[
                        _as_dict(item) for item in _as_list(raw_step.get("preconditions"))
                    ],
                    effects=[_as_dict(item) for item in _as_list(raw_step.get("effects"))],
                    cost=_coerce_float(raw_step.get("cost")),
                    depends_on=[str(dep) for dep in depends_on if str(dep).strip()],
                    metadata=_as_dict(raw_step.get("metadata")),
                    raw=dict(raw_step),
                )
            )

        response_metadata = _as_dict(data.get("metadata"))
        LOGGER.info(
            "GOAP planner produced %s step(s)",
            len(steps),
            extra={"metadata_keys": sorted(response_metadata.keys())},
        )
        return GOAPPlanResult(
            steps=steps,
            metadata=response_metadata,
            raw_response=dict(data),
        )


def _resolve_agent_and_model(
    snapshot: RegistrySnapshot,
    *,
    agent_name: str,
) -> tuple[AgentRecord, ModelRecord]:
    agent = snapshot.require_agent(agent_name)
    model_alias: Optional[str] = agent.model_alias
    if not model_alias:
        params = agent.params if isinstance(agent.params, Mapping) else {}
        alias_candidate = params.get("service_alias")
        if isinstance(alias_candidate, str) and alias_candidate.strip():
            model_alias = alias_candidate.strip()
    if not model_alias:
        raise GOAPPlannerError(
            f"Agent '{agent_name}' is not associated with a GOAP connector alias"
        )
    model = snapshot.require_model(model_alias)
    if not model.endpoint:
        raise GOAPPlannerError(
            f"GOAP connector '{model.alias}' is missing an endpoint URL"
        )
    return agent, model


def goap_planner(
    *,
    goal: Any,
    world_state: Optional[Mapping[str, Any]] = None,
    actions: Optional[Sequence[GOAPActionDefinition | Mapping[str, Any]]] = None,
    toolbox: Optional[Sequence[Mapping[str, Any]]] = None,
    metadata: Optional[Mapping[str, Any]] = None,
    agent_name: str = "freshbot-goap-planner",
) -> Dict[str, Any]:
    """Execute the GOAP planner registered in the Freshbot agent registry.

    Parameters
    ----------
    goal:
        Desired end state as accepted by the GOAP service (string or mapping).
    world_state:
        Current facts about the environment. May be omitted when the planner
        can infer defaults.
    actions:
        Iterable of action definitions (either dataclass instances or plain
        mappings) that the planner may use during search.
    toolbox:
        Optional list describing available tools/skills; forwarded verbatim to the
        planner so higher-level orchestration flows can reason about capabilities.
    metadata:
        Additional payload forwarded to the planner (e.g. planner hints).
    agent_name:
        Name of the agent record to resolve from cfg.agents. Defaults to
        ``freshbot-goap-planner`` as defined in the registry.

    Returns
    -------
    dict
        Structured payload containing agent + model metadata, the normalised plan
        steps, and the raw response from the planner service.
    """

    snapshot = get_registry()
    agent, model = _resolve_agent_and_model(snapshot, agent_name=agent_name)
    params = agent.params if isinstance(agent.params, Mapping) else {}
    client = GOAPClient.from_model(model, params=params)
    result = client.plan(
        goal=goal,
        world_state=world_state,
        actions=actions,
        toolbox=toolbox,
        metadata=metadata,
    )
    return {
        "agent": agent.name,
        "model_alias": model.alias,
        "goal": goal,
        "world_state": world_state or {},
        "toolbox": list(toolbox or []),
        "metadata": dict(metadata or {}),
        "plan": [step.to_payload() for step in result.steps],
        "planner_metadata": dict(result.metadata or {}),
        "raw_response": dict(result.raw_response or {}),
    }


__all__ = [
    "GOAPActionDefinition",
    "GOAPPlanResult",
    "GOAPPlanStep",
    "GOAPClient",
    "GOAPPlannerError",
    "goap_planner",
]
