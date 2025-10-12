"""Register Prefect flows and deployments defined in Freshbot."""

from __future__ import annotations

import argparse
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import yaml
from prefect.deployments import deploy
from prefect.deployments.runner import EntrypointType, RunnerDeployment
from prefect.utilities.importtools import import_object

try:  # Prefect 2 schedule models moved modules a few times
    from prefect.client.schemas.schedules import CronSchedule
except ModuleNotFoundError:  # pragma: no cover - fallback for older versions
    from prefect.client.schemas.schedules import CronSchedule  # type: ignore

LOGGER = logging.getLogger(__name__)


@dataclass
class FlowSpec:
    callable: str
    name: Optional[str] = None
    work_queue: Optional[str] = None
    work_pool: Optional[str] = None
    tags: Optional[List[str]] = None
    parameters: Optional[Dict[str, Any]] = None
    description: Optional[str] = None
    schedule: Optional[Dict[str, Any]] = None
    infra_overrides: Optional[Dict[str, Any]] = None


def load_flow_specs(path: Path) -> List[FlowSpec]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not payload:
        return []
    if isinstance(payload, dict):
        entries = payload.get("flows", [])
    elif isinstance(payload, list):
        entries = payload
    else:
        raise TypeError(f"Unsupported flows payload type: {type(payload)!r}")

    specs: List[FlowSpec] = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise TypeError("Flow specs must be dictionaries")
        if "callable" not in entry:
            raise ValueError("Flow spec missing required 'callable' field")
        specs.append(
            FlowSpec(
                callable=entry["callable"],
                name=entry.get("name"),
                work_queue=entry.get("work_queue"),
                work_pool=entry.get("work_pool"),
                tags=entry.get("tags"),
                parameters=entry.get("parameters"),
                description=entry.get("description"),
                schedule=entry.get("schedule"),
                infra_overrides=entry.get("infra_overrides"),
            )
        )
    return specs


def build_schedule(spec: Optional[Dict[str, Any]]) -> Optional[CronSchedule]:
    if not spec:
        return None
    if "cron" not in spec:
        raise ValueError("Schedule spec requires a 'cron' expression")
    return CronSchedule(cron=spec["cron"], timezone=spec.get("timezone"))


def _resolve_work_pool(spec: FlowSpec, *, apply: bool, flow_name: str) -> Tuple[str, bool]:
    """Return the work-pool name for a flow, flagging whether it is a fallback."""

    env_pool = os.getenv("FRESHBOT_PREFECT_WORK_POOL")
    target_pool = spec.work_pool or env_pool
    if target_pool:
        return target_pool, False

    message = (
        "No Prefect work-pool configured for flow '%s'. "
        "Set 'work_pool' in flows.yaml or export FRESHBOT_PREFECT_WORK_POOL."
    ) % flow_name

    if apply:
        raise RuntimeError(message)

    LOGGER.warning("%s Defaulting to 'freshbot-process' for preview only.", message)
    return "freshbot-process", True


def register_flows(specs: Iterable[FlowSpec], apply: bool, filter_names: Optional[Sequence[str]] = None) -> None:
    for spec in specs:
        deployment_name = spec.name
        if not deployment_name:
            # If not provided, use the underlying flow name later
            deployment_name = None

        if filter_names and spec.callable not in filter_names and (deployment_name not in filter_names if deployment_name else True):
            continue

        flow = import_object(spec.callable)
        flow_name = getattr(flow, "name", None) or getattr(flow, "__name__", "anonymous-flow")
        deployment_name = deployment_name or flow_name

        if filter_names:
            identifiers = {spec.callable, deployment_name}
            if not any(name in identifiers for name in filter_names):
                continue

        LOGGER.info("Building deployment for flow %s (%s)", flow_name, spec.callable)
        target_pool, used_fallback = _resolve_work_pool(spec, apply=apply, flow_name=flow_name)
        deployment = RunnerDeployment.from_flow(
            flow=flow,
            name=deployment_name,
            schedule=build_schedule(spec.schedule),
            parameters=spec.parameters,
            description=spec.description,
            tags=spec.tags,
            work_queue_name=spec.work_queue,
            work_pool_name=target_pool,
            job_variables=spec.infra_overrides,
            entrypoint_type=EntrypointType.MODULE_PATH,
        )

        if apply:
            if used_fallback:
                LOGGER.warning(
                    "Applying deployments with fallback pool 'freshbot-process'. Start a worker "
                    "for that pool before triggering runs."
                )
            deployment_ids = deploy(
                deployment,
                work_pool_name=target_pool,
                build=False,
                push=False,
                ignore_warnings=True,
            )
            LOGGER.info(
                "Registered Prefect deployment %s/%s (%s)",
                flow_name,
                deployment_name,
                deployment_ids,
            )
        else:
            LOGGER.info(
                "DRY-RUN would register deployment %s/%s",
                flow_name,
                deployment_name,
            )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Register Prefect flows for Freshbot.")
    parser.add_argument(
        "--flows-spec",
        dest="flows_spec",
        default=Path("src/freshbot/flows/flows.yaml"),
        type=Path,
        help="Path to YAML file describing Prefect flows/deployments.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply registrations instead of performing a dry run.",
    )
    parser.add_argument(
        "--flow",
        action="append",
        dest="flow_filters",
        help="Limit to specific flow callable string or deployment name.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set log verbosity.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s %(message)s",
        force=True,
    )

    specs = load_flow_specs(Path(args.flows_spec))
    if not specs:
        LOGGER.warning("No flows found in %s", args.flows_spec)
        return

    register_flows(specs, apply=args.apply, filter_names=args.flow_filters)


if __name__ == "__main__":  # pragma: no cover
    main()
