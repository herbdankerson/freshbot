"""Utility helpers for loading gateway definitions from ParadeDB."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Sequence

from ..connectors.catalog import lookup as lookup_connectors


@dataclass(frozen=True)
class GatewaySchema:
    """Lightweight schema metadata for validating gateway payloads."""

    request_required: Sequence[str] = field(default_factory=tuple)
    response_required: Sequence[str] = field(default_factory=tuple)

    @staticmethod
    def from_mapping(mapping: Mapping[str, Any] | None) -> "GatewaySchema":
        if not mapping:
            return GatewaySchema()
        request = mapping.get("request") if isinstance(mapping, Mapping) else None
        response = mapping.get("response") if isinstance(mapping, Mapping) else None
        request_required: Sequence[str] = tuple(
            request.get("required", []) if isinstance(request, Mapping) else []
        )
        response_required: Sequence[str] = tuple(
            response.get("required", []) if isinstance(response, Mapping) else []
        )
        return GatewaySchema(
            request_required=request_required,
            response_required=response_required,
        )


@dataclass(frozen=True)
class GatewayConfig:
    """Resolved gateway configuration fetched from ParadeDB."""

    alias: str
    endpoint: str
    default_params: Mapping[str, Any]
    config: Mapping[str, Any]
    schema: GatewaySchema

    def validate_request(self, payload: Mapping[str, Any]) -> None:
        missing = [key for key in self.schema.request_required if key not in payload]
        if missing:
            raise ValueError(
                f"Gateway '{self.alias}' payload missing required keys: {', '.join(missing)}"
            )

    def validate_response(self, payload: Mapping[str, Any]) -> None:
        missing = [key for key in self.schema.response_required if key not in payload]
        if missing:
            raise ValueError(
                f"Gateway '{self.alias}' response missing required keys: {', '.join(missing)}"
            )


def load_gateway(alias: str) -> GatewayConfig:
    """Load a gateway configuration by alias."""

    records = lookup_connectors(alias)
    if not records:
        raise LookupError(f"No gateway connector registered with alias '{alias}'")
    record = records[0]
    default_params = dict(record.get("default_params") or {})
    config = dict(record.get("config") or default_params)
    schema = GatewaySchema.from_mapping(config.get("schema"))
    return GatewayConfig(
        alias=record["alias"],
        endpoint=record.get("endpoint", ""),
        default_params=default_params,
        config=config,
        schema=schema,
    )


__all__ = ["GatewayConfig", "GatewaySchema", "load_gateway"]
