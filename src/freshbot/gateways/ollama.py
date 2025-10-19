"""Gateway client helpers for the Ollama embedding service."""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import httpx

from .registry import GatewayConfig, load_gateway


logger = logging.getLogger(__name__)

DEFAULT_CODE_EMBED_GATEWAY = "connector_ollama_code_embedding"
_EMBED_ENABLE_FLAG = "FRESHBOT_ENABLE_CODE_EMBEDDINGS"
_SKIP_DEFAULT_FIELDS = {
    "connector_type",
    "driver",
    "description",
    "schema",
    "prompt_template",
    "prompt_prefix",
    "prompt_suffix",
    "request_overrides",
    "dims",
}


def _resolve_payload(
    gateway: GatewayConfig,
    *,
    text: str,
) -> Dict[str, Any]:
    params: Dict[str, Any] = {}
    for key, value in gateway.default_params.items():
        if key in _SKIP_DEFAULT_FIELDS:
            continue
        params[key] = value

    config = gateway.config
    model_name = params.get("model") or config.get("model")
    if not model_name:
        raise ValueError(f"Gateway '{gateway.alias}' missing target model name")

    prompt_template = config.get("prompt_template", "{text}")
    prompt_prefix = config.get("prompt_prefix")
    prompt_suffix = config.get("prompt_suffix")

    prompt = prompt_template.format(text=text)
    if prompt_prefix:
        prompt = f"{prompt_prefix}{prompt}"
    if prompt_suffix:
        prompt = f"{prompt}{prompt_suffix}"

    payload = {"model": model_name, "prompt": prompt}
    for key, value in config.get("request_overrides", {}).items():
        payload.setdefault(key, value)
    return payload


def embed_code_texts(
    texts: Sequence[str],
    *,
    gateway_alias: str = DEFAULT_CODE_EMBED_GATEWAY,
    timeout_seconds: float | None = None,
) -> List[List[float]]:
    """Embed the supplied code-focused texts via the Ollama gateway."""

    if not texts:
        return []
    gateway = load_gateway(gateway_alias)
    endpoint = gateway.endpoint.rstrip("/")
    target_dims = gateway.config.get("dims") or gateway.default_params.get("dims")
    api_style = (
        str(gateway.config.get("api_style"))
        if gateway.config.get("api_style") is not None
        else str(gateway.default_params.get("api_style"))
        if gateway.default_params.get("api_style") is not None
        else "ollama"
    ).lower()
    if not _embeddings_enabled():
        dims = int(target_dims) if target_dims else 3584
        logger.warning(
            "Code embeddings disabled (set %s=1 to enable); returning zero vectors of length %s",
            _EMBED_ENABLE_FLAG,
            dims,
        )
        return [[0.0] * dims for _ in texts]

    if not endpoint:
        raise ValueError(f"Gateway '{gateway.alias}' missing endpoint URL")
    timeout = timeout_seconds or gateway.config.get("timeout_seconds") or 120.0

    vectors: List[List[float]] = []
    with httpx.Client(timeout=timeout) as client:
        for text in texts:
            if api_style == "openai":
                payload = {
                    "model": gateway.config.get("model")
                    or gateway.default_params.get("model"),
                    "input": text,
                }
                for key, value in gateway.config.get("request_overrides", {}).items():
                    payload.setdefault(key, value)
                for key, value in gateway.default_params.get(
                    "request_overrides", {}
                ).items():
                    payload.setdefault(key, value)
            else:
                payload = _resolve_payload(gateway, text=text)
            gateway.validate_request(payload)

            response = client.post(endpoint, json=payload)
            response.raise_for_status()
            data = response.json()
            if not isinstance(data, Mapping):
                raise ValueError(
                    f"Gateway '{gateway.alias}' returned non-object payload: {data!r}"
                )
            gateway.validate_response(data)
            if api_style == "openai":
                data_entries = data.get("data")
                if not isinstance(data_entries, Sequence) or not data_entries:
                    raise ValueError(
                        f"Gateway '{gateway.alias}' response missing embedding data"
                    )
                embedding = data_entries[0].get("embedding")
            else:
                embedding = data.get("embedding")
            if not isinstance(embedding, Iterable):
                raise ValueError(
                    f"Gateway '{gateway.alias}' response missing embedding vector"
                )
            vector = [float(value) for value in embedding]
            if target_dims:
                dims = int(target_dims)
                if len(vector) > dims:
                    vector = vector[:dims]
                elif len(vector) < dims:
                    vector = vector + [0.0] * (dims - len(vector))
            vectors.append(vector)
    return vectors


__all__ = [
    "DEFAULT_CODE_EMBED_GATEWAY",
    "embed_code_texts",
    "embeddings_enabled",
]


def _embeddings_enabled() -> bool:
    value = os.getenv(_EMBED_ENABLE_FLAG, "0").strip().lower()
    return value in {"1", "true", "t", "yes", "y"}


def embeddings_enabled() -> bool:
    """Public helper so callers can detect whether real embeddings are active."""

    return _embeddings_enabled()
