"""Unified embedding gateway that routes requests based on registry providers."""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import httpx

from src.my_agentic_chatbot.config import get_settings
from src.my_agentic_chatbot.llm_calls.llm_client import get_current_run_logger
from src.my_agentic_chatbot.runtime_config import ModelConfig

from .ollama import embed_code_texts

LOGGER = logging.getLogger(__name__)


def _build_preview(inputs: Sequence[str]) -> List[Dict[str, object]]:
    preview: List[Dict[str, object]] = []
    for idx, text in enumerate(inputs[:3]):
        snippet = text[:200] + "…" if len(text) > 200 else text
        preview.append(
            {
                "index": idx,
                "length": len(text),
                "snippet": snippet,
                "truncated": len(text) > len(snippet),
            }
        )
    return preview


def _log_embedding_event(*, tool: str, payload: Dict[str, object], status: str) -> None:
    logger = get_current_run_logger()
    if logger is None:
        return
    logger.log_event("llm_call", payload, tool=tool, status=status)


def _extract_openai_vectors(payload: Mapping[str, Any]) -> List[List[float]]:
    data = payload.get("data") if isinstance(payload, Mapping) else None
    if not isinstance(data, list):
        raise RuntimeError("Embedding response missing data array")
    vectors: List[List[float]] = []
    for item in data:
        if not isinstance(item, Mapping) or "embedding" not in item:
            raise RuntimeError("Embedding item missing embedding field")
        embedding = item.get("embedding")
        if not isinstance(embedding, Iterable):
            raise RuntimeError("Embedding vector was not iterable")
        vector = [float(value) for value in embedding]
        vectors.append(vector)
    return vectors


def _resolve_base_url(config: ModelConfig) -> str:
    for candidate in (
        config.resolved_uri,
        config.uri_template,
        config.config.get("api_base"),
        config.config.get("endpoint"),
    ):
        if candidate:
            return str(candidate).rstrip("/")
    return ""


def _embed_openai_style(
    *,
    model_cfg: ModelConfig,
    inputs: Sequence[str],
    truncation_meta: Sequence[Dict[str, object]],
) -> List[List[float]]:
    settings = get_settings()
    base_url = _resolve_base_url(model_cfg)
    if not base_url:
        raise RuntimeError(f"Model '{model_cfg.name}' missing endpoint configuration")

    path = str(model_cfg.config.get("path") or "/v1/embeddings")
    if path.startswith("http://") or path.startswith("https://"):
        url = path
    else:
        url = f"{base_url}/{path.lstrip('/')}"

    headers: Dict[str, str] = {}
    config_headers = model_cfg.config.get("headers")
    if isinstance(config_headers, Mapping):
        headers.update({str(key): str(value) for key, value in config_headers.items()})
    api_key = model_cfg.config.get("api_key")
    if api_key:
        headers.setdefault("Authorization", f"Bearer {api_key}")

    payload: Dict[str, Any] = {
        "model": model_cfg.config.get("model") or model_cfg.identifier,
        "input": list(inputs),
    }
    request_overrides = model_cfg.config.get("request_overrides")
    if isinstance(request_overrides, Mapping):
        for key, value in request_overrides.items():
            payload.setdefault(key, value)
    task = model_cfg.config.get("task")
    if task and "task" not in payload:
        payload["task"] = task

    timeout_value = model_cfg.config.get("timeout_seconds") or settings.litellm_timeout_seconds
    try:
        timeout = float(timeout_value)
    except (TypeError, ValueError):
        timeout = settings.litellm_timeout_seconds

    log_payload: Dict[str, object] = {
        "endpoint": url,
        "model": payload.get("model"),
        "input_count": len(inputs),
        "input_preview": _build_preview(inputs),
    }
    if truncation_meta:
        log_payload["truncated_inputs"] = list(truncation_meta)

    with httpx.Client(timeout=timeout) as client:
        try:
            response = client.post(url, json=payload, headers=headers or None)
            response.raise_for_status()
            data = response.json()
            vectors = _extract_openai_vectors(data)
            if len(vectors) != len(inputs):
                raise RuntimeError("Embedding count mismatch")
            log_payload.update(
                {
                    "status_code": response.status_code,
                    "elapsed_ms": round(response.elapsed.total_seconds() * 1000, 2),
                    "vector_dims": len(vectors[0]) if vectors else 0,
                }
            )
            _log_embedding_event(tool=model_cfg.provider, payload=log_payload, status="success")
            return vectors
        except Exception as exc:
            log_payload.update({"error": str(exc)})
            _log_embedding_event(tool=model_cfg.provider, payload=log_payload, status="error")
            raise


def embed_texts(
    model_cfg: ModelConfig,
    inputs: Sequence[str],
    *,
    truncation_meta: Sequence[Dict[str, object]] | None = None,
) -> List[List[float]]:
    """Route embedding requests according to the provider defined in the registry."""

    if not inputs:
        return []

    provider = (model_cfg.provider or "").lower()
    meta = truncation_meta or ()

    if provider == "ollama":
        alias = str(model_cfg.config.get("gateway_alias") or model_cfg.name)
        timeout = model_cfg.config.get("timeout_seconds")
        return embed_code_texts(
            inputs,
            gateway_alias=alias,
            timeout_seconds=float(timeout) if timeout else None,
        )

    if provider in {"nvembed", "legalembed", "litellm", "openai"}:
        return _embed_openai_style(model_cfg=model_cfg, inputs=inputs, truncation_meta=meta)

    raise RuntimeError(f"Unsupported embedding provider '{model_cfg.provider}' for model '{model_cfg.name}'")


__all__ = ["embed_texts"]
