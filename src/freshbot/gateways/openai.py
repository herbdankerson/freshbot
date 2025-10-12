"""Gateway helpers for OpenAI-compatible chat endpoints."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping, MutableMapping, Optional, Sequence

import httpx

from .registry import GatewayConfig, load_gateway

_DEFAULT_PATH = "/v1/chat/completions"
_RESPONSE_CHOICE_KEYS = ("message", "delta")


def _coerce_messages(messages: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if not isinstance(messages, Iterable):
        raise TypeError("messages must be an iterable of role/content dictionaries")
    normalized: list[dict[str, Any]] = []
    for item in messages:
        if not isinstance(item, Mapping):
            raise TypeError("each message must be a mapping with at least role and content")
        role = item.get("role")
        content = item.get("content")
        if not role:
            raise ValueError("message missing role")
        if content is None:
            content = ""
        normalized.append({"role": str(role), "content": content})
    return normalized


def _apply_reasoning_fallback(choice: MutableMapping[str, Any]) -> None:
    """Ensure ``message.content`` is populated when the backend emits reasoning-only output."""

    message = choice.get("message")
    if not isinstance(message, MutableMapping):
        return
    content = message.get("content")
    if content:
        return
    reasoning_segments = message.get("reasoning_content")
    if not isinstance(reasoning_segments, Iterable):
        return
    flattened: list[str] = []
    for segment in reasoning_segments:
        if isinstance(segment, Mapping):
            text = segment.get("text")
            if text:
                flattened.append(str(text))
        elif segment:
            flattened.append(str(segment))
    if flattened:
        message["content"] = "".join(flattened)


class OpenAIGatewayClient:
    """Client bound to a ParadeDB connector entry for OpenAI-style chat calls."""

    def __init__(self, gateway: GatewayConfig) -> None:
        self._gateway = gateway
        raw_timeout = gateway.config.get("timeout_seconds") or gateway.default_params.get("timeout_seconds")
        self._timeout = float(raw_timeout) if raw_timeout else 60.0
        self._path = gateway.config.get("path") or _DEFAULT_PATH
        headers = gateway.config.get("headers") if isinstance(gateway.config, Mapping) else None
        self._headers = {str(key): str(value) for key, value in (headers or {}).items()}

    def build_payload(
        self,
        *,
        messages: Sequence[Mapping[str, Any]],
        model_override: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        response_format: Optional[Mapping[str, Any]] = None,
        extra_params: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        connector_params = dict(self._gateway.default_params)
        config = dict(self._gateway.config)
        model_name = model_override or config.get("routing_model") or connector_params.get("model")
        if not model_name:
            raise ValueError(f"Gateway '{self._gateway.alias}' missing routing model")

        payload: Dict[str, Any] = {
            "model": model_name,
            "messages": _coerce_messages(messages),
        }
        if temperature is not None:
            payload["temperature"] = temperature
        elif "temperature" in connector_params:
            payload.setdefault("temperature", connector_params["temperature"])
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if response_format is not None:
            payload["response_format"] = response_format

        extra_overrides = config.get("request_overrides") or {}
        if extra_params:
            extra_overrides = {**extra_overrides, **extra_params}
        for key, value in extra_overrides.items():
            payload.setdefault(key, value)
        return payload

    def chat(
        self,
        payload: Mapping[str, Any],
    ) -> Dict[str, Any]:
        endpoint = self._gateway.endpoint.rstrip("/")
        if not endpoint:
            raise ValueError(f"Gateway '{self._gateway.alias}' missing endpoint URL")
        request_body = dict(payload)
        self._gateway.validate_request(request_body)

        with httpx.Client(base_url=endpoint, timeout=self._timeout) as client:
            if self._headers:
                response = client.post(self._path, json=request_body, headers=self._headers)
            else:
                response = client.post(self._path, json=request_body)
            response.raise_for_status()
            data = response.json()

        if not isinstance(data, Mapping):
            raise ValueError(f"Gateway '{self._gateway.alias}' returned non-object response: {data!r}")
        self._gateway.validate_response(data)

        choices = data.get("choices")
        if isinstance(choices, Iterable):
            for choice in choices:
                if isinstance(choice, MutableMapping):
                    _apply_reasoning_fallback(choice)
        return dict(data)


def chat_completion(
    messages: Sequence[Mapping[str, Any]],
    *,
    gateway_alias: str,
    model_override: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    response_format: Optional[Mapping[str, Any]] = None,
    extra_params: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Invoke an OpenAI-compatible chat completion using the configured gateway."""

    gateway = load_gateway(gateway_alias)
    client = OpenAIGatewayClient(gateway)
    payload = client.build_payload(
        messages=messages,
        model_override=model_override,
        temperature=temperature,
        max_tokens=max_tokens,
        response_format=response_format,
        extra_params=extra_params,
    )
    response = client.chat(payload)
    return {
        "gateway": gateway.alias,
        "endpoint": gateway.endpoint,
        "payload": payload,
        "response": response,
        "content": _extract_primary_content(response),
    }


def _extract_primary_content(response: Mapping[str, Any]) -> Optional[str]:
    choices = response.get("choices")
    if not isinstance(choices, Iterable):
        return None
    for choice in choices:
        if not isinstance(choice, Mapping):
            continue
        message = choice.get("message")
        if isinstance(message, Mapping):
            content = message.get("content")
            if content:
                return str(content)
    return None


__all__ = ["OpenAIGatewayClient", "chat_completion"]
