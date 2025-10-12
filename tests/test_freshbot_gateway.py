"""Unit tests for Freshbot gateway helpers."""

from __future__ import annotations

from typing import Any, Dict

import pytest

from freshbot.gateways.openai import OpenAIGatewayClient
from freshbot.gateways.registry import GatewayConfig, GatewaySchema


class DummyResponse:
    def __init__(self, payload: Dict[str, Any]):
        self._payload = payload

    def raise_for_status(self) -> None:  # pragma: no cover - behaviourless stub
        return None

    def json(self) -> Dict[str, Any]:
        return self._payload


class DummyClient:
    def __init__(self, *, response_payload: Dict[str, Any]):
        self._response_payload = response_payload
        self.sent_json: Dict[str, Any] | None = None
        self.sent_headers: Dict[str, Any] | None = None

    # Context manager helpers so it can stand in for httpx.Client
    def __enter__(self) -> "DummyClient":  # pragma: no cover - trivial
        return self

    def __exit__(self, *exc: object) -> None:  # pragma: no cover - trivial
        return None

    def post(self, path: str, json: Dict[str, Any], headers: Dict[str, Any] | None = None) -> DummyResponse:
        self.sent_json = json
        self.sent_headers = headers
        return DummyResponse(self._response_payload)


@pytest.mark.parametrize(
    "reasoning_payload",
    [
        [{"text": "Step 1"}, {"text": "Step 2"}],
        ["Step A", "Step B"],
    ],
)
def test_openai_gateway_applies_reasoning_fallback(monkeypatch: pytest.MonkeyPatch, reasoning_payload):
    """Ensure reasoning-only outputs fill ``message.content`` for downstream callers."""

    response_payload = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "reasoning_content": reasoning_payload,
                }
            }
        ]
    }

    dummy_client = DummyClient(response_payload=response_payload)

    def fake_client(*args: Any, **kwargs: Any) -> DummyClient:
        return dummy_client

    monkeypatch.setattr("freshbot.gateways.openai.httpx.Client", fake_client)

    gateway = GatewayConfig(
        alias="connector_test",
        endpoint="http://example",
        default_params={},
        config={},
        schema=GatewaySchema(),
    )
    client = OpenAIGatewayClient(gateway)
    request_payload = {"model": "demo", "messages": [{"role": "user", "content": "ping"}]}

    result = client.chat(request_payload)

    assert dummy_client.sent_json == request_payload
    assert dummy_client.sent_headers is None
    choices = result["choices"]
    assert isinstance(choices, list)
    message = choices[0]["message"]
    expected = "".join(
        item.get("text") if isinstance(item, dict) and "text" in item else str(item)
        for item in reasoning_payload
    )
    assert message["content"] == expected


def test_openai_gateway_applies_static_headers(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify connector-defined headers are forwarded with the request."""

    response_payload = {
        "choices": [
            {"message": {"role": "assistant", "content": "ok"}}
        ]
    }

    dummy_client = DummyClient(response_payload=response_payload)

    def fake_client(*args: Any, **kwargs: Any) -> DummyClient:
        return dummy_client

    monkeypatch.setattr("freshbot.gateways.openai.httpx.Client", fake_client)

    gateway = GatewayConfig(
        alias="connector_test",
        endpoint="http://example",
        default_params={},
        config={"headers": {"Authorization": "Bearer demo"}},
        schema=GatewaySchema(),
    )
    client = OpenAIGatewayClient(gateway)
    request_payload = {"model": "demo", "messages": [{"role": "user", "content": "hi"}]}

    client.chat(request_payload)

    assert dummy_client.sent_headers == {"Authorization": "Bearer demo"}
