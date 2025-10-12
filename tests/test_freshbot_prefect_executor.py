import json
from typing import Any, Dict

import pytest

from freshbot.executors import prefect as prefect_executor


class DummyResponse:
    def __init__(self, payload: Dict[str, Any], status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code

    def json(self) -> Dict[str, Any]:
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class DummyClient:
    def __init__(self, *, expected_timeout: float, capture: Dict[str, Any]) -> None:
        self._timeout = expected_timeout
        self._capture = capture

    def __enter__(self) -> "DummyClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # pragma: no cover - nothing to clean
        return None

    def post(self, url: str, json: Dict[str, Any]) -> DummyResponse:
        self._capture["url"] = url
        self._capture["json"] = json
        self._capture["timeout"] = self._timeout
        return DummyResponse(
            {
                "deployment_name": json["deployment_name"],
                "state_name": "SCHEDULED",
                "state_type": "SCHEDULED",
                "flow_run_id": "12345",
            }
        )


def test_execute_flow_uses_api_gateway(monkeypatch: pytest.MonkeyPatch) -> None:
    capture: Dict[str, Any] = {}

    def factory(*, timeout: float) -> DummyClient:
        return DummyClient(expected_timeout=timeout, capture=capture)

    monkeypatch.setenv("FRESHBOT_API_BASE_URL", "http://localhost:9000/api/")
    monkeypatch.setenv("FRESHBOT_API_TIMEOUT", "42")
    monkeypatch.setattr(prefect_executor, "_DEFAULT_TIMEOUT", 42.0)
    monkeypatch.setattr(prefect_executor.httpx, "Client", factory)

    result = prefect_executor.execute_flow(
        "freshbot_tool_qwen_chat/freshbot-tool-qwen-chat",
        parameters={"foo": "bar"},
        wait_for_completion=True,
    )

    assert capture["url"] == "http://localhost:9000/api/freshbot/flows/execute"
    assert capture["json"] == {
        "deployment_name": "freshbot_tool_qwen_chat/freshbot-tool-qwen-chat",
        "parameters": {"foo": "bar"},
        "wait_for_completion": True,
    }
    assert result["state_name"] == "SCHEDULED"
    assert capture["timeout"] == 42.0


def test_execute_flow_raises_on_bad_status(monkeypatch: pytest.MonkeyPatch) -> None:
    def factory(*, timeout: float) -> DummyClient:  # pragma: no cover - simple factory
        class ErrorClient(DummyClient):
            def post(self, url: str, json: Dict[str, Any]) -> DummyResponse:  # type: ignore[override]
                raise RuntimeError("boom")

        return ErrorClient(expected_timeout=timeout, capture={})

    monkeypatch.setattr(prefect_executor, "_DEFAULT_TIMEOUT", 60.0)

    monkeypatch.setattr(prefect_executor.httpx, "Client", factory)

    with pytest.raises(RuntimeError, match="Failed to trigger deployment"):
        prefect_executor.execute_flow("freshbot-heartbeat-deployment/freshbot-heartbeat-deployment")
