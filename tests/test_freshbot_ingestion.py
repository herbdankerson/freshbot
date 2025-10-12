import json
from types import SimpleNamespace

import pytest

from src.freshbot.pipeline import ingestion


def test_classify_document_parses_json(monkeypatch):
    captured = {}

    def fake_invoke_tool(tool_slug, *, payload, agent=None, extra_overrides=None):
        captured["tool_slug"] = tool_slug
        captured["payload"] = payload
        body = {"domain": "code", "confidence": 0.87, "source_labels": ["document"]}
        return SimpleNamespace(result={"content": json.dumps(body)})

    monkeypatch.setattr(ingestion, "invoke_tool", fake_invoke_tool)

    result = ingestion.classify_document("some text")

    assert result == {
        "domain": "code",
        "confidence": pytest.approx(0.87),
        "source_labels": ["document"],
        "raw": {"domain": "code", "confidence": 0.87, "source_labels": ["document"]},
    }
    assert captured["tool_slug"] == ingestion.DEFAULT_CLASSIFIER_TOOL
    payload_messages = captured["payload"]["messages"]
    assert len(payload_messages) == 1
    decoded = json.loads(payload_messages[0]["content"])
    assert decoded["content"].startswith("some text")


def test_summarize_document_truncates_output(monkeypatch):
    summary_text = "Lorem ipsum dolor sit amet" * 40

    def fake_invoke_tool(tool_slug, *, payload, agent=None, extra_overrides=None):  # noqa: ARG001
        return SimpleNamespace(result={"content": summary_text})

    monkeypatch.setattr(ingestion, "invoke_tool", fake_invoke_tool)

    summary = ingestion.summarize_document("body", max_length=120)

    assert len(summary) <= 120
    assert summary.endswith("...")


def test_summarize_chunks_falls_back_on_invalid_json(monkeypatch):
    calls = []
    fallbacks = []

    def fake_run_chat_tool(**kwargs):
        calls.append(kwargs)
        return {"content": "not-json"}

    def fake_summarize_document(text, *, max_length, tool_slug=ingestion.DEFAULT_GEMINI_TOOL, agent=None):  # noqa: ARG001
        fallbacks.append((text, max_length))
        return f"summary:{text[:5]}"

    monkeypatch.setattr(ingestion, "run_chat_tool", fake_run_chat_tool)
    monkeypatch.setattr(ingestion, "summarize_document", fake_summarize_document)

    summaries = ingestion.summarize_chunks(["chunk-one", "chunk-two"], max_length=32, batch_size=10)

    assert summaries == ["summary:chunk", "summary:chunk"]
    assert len(calls) == 1
    assert fallbacks == [("chunk-one", 32), ("chunk-two", 32)]


def test_detect_emotions_formats_signals(monkeypatch):
    payload = {
        "emotion": {"label": "joy", "confidence": 0.9},
        "sentiment": {"label": "positive", "confidence": 0.8},
        "emotions": [
            {"label": "curiosity", "confidence": 0.5},
        ],
    }

    def fake_run_chat_tool(**kwargs):  # noqa: ARG001
        return {"content": json.dumps(payload)}

    monkeypatch.setattr(ingestion, "run_chat_tool", fake_run_chat_tool)

    signals = ingestion.detect_emotions("Hello world")

    assert len(signals) == 3
    primary = signals[0]
    assert primary["type"] == "emotion"
    assert primary["alias"] == ingestion.DEFAULT_CLASSIFIER_TOOL
    assert primary["label"] == "joy"
    assert primary["confidence"] == pytest.approx(0.9)
    sentiment = next(item for item in signals if item["type"] == "sentiment")
    assert sentiment["label"] == "positive"
    assert any(item["label"] == "curiosity" for item in signals)
    for record in signals:
        assert record["raw"] == payload
