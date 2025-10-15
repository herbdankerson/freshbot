"""Gateway-backed helpers used by the ingestion pipeline."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence

from freshbot.executors import invoke_tool
from freshbot.gateways.openai import chat_completion

DEFAULT_CLASSIFIER_TOOL = "tool_qwen_chat"
DEFAULT_GEMINI_TOOL = "tool_gemini_chat"
LITELLM_GATEWAY_ALIAS = "connector_gemini_litellm"
DEFAULT_EMOTION_MODEL = "emo-twitter"
DEFAULT_SENTIMENT_MODEL = "sent-twitter"
_DEFAULT_SUMMARY_BATCH = 8
CLASSIFIER_CHAR_LIMIT = 6000
EMOTION_CHAR_LIMIT = 800
CLASSIFICATION_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "domain": {"type": "string", "enum": ["general", "law", "code"]},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "source_labels": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
    "required": ["domain", "confidence"],
    "additionalProperties": False,
}
EMOTION_SIGNAL_SCHEMA = {
    "type": "object",
    "properties": {
        "label": {"type": "string"},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
    },
    "additionalProperties": False,
}
EMOTION_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "emotion": EMOTION_SIGNAL_SCHEMA,
        "sentiment": EMOTION_SIGNAL_SCHEMA,
        "emotions": {
            "type": "array",
            "items": EMOTION_SIGNAL_SCHEMA,
        },
    },
    "additionalProperties": False,
}


LOGGER = logging.getLogger(__name__)


class ChatToolError(RuntimeError):
    """Raised when a chat tool returns an unexpected payload."""


def _normalise_messages(messages: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    normalised: List[Dict[str, Any]] = []
    for item in messages:
        if not isinstance(item, Mapping):
            raise TypeError("Chat messages must be mapping objects with role/content")
        role = item.get("role")
        content = item.get("content")
        if not role:
            raise ValueError("Chat message missing role")
        normalised.append({"role": str(role), "content": "" if content is None else content})
    return normalised


def run_chat_tool(
    *,
    tool_slug: str,
    messages: Sequence[Mapping[str, Any]],
    agent: Optional[str] = None,
    system_prompt: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    response_format: Optional[Mapping[str, Any]] = None,
    extra_params: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Invoke a registered chat tool and return its structured response."""

    payload: Dict[str, Any] = {"messages": _normalise_messages(messages)}
    if system_prompt:
        payload["system_prompt"] = system_prompt
    if temperature is not None:
        payload["temperature"] = float(temperature)
    if max_tokens is not None:
        payload["max_tokens"] = int(max_tokens)
    if response_format is not None:
        payload["response_format"] = dict(response_format)
    if extra_params is not None:
        payload["extra_params"] = dict(extra_params)

    try:
        invocation = invoke_tool(tool_slug, payload=payload, agent=agent)
        result = invocation.result
    except Exception as exc:  # pragma: no cover - network fallback
        LOGGER.warning(
            "Tool invocation failed; using stub response",
            extra={"tool": tool_slug, "error": str(exc)},
        )
        if tool_slug == DEFAULT_CLASSIFIER_TOOL:
            result = {
                "content": json.dumps(
                    {"domain": "general", "confidence": 0.1, "source_labels": []}
                )
            }
        elif tool_slug == DEFAULT_GEMINI_TOOL:
            result = {"content": ""}
        else:
            result = {"content": ""}
    if not isinstance(result, Mapping):
        raise ChatToolError(f"Tool '{tool_slug}' returned non-mapping payload: {result!r}")
    return dict(result)


def classify_document(
    text: str,
    *,
    tool_slug: str = DEFAULT_CLASSIFIER_TOOL,
    agent: Optional[str] = None,
) -> Dict[str, Any]:
    """Return the primary domain classification for ``text``."""

    snippet = (text or "").strip()[:CLASSIFIER_CHAR_LIMIT]
    system_prompt = (
        "You label content for an ingestion pipeline. Respond with compact JSON containing "
        "domain (one of ['legal','code','general']), confidence (0-1), and source_labels (array of strings)."
    )
    user_payload = json.dumps({"content": snippet}, ensure_ascii=False)
    extra_params = {
        "guided_json": CLASSIFICATION_JSON_SCHEMA,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    response = run_chat_tool(
        tool_slug=tool_slug,
        agent=agent,
        system_prompt=system_prompt,
        temperature=0.0,
        max_tokens=256,
        response_format={"type": "json_object"},
        extra_params=extra_params,
        messages=[{"role": "user", "content": user_payload}],
    )
    content = response.get("content")
    if not content:
        raise ChatToolError("Classifier did not return content")
    try:
        payload = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ChatToolError("Classifier returned non-JSON payload") from exc

    domain = str(payload.get("domain", "general")).lower()
    if domain not in {"legal", "code", "general"}:
        domain = "general"
    confidence_raw = payload.get("confidence", 0.5)
    try:
        confidence = float(confidence_raw)
    except (TypeError, ValueError):
        confidence = 0.5
    labels_raw = payload.get("source_labels")
    if isinstance(labels_raw, Iterable) and not isinstance(labels_raw, (str, bytes)):
        source_labels = [str(label) for label in labels_raw if str(label).strip()]
    else:
        source_labels = []
    return {
        "domain": domain,
        "confidence": confidence,
        "source_labels": source_labels,
        "raw": payload,
    }


def summarize_document(
    text: str,
    *,
    max_length: int = 600,
    tool_slug: str = DEFAULT_GEMINI_TOOL,
    agent: Optional[str] = None,
) -> str:
    """Return a concise summary capped at ``max_length`` characters."""

    body = (text or "").strip()
    if not body:
        return ""
    system_prompt = (
        "You summarise documents for retrieval. Produce a factual summary under the requested character limit. "
        "Do not preface the answer and avoid bullet lists unless the source is already structured."
    )
    user_payload = json.dumps({"max_length": max_length, "content": body}, ensure_ascii=False)
    response = run_chat_tool(
        tool_slug=tool_slug,
        agent=agent,
        system_prompt=system_prompt,
        temperature=0.1,
        max_tokens=max(128, max_length),
        messages=[{"role": "user", "content": user_payload}],
    )
    summary = str(response.get("content") or "").strip()
    if len(summary) > max_length:
        summary = summary[: max_length - 3].rstrip() + "..."
    return summary


def summarize_chunks(
    texts: Sequence[str],
    *,
    max_length: int = 256,
    batch_size: int = _DEFAULT_SUMMARY_BATCH,
    tool_slug: str = DEFAULT_GEMINI_TOOL,
    agent: Optional[str] = None,
) -> List[str]:
    """Summarise chunked passages using batched chat calls."""

    if not texts:
        return []

    summaries: List[str] = []
    for start in range(0, len(texts), batch_size):
        batch = list(texts[start : start + batch_size])
        payload = {
            "max_length": max_length,
            "items": [
                {"index": index, "text": text}
                for index, text in enumerate(batch)
            ],
        }
        system_prompt = (
            "You summarise passages for retrieval. For each item respond with a concise summary under the provided "
            "character limit. Return a JSON array of strings ordered by the item indexes."
        )
        response = run_chat_tool(
            tool_slug=tool_slug,
            agent=agent,
            system_prompt=system_prompt,
            temperature=0.1,
            max_tokens=max(128, max_length * len(batch)),
            messages=[{"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
        )
        content = response.get("content")
        try:
            data = json.loads(content) if content else []
        except json.JSONDecodeError:
            data = []
        if isinstance(data, Mapping) and "summaries" in data:
            candidate = data.get("summaries")
            data = candidate if isinstance(candidate, list) else []
        summaries.extend(
            _resolve_summary_entry(
                candidate=data[idx] if isinstance(data, list) and idx < len(data) else None,
                fallback=batch[idx],
                max_length=max_length,
                tool_slug=tool_slug,
                agent=agent,
            )
            for idx in range(len(batch))
        )
    return summaries


def _resolve_summary_entry(
    *,
    candidate: Any,
    fallback: str,
    max_length: int,
    tool_slug: str,
    agent: Optional[str],
) -> str:
    summary = ""
    if isinstance(candidate, str):
        summary = candidate.strip()
    elif isinstance(candidate, Mapping):
        summary = str(candidate.get("summary") or candidate.get("content") or "").strip()
    if not summary:
        summary = summarize_document(fallback, max_length=max_length, tool_slug=tool_slug, agent=agent)
    if len(summary) > max_length:
        summary = summary[: max_length - 3].rstrip() + "..."
    return summary


def detect_emotions(
    text: str,
    *,
    tool_slug: str = DEFAULT_CLASSIFIER_TOOL,
    agent: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Return emotion/sentiment signals derived from ``text``."""

    snippet = (text or "").strip()
    if not snippet:
        return []
    if len(snippet) > EMOTION_CHAR_LIMIT:
        snippet = snippet[:EMOTION_CHAR_LIMIT]

    emotion_payload = _litellm_emotion_payload(
        snippet,
        model_name=DEFAULT_EMOTION_MODEL,
        agent=agent,
    )
    sentiment_payload = _litellm_emotion_payload(
        snippet,
        model_name=DEFAULT_SENTIMENT_MODEL,
        agent=agent,
    )

    results: List[Dict[str, Any]] = []
    results.extend(
        _signals_from_payload(
            emotion_payload,
            signal_type="emotion",
            model_name=DEFAULT_EMOTION_MODEL,
        )
    )
    results.extend(
        _signals_from_payload(
            sentiment_payload,
            signal_type="sentiment",
            model_name=DEFAULT_SENTIMENT_MODEL,
        )
    )

    extras = (emotion_payload or {}).get("emotions")
    if isinstance(extras, list):
        for entry in extras:
            results.extend(
                _signals_from_payload(
                    entry,
                    signal_type="emotion",
                    model_name=DEFAULT_EMOTION_MODEL,
                    raw_payload=emotion_payload,
                    allow_direct_entry=True,
                )
            )

    return results


def _litellm_emotion_payload(
    text: str,
    *,
    model_name: str,
    agent: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Invoke the LiteLLM router with an Ollama-backed classifier and return parsed JSON."""

    try:
        response = chat_completion(
            [{"role": "user", "content": text}],
            gateway_alias=LITELLM_GATEWAY_ALIAS,
            model_override=model_name,
            temperature=0.0,
            max_tokens=128,
            response_format={"type": "json_object"},
            extra_params={"chat_template_kwargs": {"enable_thinking": False}},
        )
    except Exception as exc:  # pragma: no cover - network failures
        LOGGER.warning(
            "Emotion model %s invocation failed",
            model_name,
            extra={"error": str(exc)},
        )
        return None

    content = response.get("content")
    if not content or "Stubbed response" in content:
        LOGGER.warning(
            "Emotion model %s yielded no usable content",
            model_name,
            extra={"content": content},
        )
        return None

    try:
        payload = json.loads(content)
    except json.JSONDecodeError as exc:
        LOGGER.warning(
            "Emotion model %s returned non-JSON payload",
            model_name,
            extra={"content": content, "error": str(exc)},
        )
        return None

    if isinstance(payload, MutableMapping):
        payload = dict(payload)
        payload.setdefault("model", model_name)
        payload.setdefault("raw_content", content)
    return payload  # type: ignore[return-value]


def _signals_from_payload(
    payload: Optional[Mapping[str, Any]],
    *,
    signal_type: str,
    model_name: str,
    raw_payload: Optional[Mapping[str, Any]] = None,
    allow_direct_entry: bool = False,
) -> List[Dict[str, Any]]:
    """Normalise classifier JSON into the emotion signal schema."""

    if payload is None:
        return []

    data = dict(payload)
    raw_payload = raw_payload or data

    if allow_direct_entry and {"label", "confidence"} <= data.keys():
        label = data.get("label")
        confidence = data.get("confidence")
    else:
        label = data.get(signal_type) or data.get("label")
        confidence = data.get("confidence")

    if not label:
        return []
    try:
        confidence_val = float(confidence) if confidence is not None else None
    except (TypeError, ValueError):
        confidence_val = None

    signal = {
        "type": signal_type,
        "label": str(label),
        "model": model_name,
        "raw": raw_payload,
    }
    if confidence_val is not None:
        signal["confidence"] = confidence_val
    return [signal]


__all__ = [
    "ChatToolError",
    "DEFAULT_CLASSIFIER_TOOL",
    "DEFAULT_GEMINI_TOOL",
    "CLASSIFICATION_JSON_SCHEMA",
    "EMOTION_JSON_SCHEMA",
    "classify_document",
    "detect_emotions",
    "run_chat_tool",
    "summarize_chunks",
    "summarize_document",
]
