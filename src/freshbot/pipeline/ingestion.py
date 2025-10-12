"""Gateway-backed helpers used by the ingestion pipeline."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence

from freshbot.executors import invoke_tool

DEFAULT_CLASSIFIER_TOOL = "tool_qwen_chat"
DEFAULT_GEMINI_TOOL = "tool_gemini_chat"
_DEFAULT_SUMMARY_BATCH = 8
_CLASSIFIER_CHAR_LIMIT = 6000
_EMOTION_CHAR_LIMIT = 800


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

    snippet = (text or "").strip()[:_CLASSIFIER_CHAR_LIMIT]
    system_prompt = (
        "You label content for an ingestion pipeline. Respond with compact JSON containing "
        "domain (one of ['legal','code','general']), confidence (0-1), and source_labels (array of strings)."
    )
    user_payload = json.dumps({"content": snippet}, ensure_ascii=False)
    response = run_chat_tool(
        tool_slug=tool_slug,
        agent=agent,
        system_prompt=system_prompt,
        temperature=0.0,
        max_tokens=256,
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
    if len(snippet) > _EMOTION_CHAR_LIMIT:
        snippet = snippet[:_EMOTION_CHAR_LIMIT]

    system_prompt = (
        "You analyse affect. Respond with JSON containing optional 'emotion' and 'sentiment' objects with "
        "fields label (string) and confidence (0-1). You may also provide an 'emotions' array of additional "
        "{label, confidence} objects."
    )
    payload = json.dumps({"text": snippet}, ensure_ascii=False)
    response = run_chat_tool(
        tool_slug=tool_slug,
        agent=agent,
        system_prompt=system_prompt,
        temperature=0.0,
        max_tokens=256,
        messages=[{"role": "user", "content": payload}],
    )
    content = response.get("content")
    try:
        data = json.loads(content) if content else {}
    except json.JSONDecodeError:
        data = {}

    results: List[Dict[str, Any]] = []
    for key in ("emotion", "sentiment"):
        results.extend(
            _build_signal_entries(
                block=data.get(key),
                signal_type=key,
                tool_slug=tool_slug,
            )
        )

    extras = data.get("emotions")
    if isinstance(extras, list):
        for entry in extras:
            results.extend(
                _build_signal_entries(
                    block=entry,
                    signal_type="emotion",
                    tool_slug=tool_slug,
                )
            )

    for record in results:
        record["raw"] = data
    return results


def _build_signal_entries(
    *,
    block: Any,
    signal_type: str,
    tool_slug: str,
) -> List[Dict[str, Any]]:
    if block is None:
        return []
    if isinstance(block, list):
        entries: List[Dict[str, Any]] = []
        for item in block:
            entries.extend(
                _build_signal_entries(block=item, signal_type=signal_type, tool_slug=tool_slug)
            )
        return entries
    if isinstance(block, Mapping):
        label = str(block.get("label") or block.get("value") or "").strip()
        confidence_raw = block.get("confidence")
        try:
            confidence = float(confidence_raw) if confidence_raw is not None else None
        except (TypeError, ValueError):
            confidence = None
        record: Dict[str, Any] = {
            "type": signal_type,
            "alias": tool_slug,
            "model": tool_slug,
            "raw": block,
        }
        if label:
            record["label"] = label
        if confidence is not None:
            record["confidence"] = confidence
        return [record]
    value = str(block).strip()
    if not value:
        return []
    return [
        {
            "type": signal_type,
            "alias": tool_slug,
            "model": tool_slug,
            "label": value,
        }
    ]


__all__ = [
    "ChatToolError",
    "DEFAULT_CLASSIFIER_TOOL",
    "DEFAULT_GEMINI_TOOL",
    "classify_document",
    "detect_emotions",
    "run_chat_tool",
    "summarize_chunks",
    "summarize_document",
]
