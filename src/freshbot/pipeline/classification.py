"""Domain classification helpers that combine extension heuristics with Qwen."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

try:  # pragma: no cover - optional dependency
    from etl.tasks.intake_models import Chunk  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - tests without ETL
    from typing import Any

    Chunk = Any  # type: ignore

from .ingestion import (
    CLASSIFICATION_JSON_SCHEMA,
    DEFAULT_CLASSIFIER_TOOL,
    run_chat_tool,
)

LOGGER = logging.getLogger(__name__)

# Extensions that should be treated as source code without invoking Qwen.
CODE_EXTENSIONS = {
    ".py",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
    ".java",
    ".c",
    ".cpp",
    ".cc",
    ".cs",
    ".rs",
    ".go",
    ".rb",
    ".php",
    ".swift",
    ".kt",
    ".kts",
    ".scala",
    ".sql",
    ".sh",
    ".bash",
    ".zsh",
    ".ps1",
    ".psm1",
    ".pl",
    ".pm",
    ".r",
    ".jl",
    ".m",
    ".mm",
    ".dart",
    ".lua",
    ".yaml",
    ".yml",
    ".json",
    ".toml",
    ".ini",
    ".cfg",
    ".gradle",
    ".groovy",
    ".make",
    ".mk",
}


def is_code_extension(filename: str) -> bool:
    """Return ``True`` if the path extension maps to the code domain."""

    suffix = Path(filename).suffix.lower()
    return suffix in CODE_EXTENSIONS


def _sample_chunks(chunks: Iterable[Chunk], *, limit: int = 10, char_limit: int = 800) -> List[str]:
    samples: List[str] = []
    for index, chunk in enumerate(chunks):
        if index >= limit:
            break
        text = (chunk.text or "").strip()
        if len(text) > char_limit:
            text = text[: char_limit - 1] + "…"
        samples.append(text)
    return samples


def classify_with_qwen(
    *,
    filename: str,
    chunks: Iterable[Chunk],
    tool_slug: str = DEFAULT_CLASSIFIER_TOOL,
    agent: str | None = None,
) -> Tuple[str, float, Dict[str, object]]:
    """Invoke the Qwen classifier using the first chunk samples."""

    samples = _sample_chunks(chunks)
    payload = {
        "filename": filename,
        "samples": samples,
    }
    system_prompt = (
        "Classify the provided document into one of: general, law, code. "
        "Respond with compact JSON: {\"domain\": \"...\", \"confidence\": number, \"reason\": \"...\"}."
    )
    response = run_chat_tool(
        tool_slug=tool_slug,
        agent=agent,
        system_prompt=system_prompt,
        temperature=0.0,
        max_tokens=256,
        response_format={"type": "json_object"},
        extra_params={
            "guided_json": CLASSIFICATION_JSON_SCHEMA,
            "chat_template_kwargs": {"enable_thinking": False},
        },
        messages=[
            {
                "role": "user",
                "content": json.dumps(payload),
            }
        ],
    )
    content = response.get("content")
    try:
        parsed = json.loads(content) if content else {}
    except json.JSONDecodeError:
        LOGGER.warning("Qwen classifier returned non-JSON payload: %s", content)
        parsed = {}

    domain = str(parsed.get("domain") or "").lower()
    if domain not in {"general", "law", "code"}:
        domain = "general"
    confidence_raw = parsed.get("confidence", 0.5)
    try:
        confidence = float(confidence_raw)
    except (TypeError, ValueError):
        confidence = 0.5
    return domain, confidence, parsed


def classify_domain(
    *,
    filename: str,
    chunks: List[Chunk],
    default_domain: str = "general",
    agent: str | None = None,
) -> Tuple[str, float, Dict[str, object]]:
    """Return the domain classification and supporting metadata."""

    if is_code_extension(filename):
        return "code", 1.0, {"source": "extension", "filename": filename}

    domain, confidence, raw = classify_with_qwen(
        filename=filename,
        chunks=chunks,
        agent=agent,
    )
    if domain not in {"general", "law", "code"}:
        domain = default_domain
    result = dict(raw) if isinstance(raw, dict) else {}
    result.setdefault("source", "qwen")
    result["filename"] = filename
    result["sample_count"] = len(_sample_chunks(chunks))
    return domain, confidence, result
