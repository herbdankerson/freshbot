"""Shared utilities for Freshbot Prefect flows."""

from __future__ import annotations

from typing import Dict, List, Mapping, Optional, Sequence


def merge_system_prompt(
    messages: Sequence[Mapping[str, object]],
    system_prompt: Optional[str],
) -> List[Dict[str, object]]:
    """Prepend a system prompt to a sequence of chat messages."""

    payload: List[Dict[str, object]] = []
    if system_prompt:
        payload.append({"role": "system", "content": system_prompt})
    for message in messages:
        role = message.get("role")
        content = message.get("content")
        if not role:
            raise ValueError("messages must include a 'role' key")
        payload.append({"role": str(role), "content": content})
    return payload


__all__ = ["merge_system_prompt"]
