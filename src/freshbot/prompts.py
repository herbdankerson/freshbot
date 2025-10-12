"""Helpers for loading prompt templates from ParadeDB."""

from __future__ import annotations

from typing import Optional

from sqlalchemy import text

from freshbot.db import get_engine


def fetch_prompt(name: str) -> Optional[str]:
    """Return the prompt content for ``name`` from ``cfg.prompts``."""

    engine = get_engine()
    with engine.connect() as connection:
        row = connection.execute(
            text("SELECT content FROM cfg.prompts WHERE name = :name"),
            {"name": name},
        ).scalar_one_or_none()
    return row


__all__ = ["fetch_prompt"]
