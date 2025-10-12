"""Gateway helpers for Freshbot connectors."""

from .registry import GatewayConfig, load_gateway
from .ollama import embed_code_texts, embeddings_enabled
from .openai import OpenAIGatewayClient, chat_completion

__all__ = [
    "GatewayConfig",
    "OpenAIGatewayClient",
    "chat_completion",
    "embed_code_texts",
    "embeddings_enabled",
    "load_gateway",
]
