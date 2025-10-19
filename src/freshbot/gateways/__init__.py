"""Gateway helpers for Freshbot connectors."""

from .registry import GatewayConfig, load_gateway
from .ollama import embed_code_texts, embeddings_enabled
from .embeddings import embed_texts
from .qdrant import QdrantVectorStore, VectorPoint, get_qdrant_store
from .openai import OpenAIGatewayClient, chat_completion

__all__ = [
    "GatewayConfig",
    "OpenAIGatewayClient",
    "chat_completion",
    "embed_code_texts",
    "embed_texts",
    "embeddings_enabled",
    "get_qdrant_store",
    "QdrantVectorStore",
    "VectorPoint",
    "load_gateway",
]
