"""Qdrant vector store utilities for embedding persistence."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from qdrant_client.http.exceptions import UnexpectedResponse

from src.my_agentic_chatbot.config import get_settings

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class VectorPoint:
    """Container describing a single vector destined for Qdrant."""

    point_id: str
    vector: Sequence[float]
    payload: Mapping[str, object]


def _resolve_distance(metric: str | None) -> qmodels.Distance:
    """Translate configuration distance metric strings into Qdrant enums."""

    if not metric:
        return qmodels.Distance.COSINE
    metric_normalized = metric.lower()
    if metric_normalized in {"cosine", "cos"}:
        return qmodels.Distance.COSINE
    if metric_normalized in {"dot", "dot_product"}:
        return qmodels.Distance.DOT
    if metric_normalized in {"euclid", "euclidean", "l2"}:
        return qmodels.Distance.EUCLID
    LOGGER.warning("Unknown distance metric '%s'; defaulting to cosine", metric)
    return qmodels.Distance.COSINE


class QdrantVectorStore:
    """Lightweight helper wrapping Qdrant collection management and upserts."""

    def __init__(self) -> None:
        settings = get_settings()
        self._client = QdrantClient(url=settings.qdrant_url, timeout=60.0)
        self._collections: MutableMapping[str, int] = {}

    @property
    def client(self) -> QdrantClient:
        return self._client

    @staticmethod
    def collection_name(space_name: str, kind: str) -> str:
        safe_space = space_name.replace("/", "_")
        return f"{safe_space}__{kind}"

    def ensure_collection(self, *, space_name: str, kind: str, dims: int, distance_metric: str | None) -> str:
        """Ensure the backing collection exists and matches dimensionality."""

        if dims <= 0:
            raise ValueError("Vector dimensionality must be positive")

        collection_name = self.collection_name(space_name, kind)
        cached_dims = self._collections.get(collection_name)
        if cached_dims is not None:
            if cached_dims != dims:
                raise RuntimeError(
                    f"Existing collection '{collection_name}' expected {cached_dims} dims but received {dims}"
                )
            return collection_name

        try:
            exists = self._client.collection_exists(collection_name)
        except UnexpectedResponse as exc:  # pragma: no cover - network failure
            raise RuntimeError(f"Failed to probe Qdrant collection '{collection_name}': {exc}") from exc

        if not exists:
            LOGGER.info(
                "Creating Qdrant collection '%s' (dims=%s, distance=%s)",
                collection_name,
                dims,
                distance_metric or "cosine",
            )
            self._client.create_collection(
                collection_name=collection_name,
                vectors_config=qmodels.VectorParams(
                    size=dims,
                    distance=_resolve_distance(distance_metric),
                ),
            )
            self._collections[collection_name] = dims
            return collection_name

        info = self._client.get_collection(collection_name)
        vectors_config = info.config.params.vectors
        # Vectors config may be a single VectorParams or a dict keyed by name.
        if isinstance(vectors_config, dict):
            vector_params = next(iter(vectors_config.values()))
        else:
            vector_params = vectors_config
        existing_dims = int(getattr(vector_params, "size"))
        if existing_dims != dims:
            raise RuntimeError(
                f"Collection '{collection_name}' has dimension {existing_dims} but {dims} was requested"
            )
        self._collections[collection_name] = existing_dims
        return collection_name

    def upsert_vectors(
        self,
        *,
        space_name: str,
        kind: str,
        points: Iterable[VectorPoint],
        dims: int,
        distance_metric: str | None,
    ) -> None:
        """Insert or update vectors for the given embedding space and payload kind."""

        point_list = list(points)
        if not point_list:
            return

        collection_name = self.ensure_collection(
            space_name=space_name, kind=kind, dims=dims, distance_metric=distance_metric
        )

        formatted_points = [
            qmodels.PointStruct(
                id=point.point_id,
                vector=[float(value) for value in point.vector],
                payload=dict(point.payload),
            )
            for point in point_list
        ]

        self._client.upsert(collection_name=collection_name, points=formatted_points)

    def delete_vectors(self, *, space_name: str, kind: str, point_ids: Iterable[str]) -> None:
        """Remove vectors from the backing store."""

        ids = [point_id for point_id in point_ids]
        if not ids:
            return
        collection_name = self.collection_name(space_name, kind)
        if not self._client.collection_exists(collection_name):
            return
        self._client.delete(
            collection_name=collection_name,
            points_selector=qmodels.PointIdsList(points=ids),
        )

    def fetch_vectors(
        self,
        *,
        space_name: str,
        kind: str,
        point_ids: Sequence[str],
    ) -> Dict[str, List[float]]:
        """Retrieve vectors for the specified points."""

        ids = [point_id for point_id in point_ids if point_id]
        if not ids:
            return {}

        collection_name = self.collection_name(space_name, kind)
        if not self._client.collection_exists(collection_name):
            return {}

        records = self._client.retrieve(
            collection_name=collection_name,
            ids=ids,
            with_vectors=True,
        )
        vector_map: Dict[str, List[float]] = {}
        for record in records:
            vector = getattr(record, "vector", None)
            if vector is None:
                continue
            vector_map[str(record.id)] = [float(value) for value in vector]
        return vector_map


@lru_cache(maxsize=1)
def get_qdrant_store() -> QdrantVectorStore:
    """Return a memoized Qdrant store instance."""

    return QdrantVectorStore()


__all__ = ["QdrantVectorStore", "VectorPoint", "get_qdrant_store"]
