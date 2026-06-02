from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchValue

from app.models.api import MetadataFilter, SourceReference
from app.models.settings import RetrievalSettings
from app.utils.logger import setup_logger

logger = setup_logger(__name__)


class Retriever:
    def __init__(self, client: QdrantClient, collection: str, settings: RetrievalSettings):
        self.client = client
        self.collection = collection
        self.settings = settings

    def search(self, query_vector: list[float], filters: MetadataFilter | None = None) -> list[SourceReference]:
        qdrant_filter = self._build_filter(filters) if filters else None

        results = self.client.query_points(
            collection_name=self.collection,
            query=query_vector,
            query_filter=qdrant_filter,
            limit=self.settings.top_k,
            score_threshold=self.settings.score_threshold,
        )

        sources = []
        for point in results.points:
            payload = point.payload or {}
            sources.append(SourceReference(
                content=payload.get("content", ""),
                title=payload.get("title"),
                source=payload.get("source", "unknown"),
                image_urls=payload.get("image_urls", []),
                score=point.score,
            ))

        if not sources:
            fallback = self.client.query_points(
                collection_name=self.collection,
                query=query_vector,
                query_filter=qdrant_filter,
                limit=self.settings.top_k,
            )
            for point in fallback.points:
                payload = point.payload or {}
                sources.append(SourceReference(
                    content=payload.get("content", ""),
                    title=payload.get("title"),
                    source=payload.get("source", "unknown"),
                    image_urls=payload.get("image_urls", []),
                    score=point.score,
                ))
            if sources:
                logger.info(f"Fallback: retrieved {len(sources)} documents (no threshold)")

        logger.info(f"Retrieved {len(sources)} documents (filter: {filters is not None})")
        return sources

    def _build_filter(self, filters: MetadataFilter) -> Filter | None:
        conditions = []

        if filters.source:
            conditions.append(FieldCondition(key="source", match=MatchValue(value=filters.source)))

        if filters.type:
            conditions.append(FieldCondition(key="type", match=MatchValue(value=filters.type)))

        if filters.language:
            conditions.append(FieldCondition(key="language", match=MatchValue(value=filters.language)))

        if filters.tags:
            conditions.append(FieldCondition(key="tags", match=MatchAny(any=filters.tags)))

        if not conditions:
            return None

        return Filter(must=conditions)
