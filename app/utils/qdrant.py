from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from app.models.settings import QdrantSettings
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

DISTANCE_MAP = {
    "Cosine": Distance.COSINE,
    "Euclid": Distance.EUCLID,
    "Dot": Distance.DOT,
}


def create_qdrant_client(settings: QdrantSettings) -> QdrantClient:
    client = QdrantClient(url=settings.url)
    _ensure_collection(client, settings)
    return client


def _ensure_collection(client: QdrantClient, settings: QdrantSettings) -> None:
    collections = [c.name for c in client.get_collections().collections]

    if settings.collection not in collections:
        distance = DISTANCE_MAP.get(settings.distance, Distance.COSINE)
        client.create_collection(
            collection_name=settings.collection,
            vectors_config=VectorParams(
                size=settings.vector_size,
                distance=distance,
            ),
        )
        logger.info(f"Collection '{settings.collection}' created ({settings.vector_size}d, {settings.distance})")
    else:
        logger.info(f"Collection '{settings.collection}' already exists")
