from PIL import Image
from sentence_transformers import SentenceTransformer

from app.models.settings import EmbeddingSettings
from app.utils.logger import setup_logger

logger = setup_logger(__name__)


class Embedder:
    def __init__(self, settings: EmbeddingSettings):
        self.settings = settings
        self.model = SentenceTransformer(
            settings.model,
            device=settings.device,
            trust_remote_code=True,
        )
        logger.info(f"Embedding model loaded: {settings.model} on {settings.device}")

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        embeddings = self.model.encode(
            texts,
            batch_size=self.settings.batch_size,
            normalize_embeddings=True,
        )
        return embeddings.tolist()

    def embed_query(self, query: str) -> list[float]:
        return self.embed_texts([query])[0]

    def embed_image(self, image: Image.Image) -> list[float]:
        embedding = self.model.encode(
            image,
            normalize_embeddings=True,
        )
        return embedding.tolist()

    def embed_images(self, images: list[Image.Image]) -> list[list[float]]:
        embeddings = self.model.encode(
            images,
            batch_size=self.settings.batch_size,
            normalize_embeddings=True,
        )
        return embeddings.tolist()
