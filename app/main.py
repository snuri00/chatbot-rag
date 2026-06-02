from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI

from app.config import load_default_model, load_llm_models, load_prompts, load_settings
from app.endpoints.chat import router as chat_router
from app.endpoints.health import router as health_router
from app.endpoints.ingestion import router as ingestion_router
from app.llm.registry import LLMRegistry
from app.rag.embedder import Embedder
from app.rag.pipeline import RAGPipeline
from app.rag.retriever import Retriever
from app.rag.watcher import start_watcher
from app.utils.logger import setup_logger
from app.utils.qdrant import create_qdrant_client

WORKSPACE_DIR = "/app/workspace"

logger = setup_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = load_settings()
    llm_models = load_llm_models()
    default_model = load_default_model()
    prompts = load_prompts()

    logger.info("Initializing embedding model...")
    embedder = Embedder(settings.embedding)

    logger.info("Connecting to Qdrant...")
    qdrant = create_qdrant_client(settings.qdrant)

    retriever = Retriever(qdrant, settings.qdrant.collection, settings.retrieval)
    llm_registry = LLMRegistry(llm_models, default_model)
    pipeline = RAGPipeline(embedder, retriever, llm_registry, prompts)

    app.state.settings = settings
    app.state.embedder = embedder
    app.state.qdrant = qdrant
    app.state.retriever = retriever
    app.state.llm_registry = llm_registry
    app.state.pipeline = pipeline

    logger.info("Starting workspace watcher...")
    observer, _ = start_watcher(
        workspace_dir=WORKSPACE_DIR,
        embedder=embedder,
        qdrant_client=qdrant,
        collection=settings.qdrant.collection,
        chunk_size=settings.ingestion.chunk_size,
        chunk_overlap=settings.ingestion.chunk_overlap,
    )

    logger.info("Museum RAG Chatbot ready")
    yield
    observer.stop()
    observer.join()
    logger.info("Shutting down")


app = FastAPI(
    title="Museum RAG Chatbot",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(health_router)
app.include_router(chat_router)
app.include_router(ingestion_router)


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
