import io
import tempfile
import uuid
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile, status
from PIL import Image

from app.models.api import IngestRequest, IngestResponse
from app.utils.image import normalize_image
from app.rag.chunker import chunk_text
from app.rag.reader import read_file
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

router = APIRouter(prefix="/ingest", tags=["ingestion"])


@router.post("", response_model=IngestResponse)
async def ingest_document(request: Request, body: IngestRequest):
    settings = request.app.state.settings
    embedder = request.app.state.embedder
    qdrant = request.app.state.qdrant

    chunks = chunk_text(
        body.content,
        chunk_size=settings.ingestion.chunk_size,
        overlap=settings.ingestion.chunk_overlap,
    )

    vectors = embedder.embed_texts(chunks)

    points = []
    for chunk, vector in zip(chunks, vectors):
        payload = {**body.metadata, "content": chunk}
        points.append({
            "id": str(uuid.uuid4()),
            "vector": vector,
            "payload": payload,
        })

    qdrant.upsert(
        collection_name=settings.qdrant.collection,
        points=points,
    )

    logger.info(f"Ingested {len(chunks)} chunks from document")

    return IngestResponse(document_count=1, chunk_count=len(chunks))


@router.post("/file", response_model=IngestResponse)
async def ingest_file(request: Request, file: UploadFile, source: str = "upload"):
    if not file.filename:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No filename provided")

    suffix = Path(file.filename).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content_bytes = await file.read()
        tmp.write(content_bytes)
        tmp_path = tmp.name

    try:
        text = read_file(tmp_path)
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    if not text.strip():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="File content is empty")

    body = IngestRequest(
        content=text,
        metadata={"source": source, "title": file.filename},
    )
    return await ingest_document(request, body)


@router.post("/image", response_model=IngestResponse)
async def ingest_image(
    request: Request,
    file: UploadFile = File(...),
    source: str = Form("upload"),
    title: str | None = Form(None),
    tags: str = Form(""),
):
    settings = request.app.state.settings
    embedder = request.app.state.embedder
    qdrant = request.app.state.qdrant

    if not file.filename:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No filename provided")

    content = await file.read()
    try:
        image = normalize_image(Image.open(io.BytesIO(content)))
    except Exception:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid image file")

    vector = embedder.embed_image(image)
    tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else []

    point = {
        "id": str(uuid.uuid4()),
        "vector": vector,
        "payload": {
            "content": f"Image: {title or file.filename}",
            "source": source,
            "title": title or file.filename,
            "type": "image",
            "language": "auto",
            "image_urls": [],
            "tags": ["image"] + tag_list,
        },
    }

    qdrant.upsert(collection_name=settings.qdrant.collection, points=[point])
    logger.info(f"Ingested image: {title or file.filename}")

    return IngestResponse(document_count=1, chunk_count=1)
