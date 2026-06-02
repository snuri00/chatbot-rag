import hashlib
import time
import uuid
from pathlib import Path

from PIL import Image
from watchdog.events import FileSystemEventHandler

from app.utils.image import normalize_image
from watchdog.observers import Observer

from app.rag.chunker import chunk_text
from app.rag.reader import read_file
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

TEXT_EXTENSIONS = {".pdf", ".docx", ".doc", ".txt", ".md", ".csv", ".json"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
SUPPORTED_EXTENSIONS = TEXT_EXTENSIONS | IMAGE_EXTENSIONS


def file_hash(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8192), b""):
            h.update(block)
    return h.hexdigest()


class WorkspaceHandler(FileSystemEventHandler):
    def __init__(self, embedder, qdrant_client, collection: str, chunk_size: int, chunk_overlap: int):
        self.embedder = embedder
        self.qdrant = qdrant_client
        self.collection = collection
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.processed_hashes: dict[str, str] = {}

    def _should_process(self, path: str) -> bool:
        p = Path(path)
        if p.name.startswith(".") or ".tmp" in p.name:
            return False
        if p.suffix.lower() not in SUPPORTED_EXTENSIONS:
            return False
        if not p.exists():
            return False
        return True

    def _is_image(self, path: str) -> bool:
        return Path(path).suffix.lower() in IMAGE_EXTENSIONS

    def _delete_existing(self, path: str):
        from qdrant_client.models import FieldCondition, Filter, MatchValue
        self.qdrant.delete(
            collection_name=self.collection,
            points_selector=Filter(must=[
                FieldCondition(key="file_path", match=MatchValue(value=path))
            ]),
        )

    def _ingest_image(self, path: str):
        fname = Path(path).name
        fhash = file_hash(path)

        if self.processed_hashes.get(path) == fhash:
            return

        self._delete_existing(path)

        try:
            image = normalize_image(Image.open(path))
            vector = self.embedder.embed_image(image)
        except Exception as e:
            logger.error(f"Failed to embed image {fname}: {e}")
            return

        point = {
            "id": str(uuid.uuid4()),
            "vector": vector,
            "payload": {
                "content": f"Image: {fname}",
                "source": "workspace",
                "title": fname,
                "file_path": path,
                "file_hash": fhash,
                "chunk_index": 0,
                "type": "image",
                "language": "auto",
                "image_urls": [path],
                "tags": ["image", Path(path).suffix.lower().lstrip(".")],
            },
        }

        self.qdrant.upsert(collection_name=self.collection, points=[point])
        self.processed_hashes[path] = fhash
        logger.info(f"Indexed image: {fname}")

    def _ingest_text(self, path: str):
        fname = Path(path).name
        fhash = file_hash(path)

        if self.processed_hashes.get(path) == fhash:
            return

        self._delete_existing(path)

        text = read_file(path)
        if not text.strip():
            logger.warning(f"Empty file: {fname}")
            return

        chunks = chunk_text(text, chunk_size=self.chunk_size, overlap=self.chunk_overlap)
        if not chunks:
            return

        vectors = self.embedder.embed_texts(chunks)

        points = []
        for i, (vec, chunk) in enumerate(zip(vectors, chunks)):
            points.append({
                "id": str(uuid.uuid4()),
                "vector": vec,
                "payload": {
                    "content": chunk,
                    "source": "workspace",
                    "title": fname,
                    "file_path": path,
                    "file_hash": fhash,
                    "chunk_index": i,
                    "type": "document",
                    "language": "auto",
                    "image_urls": [],
                    "tags": [Path(path).suffix.lower().lstrip(".")],
                },
            })

        self.qdrant.upsert(collection_name=self.collection, points=points)
        self.processed_hashes[path] = fhash
        logger.info(f"Indexed: {fname} ({len(points)} chunks)")

    def _ingest_file(self, path: str):
        if not self._should_process(path):
            return
        if self._is_image(path):
            self._ingest_image(path)
        else:
            self._ingest_text(path)

    def _delete_file(self, path: str):
        fname = Path(path).name
        self._delete_existing(path)
        self.processed_hashes.pop(path, None)
        logger.info(f"Removed: {fname}")

    def on_created(self, event):
        if event.is_directory:
            return
        time.sleep(1)
        self._ingest_file(event.src_path)

    def on_modified(self, event):
        if event.is_directory:
            return
        time.sleep(1)
        self._ingest_file(event.src_path)

    def on_deleted(self, event):
        if event.is_directory:
            return
        self._delete_file(event.src_path)


def scan_and_index(handler: WorkspaceHandler, workspace_dir: str):
    for path in Path(workspace_dir).rglob("*"):
        if path.is_file():
            handler._ingest_file(str(path))


def start_watcher(
    workspace_dir: str,
    embedder,
    qdrant_client,
    collection: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> tuple[Observer, WorkspaceHandler]:
    Path(workspace_dir).mkdir(parents=True, exist_ok=True)

    handler = WorkspaceHandler(embedder, qdrant_client, collection, chunk_size, chunk_overlap)
    scan_and_index(handler, workspace_dir)

    observer = Observer()
    observer.schedule(handler, workspace_dir, recursive=True)
    observer.start()

    logger.info(f"Watching workspace: {workspace_dir}")
    return observer, handler
