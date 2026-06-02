# chatbot-rag

Multimodal RAG chatbot. FastAPI backend, Streamlit UI, Qdrant vector store, and a local LLM served via either vLLM or `llama.cpp`. Documents and images dropped into a watched workspace are chunked, embedded with Jina CLIP v2, and indexed automatically.

## Architecture

- `app/` — FastAPI service (chat, ingestion, health endpoints)
- `app/rag/` — embedder, retriever, ingestion pipeline, workspace watcher
- `app/llm/` — OpenAI-compatible client + model registry
- `ui/app.py` — Streamlit chat UI
- `configs/` — `general.yml`, `llm_models.yml`, `prompts.yml`
- `scripts/seed_met_museum.py` — sample dataset seeder
- `docker-compose.yml` — `app`, `qdrant`, and a choice of `vllm` or `llamacpp` (selected via Compose profiles)

Embeddings are 1024-dim multimodal (text + image) so the same retriever serves both modalities.

## Quickstart

Set `HF_TOKEN` in `.env` (see `.env.example`). Then pick an inference backend.

llama.cpp (default for local / smaller GPUs):

```bash
./download_model.sh
docker compose --profile llamacpp up --build
```

vLLM (full-precision, multi-GPU friendly):

```bash
docker compose --profile vllm up --build
```

- Backend: `http://localhost:8000` (FastAPI, OpenAPI docs at `/docs`)
- Qdrant: `http://localhost:6333`
- LLM server: `http://localhost:8001/v1`

Run the UI:

```bash
pip install streamlit
streamlit run ui/app.py
```

## Adding documents

Drop PDFs, DOCX, or images into `workspace/`. The watcher parses, chunks, embeds, and upserts into the `museum_documents` collection. PDFs use PyMuPDF, DOCX uses `python-docx`, images go straight through the multimodal embedder.

## Configuration

- `configs/general.yml` — Qdrant endpoint, embedding model/device, chunking, retrieval `top_k` and score threshold
- `configs/llm_models.yml` — registered models, their inference parameters, and the default
- `configs/prompts.yml` — system / RAG / fallback prompts

All endpoints in YAML are overridable via env vars (`${QDRANT_URL:...}`, `${VLLM_URL:...}`).

## Requirements

- Docker + Docker Compose
- NVIDIA GPU + container toolkit (for the LLM service)
- ~6 GB disk for the Gemma 4 E2B GGUF, more for vLLM weights
