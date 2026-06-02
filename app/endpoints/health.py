from fastapi import APIRouter, Request

from app.models.api import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health_check(request: Request):
    qdrant_ok = False
    llm_ok = False

    try:
        request.app.state.qdrant.get_collections()
        qdrant_ok = True
    except Exception:
        pass

    try:
        registry = request.app.state.llm_registry
        client = registry.get_client()
        llm_ok = await client.is_available()
    except Exception:
        pass

    overall = "healthy" if qdrant_ok and llm_ok else "degraded"

    return HealthResponse(status=overall, qdrant=qdrant_ok, llm=llm_ok)
