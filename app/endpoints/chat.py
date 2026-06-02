import base64
import json

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile, status
from starlette.responses import StreamingResponse

from app.models.api import ChatRequest, ChatResponse, ClientContext, ModelInfo
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

router = APIRouter(tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
async def chat(request: Request, body: ChatRequest):
    pipeline = request.app.state.pipeline

    try:
        response = await pipeline.query(
            question=body.question,
            model_id=body.language_model,
            filters=body.filters,
            client_context=body.context,
            system_prompt_override=body.system_prompt,
            image_base64=body.image,
        )
        return response
    except KeyError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail="LLM service error")


@router.post("/chat/image", response_model=ChatResponse)
async def chat_with_image(
    request: Request,
    file: UploadFile = File(...),
    question: str = Form(...),
    language_model: str | None = Form(None),
    context_json: str | None = Form(None),
    system_prompt: str | None = Form(None),
):
    pipeline = request.app.state.pipeline

    content = await file.read()
    image_b64 = base64.b64encode(content).decode("utf-8")

    client_context = None
    if context_json:
        client_context = ClientContext.model_validate_json(context_json)

    try:
        response = await pipeline.query(
            question=question,
            model_id=language_model,
            client_context=client_context,
            system_prompt_override=system_prompt,
            image_base64=image_b64,
        )
        return response
    except KeyError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Chat image error: {e}")
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail="LLM service error")


@router.post("/chat/stream")
async def chat_stream(request: Request, body: ChatRequest):
    pipeline = request.app.state.pipeline

    try:
        stream, sources, model_label = await pipeline.query_stream(
            question=body.question,
            model_id=body.language_model,
            filters=body.filters,
            client_context=body.context,
            system_prompt_override=body.system_prompt,
            image_base64=body.image,
        )

        async def wrapped_stream():
            sources_data = [s.model_dump() for s in sources]
            yield f"data: {json.dumps({'type': 'sources', 'sources': sources_data, 'model': model_label})}\n\n"
            async for chunk in stream:
                yield f"data: {json.dumps({'type': 'content', 'content': chunk})}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"

        return StreamingResponse(
            wrapped_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
    except KeyError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Chat stream error: {e}")
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail="LLM service error")


@router.get("/models", response_model=list[ModelInfo])
async def list_models(request: Request):
    registry = request.app.state.llm_registry
    return await registry.list_models()
