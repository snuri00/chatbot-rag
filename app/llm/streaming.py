import json
from collections.abc import AsyncIterator

from starlette.responses import StreamingResponse


async def _event_generator(chunks: AsyncIterator[str]) -> AsyncIterator[str]:
    async for chunk in chunks:
        data = json.dumps({"type": "content", "content": chunk})
        yield f"data: {data}\n\n"
    yield f"data: {json.dumps({'type': 'done'})}\n\n"


def create_sse_response(chunks: AsyncIterator[str]) -> StreamingResponse:
    return StreamingResponse(
        _event_generator(chunks),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
