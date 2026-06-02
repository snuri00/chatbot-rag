import json
from collections.abc import AsyncIterator

from openai import AsyncOpenAI

from app.models.settings import LLMModelConfig
from app.utils.logger import setup_logger

logger = setup_logger(__name__)


def _build_user_content(text: str, image_base64: str | None = None) -> str | list[dict]:
    if not image_base64:
        return text
    return [
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
        {"type": "text", "text": text},
    ]


class ToolCall:
    def __init__(self, name: str, arguments: dict):
        self.name = name
        self.arguments = arguments


class LLMResponse:
    def __init__(self, content: str | None = None, tool_calls: list[ToolCall] | None = None):
        self.content = content
        self.tool_calls = tool_calls or []

    @property
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0


class LLMClient:
    def __init__(self, config: LLMModelConfig):
        self.config = config
        self.client = AsyncOpenAI(
            base_url=config.api.url,
            api_key="not-needed",
        )

    def _build_messages(self, system_prompt: str, user_message: str, image_base64: str | None = None) -> list[dict]:
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": _build_user_content(user_message, image_base64)},
        ]

    async def generate_with_tools(
        self,
        system_prompt: str,
        user_message: str,
        tools: list[dict],
        image_base64: str | None = None,
    ) -> LLMResponse:
        response = await self.client.chat.completions.create(
            model=self.config.model,
            messages=self._build_messages(system_prompt, user_message, image_base64),
            tools=tools,
            temperature=self.config.inference.temperature,
            top_p=self.config.inference.top_p,
            max_tokens=self.config.inference.max_tokens,
        )
        message = response.choices[0].message

        if message.tool_calls:
            calls = []
            for tc in message.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except (json.JSONDecodeError, TypeError):
                    args = {}
                calls.append(ToolCall(name=tc.function.name, arguments=args))
            return LLMResponse(content=message.content, tool_calls=calls)

        return LLMResponse(content=message.content or "")

    async def generate(self, system_prompt: str, user_message: str, image_base64: str | None = None) -> str:
        response = await self.client.chat.completions.create(
            model=self.config.model,
            messages=self._build_messages(system_prompt, user_message, image_base64),
            temperature=self.config.inference.temperature,
            top_p=self.config.inference.top_p,
            max_tokens=self.config.inference.max_tokens,
        )
        return response.choices[0].message.content or ""

    async def generate_with_context(
        self,
        system_prompt: str,
        user_message: str,
        tool_call_id: str,
        tool_result: str,
        image_base64: str | None = None,
    ) -> str:
        messages = self._build_messages(system_prompt, user_message, image_base64)
        messages.append({
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": tool_call_id,
                "type": "function",
                "function": {"name": "rag_search", "arguments": "{}"},
            }],
        })
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": tool_result,
        })

        response = await self.client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            temperature=self.config.inference.temperature,
            top_p=self.config.inference.top_p,
            max_tokens=self.config.inference.max_tokens,
        )
        return response.choices[0].message.content or ""

    async def generate_stream(self, system_prompt: str, user_message: str, image_base64: str | None = None) -> AsyncIterator[str]:
        stream = await self.client.chat.completions.create(
            model=self.config.model,
            messages=self._build_messages(system_prompt, user_message, image_base64),
            temperature=self.config.inference.temperature,
            top_p=self.config.inference.top_p,
            max_tokens=self.config.inference.max_tokens,
            stream=True,
        )
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    async def is_available(self) -> bool:
        try:
            await self.client.models.list()
            return True
        except Exception:
            return False
