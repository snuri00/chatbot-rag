import base64
import io
from collections.abc import AsyncIterator

from PIL import Image

from app.llm.registry import LLMRegistry
from app.llm.tools import ALL_TOOLS, TEXT_TOOLS
from app.models.api import ChatResponse, ClientContext, MetadataFilter, SourceReference
from app.models.settings import PromptConfig
from app.rag.embedder import Embedder
from app.rag.retriever import Retriever
from app.utils.logger import setup_logger

logger = setup_logger(__name__)


class RAGPipeline:
    def __init__(
        self,
        embedder: Embedder,
        retriever: Retriever,
        llm_registry: LLMRegistry,
        prompts: dict[str, PromptConfig],
    ):
        self.embedder = embedder
        self.retriever = retriever
        self.llm_registry = llm_registry
        self.prompts = prompts

    def _get_system_prompt(self, prompt_key: str = "museum_guide", override: str | None = None) -> str:
        if override:
            return override
        prompt_config = self.prompts.get(prompt_key, self.prompts.get("museum_guide"))
        if not prompt_config:
            return "You are a museum and cultural heritage guide."
        return prompt_config.system.replace("\n    Context:\n    {context}", "").replace("Context:\n{context}", "").strip()

    def _build_tool_result(self, sources: list[SourceReference], client_context: ClientContext | None = None) -> str:
        parts = []

        if client_context:
            ctx_lines = []
            if client_context.metadata:
                for key, value in client_context.metadata.items():
                    ctx_lines.append(f"  {key}: {value}")
            if client_context.keywords:
                ctx_lines.append(f"  keywords: {', '.join(client_context.keywords)}")
            if ctx_lines:
                parts.append("Client provided information:\n" + "\n".join(ctx_lines))

        if not sources:
            parts.append("No matching results found in the knowledge base.")
        else:
            for i, src in enumerate(sources, 1):
                confidence = int(src.score * 100)
                header = f"[{i}]"
                if src.title:
                    header += f" {src.title}"
                header += f" (source: {src.source}, match: {confidence}%)"
                parts.append(f"{header}\n{src.content}")

        return "\n\n---\n\n".join(parts)

    def _decode_image(self, image_base64: str) -> Image.Image:
        image_bytes = base64.b64decode(image_base64)
        return Image.open(io.BytesIO(image_bytes)).convert("RGB")

    def _compress_image_base64(self, image_base64: str, max_size: int = 1920, quality: int = 90) -> str:
        image = self._decode_image(image_base64)
        if max(image.size) > max_size:
            image.thumbnail((max_size, max_size))
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=quality)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def _merge_sources(self, *source_lists: list[SourceReference], max_results: int = 5) -> list[SourceReference]:
        seen = set()
        merged = []
        all_sources = [src for sources in source_lists for src in sources]
        for src in sorted(all_sources, key=lambda s: s.score, reverse=True):
            key = src.content.strip()
            if key not in seen:
                seen.add(key)
                merged.append(src)
            if len(merged) >= max_results:
                break
        return merged

    def _execute_rag_search(self, query: str, filters: MetadataFilter | None, client_context: ClientContext | None) -> list[SourceReference]:
        query_vector = self.embedder.embed_query(query)
        sources = self.retriever.search(query_vector, filters)

        if client_context and (client_context.keywords or client_context.metadata):
            context_query = " ".join(client_context.keywords + list(client_context.metadata.values()))
            context_vector = self.embedder.embed_query(context_query)
            context_sources = self.retriever.search(context_vector, filters)
            sources = self._merge_sources(sources, context_sources)

        return sources

    def _execute_image_search(self, image_base64: str, query: str, filters: MetadataFilter | None, client_context: ClientContext | None) -> list[SourceReference]:
        image = self._decode_image(image_base64)
        image_vector = self.embedder.embed_image(image)
        image_sources = self.retriever.search(image_vector, filters)

        text_sources = self._execute_rag_search(query, filters, client_context)
        return self._merge_sources(text_sources, image_sources)

    async def query(
        self,
        question: str,
        model_id: str | None = None,
        filters: MetadataFilter | None = None,
        client_context: ClientContext | None = None,
        system_prompt_override: str | None = None,
        image_base64: str | None = None,
    ) -> ChatResponse:
        client = self.llm_registry.get_client(model_id)
        config = self.llm_registry.get_config(model_id)
        system_prompt = self._get_system_prompt(override=system_prompt_override)

        tools = ALL_TOOLS if (image_base64 and config.is_multimodal) else TEXT_TOOLS
        vision_base64 = self._compress_image_base64(image_base64) if (image_base64 and config.is_multimodal) else None

        try:
            llm_response = await client.generate_with_tools(
                system_prompt, question, tools, image_base64=vision_base64,
            )
        except Exception as e:
            logger.warning(f"Tool calling failed, falling back to direct generation: {e}")
            answer = await client.generate(system_prompt, question, image_base64=vision_base64)
            return ChatResponse(answer=answer, sources=[], model=config.label)

        if not llm_response.has_tool_calls:
            logger.info(f"No tool call — direct response: model={config.model}")
            return ChatResponse(answer=llm_response.content or "", sources=[], model=config.label)

        tool_call = llm_response.tool_calls[0]
        tool_query = tool_call.arguments.get("query", question)
        logger.info(f"Tool called: {tool_call.name}(query='{tool_query}')")

        if tool_call.name == "rag_image_search" and image_base64:
            sources = self._execute_image_search(image_base64, tool_query, filters, client_context)
        else:
            sources = self._execute_rag_search(tool_query, filters, client_context)

        tool_result = self._build_tool_result(sources, client_context)

        try:
            answer = await client.generate_with_context(
                system_prompt, question,
                tool_call_id="call_rag",
                tool_result=tool_result,
                image_base64=vision_base64,
            )
        except Exception as e:
            logger.warning(f"Tool result generation failed, using simple generate: {e}")
            context_prompt = f"{system_prompt}\n\nSearch results:\n{tool_result}"
            answer = await client.generate(context_prompt, question, image_base64=vision_base64)

        logger.info(f"RAG query completed: model={config.model}, tool={tool_call.name}, sources={len(sources)}")
        return ChatResponse(answer=answer, sources=sources, model=config.label)

    async def query_stream(
        self,
        question: str,
        model_id: str | None = None,
        filters: MetadataFilter | None = None,
        client_context: ClientContext | None = None,
        system_prompt_override: str | None = None,
        image_base64: str | None = None,
    ) -> tuple[AsyncIterator[str], list[SourceReference], str]:
        client = self.llm_registry.get_client(model_id)
        config = self.llm_registry.get_config(model_id)
        system_prompt = self._get_system_prompt(override=system_prompt_override)

        tools = ALL_TOOLS if (image_base64 and config.is_multimodal) else TEXT_TOOLS
        vision_base64 = self._compress_image_base64(image_base64) if (image_base64 and config.is_multimodal) else None

        try:
            llm_response = await client.generate_with_tools(
                system_prompt, question, tools, image_base64=vision_base64,
            )
        except Exception:
            stream = client.generate_stream(system_prompt, question, image_base64=vision_base64)
            return stream, [], config.label

        if not llm_response.has_tool_calls:
            async def direct_stream():
                if llm_response.content:
                    yield llm_response.content
            return direct_stream(), [], config.label

        tool_call = llm_response.tool_calls[0]
        tool_query = tool_call.arguments.get("query", question)

        if tool_call.name == "rag_image_search" and image_base64:
            sources = self._execute_image_search(image_base64, tool_query, filters, client_context)
        else:
            sources = self._execute_rag_search(tool_query, filters, client_context)

        tool_result = self._build_tool_result(sources, client_context)
        context_prompt = f"{system_prompt}\n\nSearch results:\n{tool_result}"
        stream = client.generate_stream(context_prompt, question, image_base64=vision_base64)

        return stream, sources, config.label
