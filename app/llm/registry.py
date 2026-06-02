from app.llm.client import LLMClient
from app.models.api import ModelInfo
from app.models.settings import LLMModelConfig
from app.utils.logger import setup_logger

logger = setup_logger(__name__)


class LLMRegistry:
    def __init__(self, models: dict[str, LLMModelConfig], default_model: str):
        self.configs = models
        self.default_model = default_model
        self.clients: dict[str, LLMClient] = {}

        for model_id, config in models.items():
            self.clients[model_id] = LLMClient(config)

        logger.info(f"LLM Registry initialized with {len(self.clients)} models (default: {default_model})")

    def get_client(self, model_id: str | None = None) -> LLMClient:
        model_id = model_id or self.default_model
        if model_id not in self.clients:
            raise KeyError(f"Model '{model_id}' not found. Available: {list(self.clients.keys())}")
        return self.clients[model_id]

    def get_config(self, model_id: str | None = None) -> LLMModelConfig:
        model_id = model_id or self.default_model
        return self.configs[model_id]

    async def list_models(self) -> list[ModelInfo]:
        models = []
        for model_id, config in self.configs.items():
            client = self.clients[model_id]
            available = await client.is_available()
            models.append(ModelInfo(
                id=model_id,
                label=config.label,
                is_multimodal=config.is_multimodal,
                available=available,
            ))
        return models
