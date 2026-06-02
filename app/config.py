import os
import re
from pathlib import Path

import yaml

from app.models.settings import (
    AppSettings,
    LLMModelConfig,
    PromptConfig,
)

CONFIG_DIR = Path(__file__).parent.parent / "configs"


def _resolve_env_vars(value: str) -> str:
    pattern = r"\$\{(\w+):([^}]*)\}"
    def replacer(match):
        env_var, default = match.group(1), match.group(2)
        return os.environ.get(env_var, default)
    return re.sub(pattern, replacer, value)


def _resolve_dict(data: dict) -> dict:
    resolved = {}
    for key, value in data.items():
        if isinstance(value, str):
            resolved[key] = _resolve_env_vars(value)
        elif isinstance(value, dict):
            resolved[key] = _resolve_dict(value)
        elif isinstance(value, list):
            resolved[key] = [_resolve_env_vars(v) if isinstance(v, str) else v for v in value]
        else:
            resolved[key] = value
    return resolved


def _load_yaml(filename: str) -> dict:
    filepath = CONFIG_DIR / filename
    with open(filepath) as f:
        data = yaml.safe_load(f)
    return _resolve_dict(data)


def load_settings() -> AppSettings:
    data = _load_yaml("general.yml")
    return AppSettings(**data)


def load_llm_models() -> dict[str, LLMModelConfig]:
    data = _load_yaml("llm_models.yml")
    models = {}
    for model_id, model_data in data.get("models", {}).items():
        models[model_id] = LLMModelConfig(**model_data)
    return models


def load_default_model() -> str:
    data = _load_yaml("llm_models.yml")
    return data.get("default_model", "")


def load_prompts() -> dict[str, PromptConfig]:
    data = _load_yaml("prompts.yml")
    prompts = {}
    for prompt_id, prompt_data in data.items():
        prompts[prompt_id] = PromptConfig(**prompt_data)
    return prompts
