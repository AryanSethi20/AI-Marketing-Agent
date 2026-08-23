"""Runtime settings, loaded from environment variables / .env only."""

from __future__ import annotations

from functools import lru_cache
from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROVIDER_BASE_URLS = {
    "openai": "https://api.openai.com/v1",
    "groq": "https://api.groq.com/openai/v1",
    "openrouter": "https://openrouter.ai/api/v1",
}

SUPPORTED_PROVIDERS = {"ollama", "openai", "groq", "openrouter", "custom"}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # provider routing
    model_provider: str = "ollama"
    model_name: str = "qwen2.5:32b"

    # credentials / endpoints
    openai_api_key: Optional[str] = None
    ollama_base_url: str = "http://localhost:11434/v1"
    custom_base_url: Optional[str] = None

    # generation + search tuning
    llm_temperature: float = Field(default=0.6, ge=0.0, le=2.0)
    llm_max_retries: int = Field(default=3, ge=1, le=10)
    # Default timeout is set to 900 seconds a 32b parameters model can take a long time to respond to requests
    request_timeout_seconds: float = Field(default=900.0, gt=0)
    max_search_results: int = Field(default=8, ge=1, le=25)

    @field_validator("model_provider")
    @classmethod
    def _validate_provider(cls, value: str) -> str:
        provider = value.strip().lower()
        if provider not in SUPPORTED_PROVIDERS:
            raise ValueError(
                f"MODEL_PROVIDER must be one of {sorted(SUPPORTED_PROVIDERS)}, "
                f"got {value!r}"
            )
        return provider

    @property
    def resolved_base_url(self) -> str:
        if self.model_provider == "ollama":
            return self.ollama_base_url.rstrip("/")
        if self.model_provider == "custom":
            if not self.custom_base_url:
                raise ValueError("MODEL_PROVIDER=custom requires CUSTOM_BASE_URL")
            return self.custom_base_url.rstrip("/")
        return PROVIDER_BASE_URLS[self.model_provider]

    @property
    def resolved_api_key(self) -> str:
        # Ollama ignores auth, but the OpenAI SDK refuses an empty key,
        # so send a placeholder when running locally.
        if self.model_provider == "ollama":
            return self.openai_api_key or "ollama-local"
        if not self.openai_api_key:
            raise ValueError(
                f"MODEL_PROVIDER={self.model_provider} requires OPENAI_API_KEY"
            )
        return self.openai_api_key


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
