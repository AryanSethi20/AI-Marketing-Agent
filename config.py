"""Application configuration.

Loads all runtime settings strictly from environment variables (optionally via a
local ``.env`` file). No secrets are ever hardcoded in the codebase.

Supported providers
-------------------
- ``ollama``     : Local models served by Ollama's OpenAI-compatible endpoint.
- ``openai``     : Official OpenAI API.
- ``groq``       : Groq's free/fast OpenAI-compatible API.
- ``openrouter`` : OpenRouter's OpenAI-compatible API.
- ``custom``     : Any OpenAI-compatible endpoint via ``CUSTOM_BASE_URL``.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Default base URLs for well-known OpenAI-compatible providers.
_PROVIDER_BASE_URLS: dict[str, str] = {
    "openai": "https://api.openai.com/v1",
    "groq": "https://api.groq.com/openai/v1",
    "openrouter": "https://openrouter.ai/api/v1",
}

_SUPPORTED_PROVIDERS = {"ollama", "openai", "groq", "openrouter", "custom"}


class Settings(BaseSettings):
    """Strongly-typed application settings sourced from the environment."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Model routing -----------------------------------------------------
    model_provider: str = Field(
        default="ollama",
        description="One of: ollama | openai | groq | openrouter | custom",
    )
    model_name: str = Field(
        default="qwen2.5:32b",
        description="Model identifier understood by the chosen provider.",
    )

    # --- Credentials & endpoints (never hardcoded) -------------------------
    openai_api_key: Optional[str] = Field(
        default=None,
        description="API key for OpenAI/Groq/OpenRouter/custom endpoints.",
    )
    ollama_base_url: str = Field(
        default="http://localhost:11434/v1",
        description="Base URL of the local Ollama OpenAI-compatible server.",
    )
    custom_base_url: Optional[str] = Field(
        default=None,
        description="Base URL when MODEL_PROVIDER=custom.",
    )

    # --- Generation behaviour ----------------------------------------------
    llm_temperature: float = Field(default=0.4, ge=0.0, le=2.0)
    llm_max_retries: int = Field(default=3, ge=1, le=10)
    request_timeout_seconds: float = Field(default=120.0, gt=0)

    # --- Search behaviour ---------------------------------------------------
    max_search_results: int = Field(default=8, ge=1, le=25)

    @field_validator("model_provider")
    @classmethod
    def _validate_provider(cls, value: str) -> str:
        provider = value.strip().lower()
        if provider not in _SUPPORTED_PROVIDERS:
            raise ValueError(
                f"MODEL_PROVIDER must be one of {sorted(_SUPPORTED_PROVIDERS)}, "
                f"got {value!r}."
            )
        return provider

    # --- Derived helpers ----------------------------------------------------
    @property
    def resolved_base_url(self) -> str:
        """Return the effective OpenAI-compatible base URL for the provider."""
        if self.model_provider == "ollama":
            return self.ollama_base_url.rstrip("/")
        if self.model_provider == "custom":
            if not self.custom_base_url:
                raise ValueError(
                    "MODEL_PROVIDER=custom requires CUSTOM_BASE_URL to be set."
                )
            return self.custom_base_url.rstrip("/")
        return _PROVIDER_BASE_URLS[self.model_provider]

    @property
    def resolved_api_key(self) -> str:
        """Return the API key to send with requests.

        Ollama ignores auth, but the OpenAI SDK requires a non-empty key, so a
        harmless placeholder is used locally. Remote providers must supply a
        real key via ``OPENAI_API_KEY``.
        """
        if self.model_provider == "ollama":
            return self.openai_api_key or "ollama-local"
        if not self.openai_api_key:
            raise ValueError(
                f"MODEL_PROVIDER={self.model_provider!r} requires OPENAI_API_KEY "
                "to be set in the environment or .env file."
            )
        return self.openai_api_key


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached, validated :class:`Settings` instance."""
    return Settings()
