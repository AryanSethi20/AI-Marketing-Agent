"""Tests for environment-driven configuration."""

import pytest

from config import Settings


def _settings(**overrides):
    # _env_file=None keeps the developer's local .env out of the tests.
    return Settings(_env_file=None, **overrides)


def test_unknown_provider_rejected():
    with pytest.raises(ValueError):
        _settings(model_provider="banana")


def test_provider_is_normalised():
    assert _settings(model_provider="  OpenAI ").model_provider == "openai"


def test_ollama_needs_no_api_key():
    s = _settings(model_provider="ollama")
    assert s.resolved_api_key  # placeholder, but non-empty for the SDK
    assert s.resolved_base_url.startswith("http://localhost:11434")


def test_remote_provider_requires_api_key():
    s = _settings(model_provider="groq", openai_api_key=None)
    with pytest.raises(ValueError):
        _ = s.resolved_api_key


def test_custom_provider_requires_base_url():
    s = _settings(model_provider="custom", openai_api_key="k")
    with pytest.raises(ValueError):
        _ = s.resolved_base_url


def test_known_provider_base_urls():
    assert _settings(model_provider="groq").resolved_base_url == (
        "https://api.groq.com/openai/v1"
    )
