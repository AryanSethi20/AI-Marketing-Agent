"""One chat client for every provider.

Ollama, Groq, and OpenRouter all expose OpenAI-compatible APIs, so a single
OpenAI SDK client covers everything - only base_url, api_key, and model
differ, and those come from Settings.
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Optional, Type, TypeVar

from openai import APIError, OpenAI
from pydantic import BaseModel, ValidationError

from config import Settings, get_settings

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


class LLMError(RuntimeError):
    """The LLM could not produce a usable response after retries."""


class LLMClient:
    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()
        self.model = self.settings.model_name
        self._client = OpenAI(
            base_url=self.settings.resolved_base_url,
            api_key=self.settings.resolved_api_key,
            timeout=self.settings.request_timeout_seconds,
            max_retries=0,
        )
        logger.info(
            "LLM client ready (provider=%s, model=%s, base_url=%s)",
            self.settings.model_provider,
            self.model,
            self.settings.resolved_base_url,
        )

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: Optional[float] = None,
        json_mode: bool = False,
    ) -> str:
        """Run one chat completion and return the assistant's text."""
        temp = self.settings.llm_temperature if temperature is None else temperature
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        last_error: Optional[Exception] = None
        use_json_format = json_mode
        for attempt in range(1, self.settings.llm_max_retries + 1):
            try:
                kwargs: dict = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": temp,
                }
                if use_json_format:
                    kwargs["response_format"] = {"type": "json_object"}
                response = self._client.chat.completions.create(**kwargs)
                content = (response.choices[0].message.content or "").strip()
                if not content:
                    raise LLMError("provider returned an empty completion")
                return content
            except APIError as exc:
                if use_json_format:
                    logger.warning(
                        "response_format rejected (%s), retrying without it", exc
                    )
                    use_json_format = False
                    continue
                last_error = exc
            except Exception as exc:  # noqa: BLE001 - surfaced via LLMError
                last_error = exc

            wait = min(2 ** attempt, 10)
            logger.warning(
                "LLM call failed (attempt %d/%d): %s - retrying in %ss",
                attempt,
                self.settings.llm_max_retries,
                last_error,
                wait,
            )
            time.sleep(wait)

        raise LLMError(f"LLM completion failed after retries: {last_error}")

    def complete_json(
        self,
        system_prompt: str,
        user_prompt: str,
        schema: Type[T],
        *,
        temperature: Optional[float] = None,
    ) -> T:
        """Complete and validate the reply against a Pydantic schema.

        The schema is injected into the system prompt. If the model returns
        malformed or non-conforming JSON, the parse/validation error is fed
        back to it so it can correct itself on the next attempt.
        """
        schema_json = json.dumps(schema.model_json_schema(), indent=2)
        strict_system = (
            f"{system_prompt}\n\n"
            "You MUST respond with a single valid JSON object only - no prose, "
            "no markdown fences, no commentary. The JSON must conform exactly "
            f"to this JSON Schema:\n{schema_json}"
        )

        prompt = user_prompt
        last_error: Optional[Exception] = None
        for attempt in range(1, self.settings.llm_max_retries + 1):
            raw = self.complete(
                strict_system, prompt, temperature=temperature, json_mode=True
            )
            try:
                return schema.model_validate(_extract_json(raw))
            except (json.JSONDecodeError, ValidationError, ValueError) as exc:
                last_error = exc
                logger.warning(
                    "structured output invalid (attempt %d/%d): %s",
                    attempt,
                    self.settings.llm_max_retries,
                    exc,
                )
                prompt = (
                    f"{user_prompt}\n\n"
                    "Your previous response was invalid JSON for the required "
                    f"schema. Error: {exc}\n"
                    f"Previous response:\n{raw}\n\n"
                    "Return ONLY the corrected JSON object."
                )

        raise LLMError(f"failed to obtain schema-valid JSON: {last_error}")


def _extract_json(text: str) -> dict:
    """Pull a JSON object out of model output.

    Handles raw JSON, ```json fenced blocks```, and JSON buried in prose
    (sliced from the first '{' to the last '}').
    """
    candidate = text.strip()

    fence_match = _JSON_FENCE_RE.search(candidate)
    if fence_match:
        candidate = fence_match.group(1).strip()

    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        start, end = candidate.find("{"), candidate.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        return json.loads(candidate[start : end + 1])
