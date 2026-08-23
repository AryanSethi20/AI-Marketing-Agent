"""Shared test doubles.

The agents only ever call ``llm.complete_json(...)``, so a tiny duck-typed
fake is enough — no need to touch the real OpenAI client in tests.
"""

from llm_factory import LLMError


class FakeLLM:
    """Stands in for LLMClient. Returns a canned object or raises LLMError."""

    def __init__(self, result=None, fail=False):
        self.result = result
        self.fail = fail
        self.calls = []

    def complete_json(self, system_prompt, user_prompt, schema, **kwargs):
        self.calls.append({"system": system_prompt, "user": user_prompt})
        if self.fail:
            raise LLMError("simulated provider failure")
        return self.result
