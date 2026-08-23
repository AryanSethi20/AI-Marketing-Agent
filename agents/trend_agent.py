"""Turns raw live search results into a clean list of trending topics."""

from __future__ import annotations

import logging
from typing import List

from pydantic import BaseModel, Field

from llm_factory import LLMClient, LLMError
from tools.search_tool import SearchResult, search_trending_topics

logger = logging.getLogger(__name__)

_MIN_TRENDS = 5
_MAX_TRENDS = 10
# The search tool runs 4 queries x max_search_results each; cap how many
# result lines get quoted in the prompt so it stays bounded.
_MAX_PROMPT_RESULTS = 30

# Field descriptions below are not just docs - they end up in the JSON schema
# that complete_json() injects into the prompt, so they steer the model.


class Trend(BaseModel):
    name: str = Field(..., description="Short, punchy trend name (3-8 words).")
    summary: str = Field(
        ..., description="1-2 sentence explanation of the trend and why it is hot."
    )
    evidence: str = Field(
        default="",
        description="Which headlines/signals from the search data support this.",
    )


class TrendList(BaseModel):
    trends: List[Trend] = Field(..., description="Distinct trending topics.")


_SYSTEM_PROMPT = (
    "You are a senior market intelligence analyst. You are given raw, "
    "real-time web and news search results. Your job is to extract the "
    "genuinely current, distinct trending topics for the given keyword. "
    "Prefer concrete, campaign-able trends over vague themes. Never invent "
    "trends that have no support in the provided search data."
)


class TrendAgent:
    def __init__(self, llm: LLMClient, max_search_results: int = 8) -> None:
        self.llm = llm
        self.max_search_results = max_search_results

    def discover(self, keyword: str) -> List[Trend]:
        """Return 5-10 trending topics for the keyword."""
        results = search_trending_topics(keyword, self.max_search_results)
        if not results:
            raise RuntimeError(
                f"No live search results could be retrieved for keyword "
                f"{keyword!r}. Check your network connection and try again."
            )

        search_block = "\n".join(
            r.to_prompt_line() for r in results[:_MAX_PROMPT_RESULTS]
        )
        user_prompt = (
            f"Keyword: {keyword!r}\n\n"
            f"Real-time search results:\n{search_block}\n\n"
            f"Extract between {_MIN_TRENDS} and {_MAX_TRENDS} distinct trending "
            "topics grounded in this data. Each trend must cite its supporting "
            "evidence from the results above."
        )

        try:
            parsed = self.llm.complete_json(_SYSTEM_PROMPT, user_prompt, TrendList)
            trends = [t for t in parsed.trends if t.name.strip()]
            if trends:
                logger.info("Extracted %d trends for %r", len(trends), keyword)
                return trends[:_MAX_TRENDS]
            logger.warning("LLM returned an empty trend list, using fallback")
        except LLMError as exc:
            logger.error("Trend extraction failed (%s), using headline fallback", exc)

        return self._fallback_from_headlines(results)

    @staticmethod
    def _fallback_from_headlines(results: List[SearchResult]) -> List[Trend]:
        # If the LLM is down we can still promote raw headlines to trend
        # candidates - the pipeline degrades instead of dying here.
        fallback = [
            Trend(
                name=r.title[:80],
                summary=r.snippet[:240] or "Signal surfaced from live search.",
                evidence=r.url,
            )
            for r in results[:_MAX_TRENDS]
            if r.title.strip()
        ]
        if not fallback:
            raise RuntimeError("Unable to derive any trends from search data.")
        return fallback
