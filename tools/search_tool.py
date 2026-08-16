"""Web search tool backed by DuckDuckGo (no API key required).

Fetches both news and general web results for trend discovery. Uses the
``ddgs`` package (successor of ``duckduckgo-search``) with a compatibility
fallback import, retry/backoff for rate limits, and graceful degradation —
an empty result list never crashes the pipeline.
"""

from __future__ import annotations

import logging
import time
from typing import List

from pydantic import BaseModel, Field

try:  # ddgs is the maintained package name; keep legacy fallback.
    from ddgs import DDGS
except ImportError:  # pragma: no cover - environment-dependent
    from duckduckgo_search import DDGS  # type: ignore[no-redef]

logger = logging.getLogger(__name__)

_MAX_ATTEMPTS = 3
_BACKOFF_SECONDS = 2.0


class SearchResult(BaseModel):
    """A single normalized web/news search result."""

    title: str = Field(..., description="Headline or page title.")
    snippet: str = Field(default="", description="Short body/summary text.")
    url: str = Field(default="", description="Source URL.")
    source: str = Field(default="web", description="'news' or 'web'.")

    def to_prompt_line(self) -> str:
        """Render the result as a compact line for LLM prompts."""
        return f"- [{self.source}] {self.title} — {self.snippet} ({self.url})"


def _run_query(query: str, max_results: int, news: bool) -> List[SearchResult]:
    """Execute one DDG query with retries; returns [] on persistent failure."""
    kind = "news" if news else "web"
    for attempt in range(1, _MAX_ATTEMPTS + 1):
        try:
            with DDGS() as ddgs:
                if news:
                    raw = ddgs.news(query, max_results=max_results)
                else:
                    raw = ddgs.text(query, max_results=max_results)
            results: List[SearchResult] = []
            for item in raw or []:
                results.append(
                    SearchResult(
                        title=str(item.get("title", "")).strip(),
                        snippet=str(
                            item.get("body") or item.get("excerpt") or ""
                        ).strip(),
                        url=str(item.get("href") or item.get("url") or "").strip(),
                        source=kind,
                    )
                )
            logger.info("DDG %s search %r -> %d results", kind, query, len(results))
            return results
        except Exception as exc:  # noqa: BLE001 - search must degrade gracefully
            wait = _BACKOFF_SECONDS * attempt
            logger.warning(
                "DDG %s search failed (attempt %d/%d): %s — retrying in %.1fs",
                kind,
                attempt,
                _MAX_ATTEMPTS,
                exc,
                wait,
            )
            time.sleep(wait)
    logger.error("DDG %s search %r failed after %d attempts.", kind, query, _MAX_ATTEMPTS)
    return []


def search_trending_topics(keyword: str, max_results: int = 8) -> List[SearchResult]:
    """Gather real-time signals about what is trending for ``keyword``.

    Combines news results (freshness) with targeted web queries (breadth),
    then de-duplicates by URL/title.

    Args:
        keyword: Domain keyword, e.g. ``"food"`` or ``"fintech"``.
        max_results: Max results per underlying query.

    Returns:
        A de-duplicated list of :class:`SearchResult` (possibly empty).
    """
    queries_news = [f"{keyword} trends 2026", f"{keyword} industry news"]
    queries_web = [
        f"latest {keyword} trends",
        f"emerging {keyword} consumer trends",
    ]

    combined: List[SearchResult] = []
    for query in queries_news:
        combined.extend(_run_query(query, max_results=max_results, news=True))
    for query in queries_web:
        combined.extend(_run_query(query, max_results=max_results, news=False))

    seen: set[str] = set()
    unique: List[SearchResult] = []
    for result in combined:
        key = result.url or result.title.lower()
        if key and key not in seen:
            seen.add(key)
            unique.append(result)

    logger.info("Collected %d unique search results for %r", len(unique), keyword)
    return unique
