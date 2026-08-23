"""DuckDuckGo search for trend discovery. No API key needed."""

from __future__ import annotations

import logging
import time
from typing import List

from pydantic import BaseModel

try:
    from ddgs import DDGS
except ImportError:  # older installs still ship the legacy package name
    from duckduckgo_search import DDGS  # type: ignore[no-redef]

logger = logging.getLogger(__name__)

_MAX_ATTEMPTS = 3
_BACKOFF_SECONDS = 2.0


class SearchResult(BaseModel):
    title: str
    snippet: str = ""
    url: str = ""
    source: str = "web"  # "news" or "web"

    def to_prompt_line(self) -> str:
        return f"- [{self.source}] {self.title} — {self.snippet} ({self.url})"


def _run_query(query: str, max_results: int, news: bool) -> List[SearchResult]:
    """One DDG query with retries. Returns [] if it keeps failing - the
    caller decides whether an empty result set is fatal."""
    kind = "news" if news else "web"
    for attempt in range(1, _MAX_ATTEMPTS + 1):
        try:
            with DDGS() as ddgs:
                if news:
                    raw = ddgs.news(query, max_results=max_results)
                else:
                    raw = ddgs.text(query, max_results=max_results)
            results = [
                SearchResult(
                    title=str(item.get("title", "")).strip(),
                    snippet=str(item.get("body") or item.get("excerpt") or "").strip(),
                    url=str(item.get("href") or item.get("url") or "").strip(),
                    source=kind,
                )
                for item in raw or []
            ]
            logger.info("DDG %s search %r -> %d results", kind, query, len(results))
            return results
        except Exception as exc:  # noqa: BLE001 - search must degrade gracefully
            wait = _BACKOFF_SECONDS * attempt
            logger.warning(
                "DDG %s search failed (attempt %d/%d): %s - retrying in %.1fs",
                kind, attempt, _MAX_ATTEMPTS, exc, wait,
            )
            time.sleep(wait)
    logger.error("DDG %s search %r failed after %d attempts", kind, query, _MAX_ATTEMPTS)
    return []


def search_trending_topics(keyword: str, max_results: int = 8) -> List[SearchResult]:
    """Combine news results (freshness) with web results (breadth) and
    de-duplicate by URL/title."""
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
