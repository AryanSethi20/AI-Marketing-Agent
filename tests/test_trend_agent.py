"""Tests for TrendAgent's deterministic headline fallback."""

import pytest

from agents.trend_agent import TrendAgent
from tools.search_tool import SearchResult


def _result(i):
    return SearchResult(
        title=f"Headline {i}",
        snippet=f"Snippet {i}",
        url=f"https://example.com/{i}",
        source="news",
    )


def test_fallback_promotes_headlines():
    results = [_result(i) for i in range(4)]
    trends = TrendAgent._fallback_from_headlines(results)
    assert len(trends) == 4
    assert trends[0].name == "Headline 0"
    assert trends[0].evidence == "https://example.com/0"


def test_fallback_caps_at_max_trends():
    results = [_result(i) for i in range(15)]
    assert len(TrendAgent._fallback_from_headlines(results)) == 10


def test_fallback_skips_blank_titles_and_raises_if_none():
    blank = [SearchResult(title="   ", snippet="x", url="u", source="web")]
    with pytest.raises(RuntimeError):
        TrendAgent._fallback_from_headlines(blank)
