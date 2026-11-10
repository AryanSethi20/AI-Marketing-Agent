"""Tests for EvaluatorAgent's exactly-3-topics guarantee."""

import pytest

from agents.evaluator_agent import EvaluationResult, EvaluatorAgent, SelectedTopic
from agents.trend_agent import Trend
from tests.fakes import FakeLLM


def _trends(n):
    return [
        Trend(name=f"Trend {i}", summary=f"Summary {i}", evidence=f"url-{i}")
        for i in range(1, n + 1)
    ]


def _topic(name, score):
    return SelectedTopic(
        name=name, summary="s", relevance_score=score, rationale="fits the brand"
    )


def test_truncates_when_model_over_returns():
    result = EvaluationResult(
        selected_topics=[_topic(f"T{i}", 90 - i) for i in range(5)]
    )
    agent = EvaluatorAgent(FakeLLM(result=result))
    selected = agent.select_top_topics("Acme Corp", _trends(6))
    assert len(selected) == 3
    assert [s.name for s in selected] == ["T0", "T1", "T2"]


def test_pads_when_model_under_returns():
    result = EvaluationResult(selected_topics=[_topic("Trend 1", 95)])
    agent = EvaluatorAgent(FakeLLM(result=result))
    selected = agent.select_top_topics("Acme Corp", _trends(5))
    assert len(selected) == 3
    # Padding must not duplicate the topic the model already picked.
    names = [s.name for s in selected]
    assert len(set(n.lower() for n in names)) == 3


def test_falls_back_entirely_on_llm_failure():
    agent = EvaluatorAgent(FakeLLM(fail=True))
    selected = agent.select_top_topics("Acme Corp", _trends(4))
    assert len(selected) == 3
    assert all(s.relevance_score == 50 for s in selected)


def test_results_sorted_by_score_desc():
    result = EvaluationResult(
        selected_topics=[_topic("Low", 40), _topic("High", 90), _topic("Mid", 70)]
    )
    agent = EvaluatorAgent(FakeLLM(result=result))
    selected = agent.select_top_topics("Acme Corp", _trends(3))
    assert [s.relevance_score for s in selected] == [90, 70, 40]


def test_empty_trend_list_raises():
    agent = EvaluatorAgent(FakeLLM(fail=True))
    with pytest.raises(ValueError):
        agent.select_top_topics("Acme Corp", [])
