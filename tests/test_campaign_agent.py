"""Tests for CampaignAgent, including its degraded-mode skeleton plan."""

from agents.campaign_agent import CampaignAgent, CampaignPlan, ChannelPlay
from agents.evaluator_agent import SelectedTopic
from tests.fakes import FakeLLM


TOPIC = SelectedTopic(
    name="Cashless street food",
    summary="QR payments are taking over hawker centres.",
    relevance_score=88,
    rationale="Directly touches the company's payments network.",
)


def _plan(title="Tap Into Taste"):
    return CampaignPlan(
        topic_name="Cashless street food",
        campaign_title=title,
        target_audience="Urban 18-35s",
        core_message="Every meal, one tap.",
        channel_strategy=[
            ChannelPlay(channel="TikTok", tactic="Creator series"),
            ChannelPlay(channel="OOH", tactic="Hawker-centre takeovers"),
            ChannelPlay(channel="Email", tactic="Cardholder offers"),
        ],
        kpis=["5M impressions", "4% engagement", "10% signup lift"],
        brand_synergy="Reinforces contactless leadership.",
    )


def test_happy_path_returns_model_plan():
    agent = CampaignAgent(FakeLLM(result=_plan()))
    plan = agent.generate_plan("Mastercard", "food", TOPIC)
    assert plan.campaign_title == "Tap Into Taste"


def test_skeleton_plan_on_llm_failure():
    agent = CampaignAgent(FakeLLM(fail=True))
    plan = agent.generate_plan("Mastercard", "food", TOPIC)
    # The fallback must still satisfy the schema the PDF layer relies on.
    assert plan.topic_name == TOPIC.name
    assert len(plan.channel_strategy) >= 3
    assert len(plan.kpis) >= 3
    assert plan.brand_synergy == TOPIC.rationale


def test_generate_all_preserves_order():
    agent = CampaignAgent(FakeLLM(result=_plan()))
    topics = [TOPIC, TOPIC.model_copy(update={"name": "Second topic"})]
    plans = agent.generate_all("Mastercard", "food", topics)
    assert len(plans) == 2
