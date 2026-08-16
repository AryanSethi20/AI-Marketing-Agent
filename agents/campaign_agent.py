"""Campaign Generation Agent.

For each selected topic, produces a complete, schema-validated marketing
campaign plan covering: Target Audience, Core Message, Channel Strategy,
KPIs, and Brand Synergy.
"""

from __future__ import annotations

import logging
from typing import List

from pydantic import BaseModel, Field

from agents.evaluator_agent import SelectedTopic
from llm_factory import LLMClient, LLMError

logger = logging.getLogger(__name__)


class ChannelPlay(BaseModel):
    """A single channel within the channel strategy."""

    channel: str = Field(..., description="e.g. 'Instagram & TikTok', 'CTV', 'Email'.")
    tactic: str = Field(..., description="Concrete activation tactic on this channel.")


class CampaignPlan(BaseModel):
    """Full campaign plan for one topic (one PDF page)."""

    topic_name: str = Field(..., description="The trend/topic this plan activates.")
    campaign_title: str = Field(..., description="Creative campaign name/tagline.")
    target_audience: str = Field(
        ..., description="Primary segments, demographics, and psychographics."
    )
    core_message: str = Field(
        ..., description="The single-minded proposition of the campaign."
    )
    channel_strategy: List[ChannelPlay] = Field(
        ..., min_length=3, max_length=6, description="3–6 channel activations."
    )
    kpis: List[str] = Field(
        ..., min_length=3, max_length=6, description="3–6 measurable KPIs with targets."
    )
    brand_synergy: str = Field(
        ...,
        description=(
            "How the campaign reinforces the company's brand identity, "
            "strategy, and product ecosystem."
        ),
    )


_SYSTEM_PROMPT = (
    "You are an award-winning marketing strategist. You design sharp, "
    "actionable campaign plans that a brand team could brief tomorrow. "
    "Be specific: name real channel behaviours, quantify KPI targets, and "
    "tie everything back to the company's actual brand and products. "
    "Keep every field concise enough to fit on a single report page."
)


class CampaignAgent:
    """Generates one :class:`CampaignPlan` per selected topic."""

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def generate_plan(
        self, company: str, keyword: str, topic: SelectedTopic
    ) -> CampaignPlan:
        """Generate a full campaign plan for a single topic.

        Falls back to a clearly-labelled skeleton plan if the LLM fails, so a
        partial provider outage still yields a complete 3-page report.
        """
        user_prompt = (
            f"Company: {company}\n"
            f"Industry keyword: {keyword}\n\n"
            f"Trending topic: {topic.name}\n"
            f"Topic summary: {topic.summary}\n"
            f"Why it fits the brand: {topic.rationale}\n\n"
            "Create a complete marketing campaign plan for this company to "
            "activate this trend. Include: a creative campaign title, target "
            "audience, core message, a 3–6 item channel strategy (channel + "
            "tactic), 3–6 KPIs with concrete numeric targets, and a brand "
            "synergy statement."
        )

        try:
            plan = self.llm.complete_json(_SYSTEM_PROMPT, user_prompt, CampaignPlan)
            plan.topic_name = plan.topic_name or topic.name
            logger.info("Generated campaign plan: %s", plan.campaign_title)
            return plan
        except LLMError as exc:
            logger.error(
                "Campaign generation failed for %r (%s); using skeleton plan.",
                topic.name,
                exc,
            )
            return self._skeleton_plan(company, topic)

    def generate_all(
        self, company: str, keyword: str, topics: List[SelectedTopic]
    ) -> List[CampaignPlan]:
        """Generate plans for every selected topic, preserving order."""
        return [self.generate_plan(company, keyword, t) for t in topics]

    @staticmethod
    def _skeleton_plan(company: str, topic: SelectedTopic) -> CampaignPlan:
        """Minimal deterministic plan used only when the LLM is unavailable."""
        return CampaignPlan(
            topic_name=topic.name,
            campaign_title=f"{company} x {topic.name}",
            target_audience=(
                "Core customers and prospects most engaged with this trend "
                "(to be refined — generated in degraded mode)."
            ),
            core_message=topic.summary,
            channel_strategy=[
                ChannelPlay(channel="Social", tactic="Short-form video series."),
                ChannelPlay(channel="Owned media", tactic="Hero landing page + blog."),
                ChannelPlay(channel="Partnerships", tactic="Co-branded activation."),
            ],
            kpis=[
                "Reach: 5M impressions in 8 weeks",
                "Engagement rate: >4% on social",
                "Conversion: +10% campaign-attributed signups",
            ],
            brand_synergy=topic.rationale,
        )
