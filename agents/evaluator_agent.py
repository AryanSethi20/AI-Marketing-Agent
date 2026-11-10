"""Scores trends against the company's brand and picks the top 3."""

from __future__ import annotations

import logging
from typing import List

from pydantic import BaseModel, Field

from agents.trend_agent import Trend
from llm_factory import LLMClient, LLMError

logger = logging.getLogger(__name__)

TOP_N = 3


class SelectedTopic(BaseModel):
    name: str = Field(..., description="Trend name (kept from the input list).")
    summary: str = Field(..., description="Short description of the trend.")
    relevance_score: int = Field(
        ..., ge=1, le=100, description="Brand-fit score from 1 (poor) to 100."
    )
    rationale: str = Field(
        ...,
        description=(
            "2-3 sentences on why this trend fits the company's brand, "
            "strategy, and products."
        ),
    )


class EvaluationResult(BaseModel):
    selected_topics: List[SelectedTopic] = Field(
        ..., description=f"Exactly the top {TOP_N} topics, highest score first."
    )


_SYSTEM_PROMPT = (
    "You are a brand strategy director. You evaluate market trends for fit "
    "against a specific company's brand identity, strategic priorities, and "
    "product portfolio. Be selective and commercially minded: favour trends "
    "the company can credibly own, activate quickly, and measure."
)


class EvaluatorAgent:
    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def select_top_topics(self, company: str, trends: List[Trend]) -> List[SelectedTopic]:
        """Return exactly TOP_N topics, whatever the model does."""
        if not trends:
            raise ValueError("EvaluatorAgent received an empty trend list.")

        trend_block = "\n".join(
            f"{i}. {t.name}: {t.summary} (evidence: {t.evidence})"
            for i, t in enumerate(trends, start=1)
        )
        user_prompt = (
            f"Company: {company}\n\n"
            "Using your knowledge of this company's brand identity, strategy, "
            "and products, evaluate the candidate trends below.\n\n"
            f"Candidate trends:\n{trend_block}\n\n"
            f"Select EXACTLY the top {TOP_N} trends with the strongest brand "
            "fit and commercial potential for this company. Score each 1-100 "
            "and justify the fit. Order them highest score first."
        )

        try:
            parsed = self.llm.complete_json(
                _SYSTEM_PROMPT, user_prompt, EvaluationResult
            )
            selected = parsed.selected_topics
        except LLMError as exc:
            logger.error("Evaluation failed (%s), falling back to first trends", exc)
            selected = []

        # The report format requires exactly 3 topics, so enforce it in code rather
        # than trusting the prompt: truncate if the model over-returned, pad
        # from the remaining candidates if it under-returned or errored.
        selected = [s for s in selected if s.name.strip()][:TOP_N]
        if len(selected) < TOP_N:
            logger.warning(
                "Evaluator returned %d/%d topics, padding from candidates",
                len(selected), TOP_N,
            )
            chosen_names = {s.name.lower() for s in selected}
            for trend in trends:
                if len(selected) >= TOP_N:
                    break
                if trend.name.lower() in chosen_names:
                    continue
                selected.append(
                    SelectedTopic(
                        name=trend.name,
                        summary=trend.summary,
                        relevance_score=50,
                        rationale=(
                            "Included as a strong candidate from live trend "
                            "discovery pending deeper brand-fit analysis."
                        ),
                    )
                )

        selected.sort(key=lambda s: s.relevance_score, reverse=True)
        logger.info(
            "Selected topics for %s: %s",
            company,
            ", ".join(f"{s.name} ({s.relevance_score})" for s in selected),
        )
        return selected[:TOP_N]
