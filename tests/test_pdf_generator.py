"""Tests for the PDF report generator (no LLM or network involved)."""

import pytest

from agents.campaign_agent import CampaignPlan, ChannelPlay
from agents.evaluator_agent import SelectedTopic
from utils.pdf_generator import generate_pdf_report


def _topic(i):
    return SelectedTopic(
        name=f"Topic {i}",
        summary="A trend worth acting on.",
        relevance_score=90 - i,
        rationale="Strong brand fit.",
    )


def _plan(i, **overrides):
    fields = dict(
        topic_name=f"Topic {i}",
        campaign_title=f"Campaign {i}",
        target_audience="Urban 18-35s",
        core_message="One tap, endless possibilities.",
        channel_strategy=[
            ChannelPlay(channel="Social", tactic="Short-form video"),
            ChannelPlay(channel="OOH", tactic="Transit takeovers"),
            ChannelPlay(channel="Email", tactic="Cardholder offers"),
        ],
        kpis=["5M impressions", "4% engagement", "10% lift"],
        brand_synergy="Reinforces the brand promise.",
    )
    fields.update(overrides)
    return CampaignPlan(**fields)


def test_writes_valid_pdf(tmp_path):
    out = tmp_path / "report.pdf"
    topics = [_topic(i) for i in range(3)]
    plans = [_plan(i) for i in range(3)]
    path = generate_pdf_report(str(out), "Acme Corp", "food", topics, plans)
    data = out.read_bytes()
    assert path == str(out)
    assert data.startswith(b"%PDF")
    assert len(data) > 1000


def test_survives_markup_characters_in_llm_text(tmp_path):
    # LLM output with '&' and '<' used to be able to crash ReportLab's
    # Paragraph parser — this locks in the escaping fix.
    out = tmp_path / "report.pdf"
    plan = _plan(
        0,
        campaign_title="Fast & Furious <Deals>",
        core_message="More for less & even more",
    )
    generate_pdf_report(str(out), "M&S", "food & drink", [_topic(0)], [plan])
    assert out.read_bytes().startswith(b"%PDF")


def test_mismatched_lengths_rejected(tmp_path):
    with pytest.raises(ValueError):
        generate_pdf_report(
            str(tmp_path / "x.pdf"), "Acme Corp", "food", [_topic(0)], []
        )
