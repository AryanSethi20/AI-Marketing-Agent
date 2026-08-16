"""PDF report generator (ReportLab / platypus).

Renders the final marketing report with a modern visual system:

- Dark-slate section headers with an accent underline.
- Clean sans-serif typography (Helvetica family).
- Structured tables for Channel Strategy and KPIs.
- An explicit ``PageBreak()`` between topics so each topic + campaign plan
  occupies exactly one page (Topic 1 → page 1, Topic 2 → page 2, ...).
"""

from __future__ import annotations

import datetime as _dt
import logging
from typing import List

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    HRFlowable,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from agents.campaign_agent import CampaignPlan
from agents.evaluator_agent import SelectedTopic

logger = logging.getLogger(__name__)

# --- Modern palette ---------------------------------------------------------
SLATE_DARK = colors.HexColor("#1E293B")   # header background
SLATE_MID = colors.HexColor("#475569")    # secondary text
ACCENT = colors.HexColor("#F97316")       # primary accent (warm orange)
ACCENT_SOFT = colors.HexColor("#FFF7ED")  # accent tint for table headers
PAPER = colors.HexColor("#FFFFFF")
LINE_SOFT = colors.HexColor("#E2E8F0")
TEXT_DARK = colors.HexColor("#0F172A")


def _build_styles() -> dict[str, ParagraphStyle]:
    """Create the typographic system used across the report."""
    base = getSampleStyleSheet()
    styles = {
        "title": ParagraphStyle(
            "ReportTitle",
            parent=base["Title"],
            fontName="Helvetica-Bold",
            fontSize=20,
            textColor=PAPER,
            alignment=TA_LEFT,
            leading=24,
        ),
        "subtitle": ParagraphStyle(
            "ReportSubtitle",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=10,
            textColor=colors.HexColor("#CBD5E1"),
            leading=14,
        ),
        "topic_kicker": ParagraphStyle(
            "TopicKicker",
            parent=base["Normal"],
            fontName="Helvetica-Bold",
            fontSize=9,
            textColor=ACCENT,
            leading=12,
            spaceAfter=2,
        ),
        "topic_title": ParagraphStyle(
            "TopicTitle",
            parent=base["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=17,
            textColor=TEXT_DARK,
            leading=21,
            spaceAfter=2,
        ),
        "campaign_title": ParagraphStyle(
            "CampaignTitle",
            parent=base["Normal"],
            fontName="Helvetica-BoldOblique",
            fontSize=11.5,
            textColor=SLATE_MID,
            leading=15,
            spaceAfter=6,
        ),
        "section": ParagraphStyle(
            "SectionHeading",
            parent=base["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=10.5,
            textColor=SLATE_DARK,
            leading=13,
            spaceBefore=8,
            spaceAfter=3,
        ),
        "body": ParagraphStyle(
            "Body",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=9.5,
            textColor=TEXT_DARK,
            leading=13,
        ),
        "table_head": ParagraphStyle(
            "TableHead",
            parent=base["Normal"],
            fontName="Helvetica-Bold",
            fontSize=9,
            textColor=SLATE_DARK,
            leading=12,
        ),
        "table_body": ParagraphStyle(
            "TableBody",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=9,
            textColor=TEXT_DARK,
            leading=12,
        ),
    }
    return styles


def _header_banner(company: str, keyword: str, styles: dict) -> Table:
    """Dark-slate banner shown at the top of every topic page."""
    today = _dt.date.today().strftime("%d %B %Y")
    inner = [
        [Paragraph(f"{company} — AI Marketing Trend Report", styles["title"])],
        [
            Paragraph(
                f"Keyword focus: <b>{keyword.title()}</b> &nbsp;•&nbsp; "
                f"Generated {today} &nbsp;•&nbsp; Powered by live web trend analysis",
                styles["subtitle"],
            )
        ],
    ]
    banner = Table(inner, colWidths=[170 * mm])
    banner.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), SLATE_DARK),
                ("LEFTPADDING", (0, 0), (-1, -1), 12),
                ("RIGHTPADDING", (0, 0), (-1, -1), 12),
                ("TOPPADDING", (0, 0), (0, 0), 10),
                ("BOTTOMPADDING", (0, -1), (-1, -1), 10),
                ("LINEBELOW", (0, -1), (-1, -1), 2.5, ACCENT),
            ]
        )
    )
    return banner


def _section(label: str, styles: dict) -> List:
    """Section heading with a soft rule underneath."""
    return [
        Paragraph(label.upper(), styles["section"]),
        HRFlowable(width="100%", thickness=0.6, color=LINE_SOFT, spaceAfter=4),
    ]


def _channel_table(plan: CampaignPlan, styles: dict) -> Table:
    rows = [
        [
            Paragraph("Channel", styles["table_head"]),
            Paragraph("Activation Tactic", styles["table_head"]),
        ]
    ]
    for play in plan.channel_strategy:
        rows.append(
            [
                Paragraph(play.channel, styles["table_body"]),
                Paragraph(play.tactic, styles["table_body"]),
            ]
        )
    table = Table(rows, colWidths=[45 * mm, 125 * mm])
    table.setStyle(_grid_style())
    return table


def _kpi_table(plan: CampaignPlan, styles: dict) -> Table:
    rows = [
        [
            Paragraph("#", styles["table_head"]),
            Paragraph("Key Performance Indicator", styles["table_head"]),
        ]
    ]
    for i, kpi in enumerate(plan.kpis, start=1):
        rows.append(
            [
                Paragraph(str(i), styles["table_body"]),
                Paragraph(kpi, styles["table_body"]),
            ]
        )
    table = Table(rows, colWidths=[10 * mm, 160 * mm])
    table.setStyle(_grid_style())
    return table


def _grid_style() -> TableStyle:
    return TableStyle(
        [
            ("BACKGROUND", (0, 0), (-1, 0), ACCENT_SOFT),
            ("LINEBELOW", (0, 0), (-1, 0), 1, ACCENT),
            ("GRID", (0, 1), (-1, -1), 0.4, LINE_SOFT),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]
    )


def _topic_page(
    index: int,
    company: str,
    keyword: str,
    topic: SelectedTopic,
    plan: CampaignPlan,
    styles: dict,
) -> List:
    """Assemble all flowables for one topic's page."""
    flow: List = [
        _header_banner(company, keyword, styles),
        Spacer(1, 8 * mm),
        Paragraph(
            f"TOPIC {index} OF 3 &nbsp;•&nbsp; RELEVANCE SCORE "
            f"{topic.relevance_score}/100",
            styles["topic_kicker"],
        ),
        Paragraph(plan.topic_name or topic.name, styles["topic_title"]),
        Paragraph(f"Campaign: “{plan.campaign_title}”", styles["campaign_title"]),
        Paragraph(topic.summary, styles["body"]),
        Spacer(1, 2 * mm),
    ]

    flow += _section("Why this fits the brand", styles)
    flow.append(Paragraph(topic.rationale, styles["body"]))

    flow += _section("Target audience", styles)
    flow.append(Paragraph(plan.target_audience, styles["body"]))

    flow += _section("Core message", styles)
    flow.append(Paragraph(plan.core_message, styles["body"]))

    flow += _section("Channel strategy", styles)
    flow.append(_channel_table(plan, styles))

    flow += _section("Key performance indicators", styles)
    flow.append(_kpi_table(plan, styles))

    flow += _section("Brand synergy", styles)
    flow.append(Paragraph(plan.brand_synergy, styles["body"]))
    return flow


def _footer(canvas, doc) -> None:  # noqa: ANN001 - reportlab callback signature
    """Page footer with rule, brand mark, and page number."""
    canvas.saveState()
    width, _ = A4
    canvas.setStrokeColor(LINE_SOFT)
    canvas.setLineWidth(0.6)
    canvas.line(20 * mm, 14 * mm, width - 20 * mm, 14 * mm)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(SLATE_MID)
    canvas.drawString(20 * mm, 9.5 * mm, "AI Marketing Agent — Confidential Draft")
    canvas.drawRightString(width - 20 * mm, 9.5 * mm, f"Page {doc.page} of 3")
    canvas.restoreState()


def generate_pdf_report(
    output_path: str,
    company: str,
    keyword: str,
    topics: List[SelectedTopic],
    plans: List[CampaignPlan],
) -> str:
    """Render the final report: exactly one page per topic, 3 pages total.

    Args:
        output_path: Destination ``.pdf`` path.
        company: Target company name.
        keyword: The searched keyword.
        topics: The 3 selected topics (ordered).
        plans: Matching campaign plans (same order/length as ``topics``).

    Returns:
        The output path, once written.

    Raises:
        ValueError: If ``topics`` and ``plans`` lengths differ.
    """
    if len(topics) != len(plans):
        raise ValueError("topics and plans must have identical lengths.")

    styles = _build_styles()
    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        leftMargin=20 * mm,
        rightMargin=20 * mm,
        topMargin=16 * mm,
        bottomMargin=20 * mm,
        title=f"{company} Marketing Trend Report — {keyword}",
        author="AI Marketing Agent",
    )

    story: List = []
    for i, (topic, plan) in enumerate(zip(topics, plans), start=1):
        story.extend(_topic_page(i, company, keyword, topic, plan, styles))
        if i < len(topics):
            story.append(PageBreak())  # hard break: one topic per page

    doc.build(story, onFirstPage=_footer, onLaterPages=_footer)
    logger.info("PDF report written to %s", output_path)
    return output_path
