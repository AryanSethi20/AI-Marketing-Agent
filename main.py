"""AI Marketing Agent — CLI entrypoint.

Pipeline:
    1. TrendAgent      — live DuckDuckGo search → structured trend extraction.
    2. EvaluatorAgent  — brand-fit scoring → top 3 topics for the company.
    3. CampaignAgent   — full campaign plan per topic (Pydantic-validated).
    4. PDF generator   — modern 3-page report, one topic per page.

Usage:
    python main.py --keyword "gaming" --company "Mastercard" --output "campaign.pdf"
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

from dotenv import load_dotenv

load_dotenv()

from agents.campaign_agent import CampaignAgent
from agents.evaluator_agent import EvaluatorAgent
from agents.trend_agent import TrendAgent
from config import get_settings
from llm_factory import get_llm_client
from utils.pdf_generator import generate_pdf_report

logger = logging.getLogger("marketing_agent")

def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )
    
    if not verbose:
        for noisy in ("httpx", "openai", "urllib3", "primp"):
            logging.getLogger(noisy).setLevel(logging.WARNING)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="marketing-agent",
        description=(
            "Discover live trends for a keyword, select the 3 most relevant "
            "to a company, and generate a 3-page PDF campaign report."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--keyword",
        required=True,
        help='Domain keyword to research (e.g. "food", "fintech", "gaming").',
    )
    parser.add_argument(
        "--company",
        required=True,
        help='Target company for brand-fit evaluation (e.g. "Mastercard").',
    )
    parser.add_argument(
        "--output",
        default="campaign_report.pdf",
        help="Output PDF file path.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging (including HTTP client logs).",
    )
    return parser.parse_args(argv)


def run(keyword: str, company: str, output: str) -> str:
    """Execute the full agent pipeline and return the written PDF path."""
    settings = get_settings()
    logger.info(
        "Starting pipeline | keyword=%r company=%r provider=%s model=%s",
        keyword,
        company,
        settings.model_provider,
        settings.model_name,
    )
    started = time.perf_counter()

    llm = get_llm_client(settings)

    logger.info("Step 1/4 — Discovering live trends for %r ...", keyword)
    trend_agent = TrendAgent(llm, max_search_results=settings.max_search_results)
    trends = trend_agent.discover(keyword)
    logger.info("Discovered %d candidate trends.", len(trends))

    # 2) Relevance evaluation ------------------------------------------------
    logger.info("Step 2/4 — Selecting top 3 topics for %s ...", company)
    evaluator = EvaluatorAgent(llm)
    topics = evaluator.select_top_topics(company, trends)

    # 3) Campaign generation -------------------------------------------------
    logger.info("Step 3/4 — Generating campaign plans ...")
    campaign_agent = CampaignAgent(llm)
    plans = campaign_agent.generate_all(company, keyword, topics)

    # 4) PDF rendering ---------------------------------------------------------
    logger.info("Step 4/4 — Rendering PDF report ...")
    path = generate_pdf_report(output, company, keyword, topics, plans)

    elapsed = time.perf_counter() - started
    logger.info("Done in %.1fs — report saved to %s", elapsed, path)
    return path


def main(argv: list[str] | None = None) -> int:
    """CLI wrapper with clean exit codes and user-friendly errors."""
    args = _parse_args(argv)
    _configure_logging(args.verbose)
    try:
        run(keyword=args.keyword, company=args.company, output=args.output)
        return 0
    except KeyboardInterrupt:
        logger.error("Interrupted by user.")
        return 130
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        logger.error("Pipeline failed: %s", exc)
        if args.verbose:
            logger.exception("Traceback:")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
