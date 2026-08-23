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
from llm_factory import LLMClient
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
        for noisy in ("httpx", "httpx2", "openai", "urllib3", "primp", "ddgs"):
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
        help='domain keyword to research, e.g. "food", "fintech", "gaming"',
    )
    parser.add_argument(
        "--company",
        required=True,
        help='target company for brand-fit evaluation, e.g. "Mastercard"',
    )
    parser.add_argument(
        "--output",
        default="campaign_report.pdf",
        help="output PDF file path",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="debug logging, including HTTP client logs",
    )
    return parser.parse_args(argv)


def run(keyword: str, company: str, output: str) -> str:
    settings = get_settings()
    logger.info(
        "Starting pipeline | keyword=%r company=%r provider=%s model=%s",
        keyword,
        company,
        settings.model_provider,
        settings.model_name,
    )
    started = time.perf_counter()

    llm = LLMClient(settings)

    logger.info("Step 1/4 - discovering live trends for %r ...", keyword)
    trend_agent = TrendAgent(llm, max_search_results=settings.max_search_results)
    trends = trend_agent.discover(keyword)
    logger.info("Discovered %d candidate trends.", len(trends))

    logger.info("Step 2/4 - selecting top 3 topics for %s ...", company)
    topics = EvaluatorAgent(llm).select_top_topics(company, trends)

    logger.info("Step 3/4 - generating campaign plans ...")
    plans = CampaignAgent(llm).generate_all(company, keyword, topics)

    logger.info("Step 4/4 - rendering PDF report ...")
    path = generate_pdf_report(output, company, keyword, topics, plans)

    logger.info("Done in %.1fs - report saved to %s", time.perf_counter() - started, path)
    return path


def main(argv: list[str] | None = None) -> int:
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