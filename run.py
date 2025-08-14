import argparse, os
import logging
from src.crew_run import run_crew
from src.tools.pipeline_tool_impl import run_pipeline
from src.tools.portfolio_generator import run_portfolio_comparison

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--start", type=str, default="2020-01-01")
    p.add_argument("--end", type=str, default="2025-08-12")
    p.add_argument("--no-llm", action="store_true", help="Bypass CrewAI and run the pipeline tool directly.")
    p.add_argument(
        "--mock-data",
        action="store_true",
        help="Use mock data instead of real market data.",
    )
    p.add_argument(
        "--compare-portfolios",
        action="store_true",
        help="Run multiple portfolio tests and generate comparison report.",
    )
    p.add_argument(
        "--num-portfolios",
        type=int,
        default=5,
        help="Number of portfolios to test (default: 5)",
    )
    args = p.parse_args()

    logger.info(
        f"Arguments: start={args.start}, end={args.end}, no_llm={args.no_llm}, mock_data={args.mock_data}, compare_portfolios={args.compare_portfolios}, num_portfolios={args.num_portfolios}"
    )

    if args.compare_portfolios:
        try:
            logger.info("Running portfolio comparison...")
            results = run_portfolio_comparison(
                start_date=args.start,
                end_date=args.end,
                num_portfolios=args.num_portfolios,
                use_real_data=not args.mock_data,
            )
            print(
                f"Portfolio comparison completed. Generated {len(results)} portfolio reports."
            )
            print("Check reports/portfolio_comparison.md for detailed comparison.")
        except Exception as e:
            logger.error(f"Portfolio comparison failed: {e}")
            import traceback

            logger.error(f"Traceback: {traceback.format_exc()}")
            raise
    elif args.no_llm:
        try:
            logger.info("Running pipeline directly...")
            result = run_pipeline(
                args.start, args.end, use_real_data=not args.mock_data
            )
            print(result)
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            import traceback

            logger.error(f"Traceback: {traceback.format_exc()}")
            raise
    else:
        out = run_crew(args.start, args.end, use_llm=True)
        try:
            print(out)  # may be dict or string depending on version
        except Exception:
            pass
        print("Done. See reports/ for artifacts.")
